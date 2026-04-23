// Lanelet geometry + editing operations.
//
// Every function that creates, moves, resizes, or duplicates lanelets goes
// through the NodeRegistry in `./registry.ts`. Boundaries are polylines of
// NodeIds (length >= 2), so:
//   - dragging a junction node moves every connected lanelet coherently
//   - deleting a lanelet leaves junction nodes in place while any other
//     lanelet still needs them (caller gc's the registry afterwards)
//   - topology ("A, B → C") is implicit in shared NodeIds — no extra
//     successor/predecessor tables
//   - interior boundary nodes let the user bend lanelets into curves and
//     turns, mirroring MapToolbox's arbitrary-length `Way.Nodes` lists

import * as THREE from "three";
import type {
  Lanelet,
  LaneletSubType,
  NodeId,
  NodeRegistry,
  Vec3,
} from "./types";
import {
  addNode,
  getNode,
  moveNodes,
  placeCorner,
  ATTACH_DISTANCE,
} from "./registry";
import type { LaneletEndSnap } from "./registry";

// ---------------------------------------------------------------------------
// Subtype helpers
// ---------------------------------------------------------------------------

/**
 * Collect every NodeId owned by a lanelet whose subType differs from
 * `keepSubType`. Used by creation flows to prevent cross-type attach
 * (e.g. a crosswalk accidentally latching onto a road boundary node
 * it happens to be within the snap threshold of).
 */
export function collectNodeIdsOfOtherSubType(
  lanelets: readonly Lanelet[],
  keepSubType: LaneletSubType,
): Set<NodeId> {
  const out = new Set<NodeId>();
  for (const l of lanelets) {
    if (l.subType === keepSubType) continue;
    for (const id of l.leftBoundary)  out.add(id);
    for (const id of l.rightBoundary) out.add(id);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Pure geometry helpers
// ---------------------------------------------------------------------------

/** Perpendicular unit vector in the horizontal XZ plane pointing to the LEFT
 *  of travel (start → end), Y-up frame. */
export function leftPerpXZ(start: Vec3, end: Vec3): [number, number] {
  const dx = end[0] - start[0];
  const dz = end[2] - start[2];
  const len = Math.hypot(dx, dz);
  if (len < 1e-6) return [0, 0];
  return [dz / len, -dx / len];
}

function midpoint(a: Vec3, b: Vec3): Vec3 {
  return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
}

function mirror(pivot: Vec3, from: Vec3): Vec3 {
  return [
    2 * pivot[0] - from[0],
    2 * pivot[1] - from[1],
    2 * pivot[2] - from[2],
  ];
}

function lerp(a: Vec3, b: Vec3, t: number): Vec3 {
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ];
}

/**
 * Cheap ray-down-through-points Y sample for surface snapping.
 *
 * `pointsMesh` can be either a single `THREE.Points` (old single-buffer
 * layout) or a `THREE.Group` of `Points` chunks — after the cloud was
 * spatially tiled we pass the parent group here. `intersectObject(..., true)`
 * walks the group's descendants, and each chunk's bounding sphere gives
 * a free early-out so we only scan the tile(s) the downward ray actually
 * passes through.
 */
export function sampleSurfaceY(
  rc: THREE.Raycaster,
  pointsMesh: THREE.Object3D | null,
  x: number,
  z: number,
  topY: number,
  fallbackY: number
): number {
  if (!pointsMesh) return fallbackY;
  rc.set(new THREE.Vector3(x, topY, z), new THREE.Vector3(0, -1, 0));
  const hits = rc.intersectObject(pointsMesh, true);
  return hits.length > 0 ? hits[0].point.y : fallbackY;
}

// ---------------------------------------------------------------------------
// Lanelet creation
// ---------------------------------------------------------------------------

export interface CreateLaneletParams {
  reg: NodeRegistry;
  nextNodeId: { current: number };
  laneletId: number;
  /** Click 1 — origin for the centerline (surface-snapped by caller). */
  centerStart: Vec3;
  /** Click 2 — terminus for the centerline (surface-snapped). */
  centerEnd: Vec3;
  width: number;
  /** Per-corner surface-snap; returns the Y to use at each (x,z). */
  snapY?: (x: number, z: number, fallbackY: number) => number;
  attachDistance?: number;
  /** Pair-attach: adopt an existing lanelet's end NodeIds at the start. */
  startOverride?: LaneletEndSnap;
  endOverride?:   LaneletEndSnap;
  /**
   * How many interior (non-endpoint) nodes to place evenly along the
   * centerline between start and end. Interior nodes are what let the user
   * bend the lanelet into curves afterwards. 0 = straight lanelet with
   * only two nodes per boundary; 2 = MapToolbox-style default.
   *
   * Ignored when `interiorCenters` is set.
   */
  interiorCount?: number;
  /**
   * Explicit interior centerline waypoints, in order from start toward
   * end (NOT including the endpoints themselves). When supplied, the
   * lanelet's boundary polylines have `2 + interiorCenters.length` pairs
   * and each interior pair is placed perpendicular to the local spine
   * tangent at the given waypoint. Use this to build connector lanelets
   * (e.g. `joinLanelets`) with a specific turn shape.
   */
  interiorCenters?: Vec3[];
  /**
   * What kind of lanelet this is — "road" (default) or "crosswalk". Chosen
   * by the drawing tool, not after the fact. Crosswalks get a different
   * visualisation (zebra stripes, no direction arrow) downstream.
   */
  subType?: LaneletSubType;
  /**
   * NodeIds the attach logic should NOT consider when placing endpoint
   * corners. The main use case is keeping crosswalks from latching
   * onto road boundary nodes (and vice-versa): pass in every NodeId
   * owned by lanelets of the *other* subType and they become invisible
   * to `placeCorner`'s nearest-node search.
   */
  reservedNodeIds?: ReadonlySet<NodeId>;
}

/**
 * Build a lanelet end-to-end.
 *
 * Endpoints either come from a pair-attach override (reusing an existing
 * lanelet's end NodeIds) or from `placeCorner` (attach-snap within
 * `attachDistance` or new). Interior nodes are always allocated fresh —
 * we don't attach them because that would create weird unexpected
 * junctions with far-away lanelets just because a midpoint happens to be
 * close by.
 */
export function createLanelet(
  params: CreateLaneletParams
): { reg: NodeRegistry; lanelet: Lanelet } {
  const {
    reg: reg0,
    nextNodeId,
    laneletId,
    centerStart,
    centerEnd,
    width,
    snapY,
    attachDistance = ATTACH_DISTANCE,
    startOverride,
    endOverride,
    interiorCount = 2,
    interiorCenters,
    subType = "road",
    reservedNodeIds,
  } = params;

  // Axis: when an end is overridden we snap the relevant endpoint to the
  // adopted end's centerpoint so the interior interpolation is honest.
  const cs: Vec3 = startOverride ? startOverride.center : centerStart;
  const ce: Vec3 = endOverride   ? endOverride.center   : centerEnd;

  const half = width / 2;

  // Centerline control points for the new lanelet — starts at `cs`, ends
  // at `ce`, with interior waypoints either supplied explicitly (connector
  // lanelets) or auto-spaced along the straight `cs → ce` line.
  const centers: Vec3[] = interiorCenters
    ? [cs, ...interiorCenters, ce]
    : (() => {
        const n = 2 + Math.max(0, interiorCount);
        const arr: Vec3[] = new Array(n);
        for (let i = 0; i < n; i++) {
          const t = n === 1 ? 0 : i / (n - 1);
          arr[i] = lerp(cs, ce, t);
        }
        return arr;
      })();
  const n = centers.length;

  // Per-index boundary positions. The local left-perpendicular at index i
  // comes from the prev→next centerline tangent so curved connectors get
  // a cross-section that follows the spine (no twist on bends).
  const leftRaw:  Vec3[] = new Array(n);
  const rightRaw: Vec3[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const prev = centers[Math.max(0, i - 1)];
    const next = centers[Math.min(n - 1, i + 1)];
    const [lx, lz] = leftPerpXZ(prev, next);
    const c = centers[i];
    leftRaw[i]  = [c[0] + lx * half, c[1], c[2] + lz * half];
    rightRaw[i] = [c[0] - lx * half, c[1], c[2] - lz * half];
  }

  // Surface-snap Y at every position (fallback: the raw Y that came from
  // the lerp, which itself is lerped from the caller's two click Ys).
  const snapC = (p: Vec3): Vec3 =>
    snapY ? [p[0], snapY(p[0], p[2], p[1]), p[2]] : p;
  const leftPos  = leftRaw.map(snapC);
  const rightPos = rightRaw.map(snapC);

  // Allocate / attach NodeIds. placeCorner for endpoints (attach allowed
  // → picks up nearby existing nodes), addNode for interior (always fresh
  // to avoid surprise junctions at midpoints).
  // `taken` is passed to placeCorner as its exclusion set so the same
  // node can't be picked twice for this lanelet. We pre-seed it with
  // `reservedNodeIds` (e.g. nodes owned by lanelets of a different
  // subType) so cross-type attach never happens.
  const taken = new Set<NodeId>();
  if (reservedNodeIds) {
    for (const id of reservedNodeIds) taken.add(id);
  }
  let reg = reg0;

  const place = (pos: Vec3): NodeId => {
    const { reg: r2, id } = placeCorner(
      reg, pos, nextNodeId, attachDistance, taken
    );
    reg = r2;
    taken.add(id);
    return id;
  };
  const fresh = (pos: Vec3): NodeId => {
    const { reg: r2, id } = addNode(reg, pos, nextNodeId);
    reg = r2;
    taken.add(id);
    return id;
  };

  const leftIds:  NodeId[] = new Array(n);
  const rightIds: NodeId[] = new Array(n);

  // Start endpoint (index 0)
  if (startOverride) {
    leftIds[0]  = startOverride.leftId;
    rightIds[0] = startOverride.rightId;
    taken.add(leftIds[0]);
    taken.add(rightIds[0]);
  } else {
    leftIds[0]  = place(leftPos[0]);
    rightIds[0] = place(rightPos[0]);
  }

  // Interior (indices 1 .. n-2)
  for (let i = 1; i < n - 1; i++) {
    leftIds[i]  = fresh(leftPos[i]);
    rightIds[i] = fresh(rightPos[i]);
  }

  // End endpoint (index n-1)
  if (endOverride) {
    leftIds[n - 1]  = endOverride.leftId;
    rightIds[n - 1] = endOverride.rightId;
    taken.add(leftIds[n - 1]);
    taken.add(rightIds[n - 1]);
  } else {
    leftIds[n - 1]  = place(leftPos[n - 1]);
    rightIds[n - 1] = place(rightPos[n - 1]);
  }

  const lanelet: Lanelet = {
    id: laneletId,
    leftBoundary:  leftIds,
    rightBoundary: rightIds,
    width,
    subType,
    turnDirection: null,
  };

  return { reg, lanelet };
}

// ---------------------------------------------------------------------------
// Editing: enforce rectangular shape
// ---------------------------------------------------------------------------

/**
 * Re-lay every interior boundary node of `lanelet` onto the straight
 * line between its current start and end centers, keeping each pair
 * perpendicular to that axis at half-width apart.
 *
 * This is the one-shot straightener used both when the user toggles
 * `straight` on in the Properties panel and as a post-step after
 * endpoint drags on already-straight lanelets (so the shape can't
 * drift away from rectangular).
 *
 * Endpoint NodeIds are NEVER touched — they may be shared with
 * neighboring lanelets and moving them would yank those neighbors.
 * Interior NodeIds are always allocated fresh by `createLanelet`, so
 * repositioning them is always safe.
 */
export function straightenLanelet(
  reg: NodeRegistry,
  lanelet: Lanelet
): { reg: NodeRegistry; lanelet: Lanelet } {
  const n = lanelet.leftBoundary.length;
  if (n <= 2) return { reg, lanelet };

  const ls = getNode(reg, lanelet.leftBoundary[0]);
  const rs = getNode(reg, lanelet.rightBoundary[0]);
  const le = getNode(reg, lanelet.leftBoundary[n - 1]);
  const re = getNode(reg, lanelet.rightBoundary[n - 1]);

  const cs: Vec3 = [(ls[0] + rs[0]) * 0.5, (ls[1] + rs[1]) * 0.5, (ls[2] + rs[2]) * 0.5];
  const ce: Vec3 = [(le[0] + re[0]) * 0.5, (le[1] + re[1]) * 0.5, (le[2] + re[2]) * 0.5];

  // Use the current endpoint widths (they can legitimately differ if
  // the user widened only one end previously). Interior pairs get a
  // linearly interpolated width per index.
  const widthXZ = (l: Vec3, r: Vec3) =>
    Math.hypot(l[0] - r[0], l[2] - r[2]);
  const wStart = widthXZ(ls, rs);
  const wEnd   = widthXZ(le, re);

  const [lpx, lpz] = leftPerpXZ(cs, ce);
  if (lpx === 0 && lpz === 0) {
    // Start ≈ end, nothing meaningful to straighten.
    return { reg, lanelet };
  }

  const updates: Record<NodeId, Vec3> = {};
  for (let i = 1; i < n - 1; i++) {
    const t = i / (n - 1);
    const cx = cs[0] + (ce[0] - cs[0]) * t;
    const cy = cs[1] + (ce[1] - cs[1]) * t;
    const cz = cs[2] + (ce[2] - cs[2]) * t;
    const half = (wStart + (wEnd - wStart) * t) * 0.5;
    updates[lanelet.leftBoundary[i]]  = [cx + lpx * half, cy, cz + lpz * half];
    updates[lanelet.rightBoundary[i]] = [cx - lpx * half, cy, cz - lpz * half];
  }

  return { reg: moveNodes(reg, updates), lanelet };
}

// ---------------------------------------------------------------------------
// Editing: move a centerline node at index i
// ---------------------------------------------------------------------------

/**
 * Move the boundary pair at `index` so its midpoint becomes `newCenter`.
 *
 * Two modes, chosen by whether the index sits at a boundary of the lanelet:
 *
 *  - Endpoint (index 0 or last): TRANSLATE both left[index] and right[index]
 *    by the same delta. Endpoints are the spots that can be shared with a
 *    neighbor lanelet (pair-attach / junction), and translating keeps the
 *    shared NodeIds coherent — drag one lanelet's end and its connected
 *    neighbor follows.
 *
 *  - Interior (0 < index < last): RE-ANCHOR the pair at `newCenter`,
 *    perpendicular to the local centerline tangent derived from the
 *    neighboring control indices, at the pair's current XZ width. That's
 *    what stops the ribbon from visibly changing width when you curve
 *    the lanelet with the magenta squares: if we just translated, the
 *    pair would stop being perpendicular to the new tangent and read
 *    locally as a twist / pinch. Interior control nodes are always fresh
 *    per lanelet (createLanelet calls addNode for them), so re-anchoring
 *    never breaks a junction.
 *
 * Falls back to plain translation when the local tangent is degenerate
 * (zero-length neighbor segment), so the pair can't collapse.
 */
export function moveLaneletNodeAtIndex(
  reg: NodeRegistry,
  lanelet: Lanelet,
  index: number,
  newCenter: Vec3
): { reg: NodeRegistry; lanelet: Lanelet } {
  const n = lanelet.leftBoundary.length;
  const leftId  = lanelet.leftBoundary[index];
  const rightId = lanelet.rightBoundary[index];
  if (leftId === undefined || rightId === undefined) {
    return { reg, lanelet };
  }

  const leftOld  = getNode(reg, leftId);
  const rightOld = getNode(reg, rightId);

  const isInterior = index > 0 && index < n - 1;

  // Rect lock: interior handles are disabled, endpoint drags translate
  // the endpoint and then re-distribute every interior pair onto the
  // straight start→end axis so the shape remains a clean rectangle.
  if (lanelet.straight) {
    if (isInterior) {
      return { reg, lanelet };
    }
    const oldCx = (leftOld[0]  + rightOld[0])  * 0.5;
    const oldCy = (leftOld[1]  + rightOld[1])  * 0.5;
    const oldCz = (leftOld[2]  + rightOld[2])  * 0.5;
    const dx = newCenter[0] - oldCx;
    const dy = newCenter[1] - oldCy;
    const dz = newCenter[2] - oldCz;
    const regMoved = moveNodes(reg, {
      [leftId]:  [leftOld[0]  + dx, leftOld[1]  + dy, leftOld[2]  + dz],
      [rightId]: [rightOld[0] + dx, rightOld[1] + dy, rightOld[2] + dz],
    });
    return straightenLanelet(regMoved, lanelet);
  }

  if (isInterior) {
    // Local centerline tangent from the immediate neighbors in XZ.
    const lp = getNode(reg, lanelet.leftBoundary[index - 1]);
    const rp = getNode(reg, lanelet.rightBoundary[index - 1]);
    const ln = getNode(reg, lanelet.leftBoundary[index + 1]);
    const rn = getNode(reg, lanelet.rightBoundary[index + 1]);
    const prevCx = (lp[0] + rp[0]) * 0.5;
    const prevCz = (lp[2] + rp[2]) * 0.5;
    const nextCx = (ln[0] + rn[0]) * 0.5;
    const nextCz = (ln[2] + rn[2]) * 0.5;
    const tx = nextCx - prevCx;
    const tz = nextCz - prevCz;
    const tl = Math.hypot(tx, tz);

    if (tl > 1e-6) {
      // Current pair width (XZ), preserved across the re-anchor.
      const wx = leftOld[0] - rightOld[0];
      const wz = leftOld[2] - rightOld[2];
      const width = Math.hypot(wx, wz);
      // Left perpendicular of the tangent in XZ (Y up).
      const px =  tz / tl;
      const pz = -tx / tl;
      const half = width * 0.5;
      return {
        reg: moveNodes(reg, {
          [leftId]:  [newCenter[0] + px * half, newCenter[1], newCenter[2] + pz * half],
          [rightId]: [newCenter[0] - px * half, newCenter[1], newCenter[2] - pz * half],
        }),
        lanelet,
      };
    }
    // Degenerate tangent — fall through to the translate branch.
  }

  // Endpoint (or degenerate-interior) case: translate both nodes equally
  // so any shared junction NodeIds move together with their neighbors.
  const oldCx = (leftOld[0]  + rightOld[0])  * 0.5;
  const oldCy = (leftOld[1]  + rightOld[1])  * 0.5;
  const oldCz = (leftOld[2]  + rightOld[2])  * 0.5;

  const dx = newCenter[0] - oldCx;
  const dy = newCenter[1] - oldCy;
  const dz = newCenter[2] - oldCz;

  return {
    reg: moveNodes(reg, {
      [leftId]:  [leftOld[0]  + dx, leftOld[1]  + dy, leftOld[2]  + dz],
      [rightId]: [rightOld[0] + dx, rightOld[1] + dy, rightOld[2] + dz],
    }),
    lanelet,
  };
}

/**
 * Convenience for the common "drag the very start" / "drag the very end"
 * case, kept for compatibility with existing callers.
 */
export function moveLaneletEndpoint(
  reg: NodeRegistry,
  lanelet: Lanelet,
  which: "start" | "end",
  newCenter: Vec3
): { reg: NodeRegistry; lanelet: Lanelet } {
  const index = which === "start" ? 0 : lanelet.leftBoundary.length - 1;
  return moveLaneletNodeAtIndex(reg, lanelet, index, newCenter);
}

// ---------------------------------------------------------------------------
// Editing: surface snap (all nodes)
// ---------------------------------------------------------------------------

export function snapLaneletToSurface(
  reg: NodeRegistry,
  lanelet: Lanelet,
  snapY: (x: number, z: number, fallbackY: number) => number
): { reg: NodeRegistry; lanelet: Lanelet } {
  const updates: Record<NodeId, Vec3> = {};
  const snap = (id: NodeId) => {
    const p = getNode(reg, id);
    updates[id] = [p[0], snapY(p[0], p[2], p[1]), p[2]];
  };
  for (const id of lanelet.leftBoundary)  snap(id);
  for (const id of lanelet.rightBoundary) snap(id);
  return {
    reg: moveNodes(reg, updates),
    lanelet,
  };
}

// ---------------------------------------------------------------------------
// Editing: resize width
// ---------------------------------------------------------------------------

/**
 * Widen / narrow a lanelet by reseating each boundary pair at `newWidth`
 * around its current centerline index-point.
 *
 * We MOVE the existing boundary NodeIds to their new positions rather than
 * allocating fresh ones. Consequence: any NodeId shared with a neighboring
 * lanelet (i.e. a junction) stays shared — the neighbor's corresponding
 * corner is pulled along, and the junction remains intact. This is what
 * the user expects when widening lane B that connects to lane A: they
 * stay connected; the end of A just widens to match.
 *
 * The local perpendicular at each index is derived from the centerline's
 * tangent (previous → next midpoint), so curves widen cleanly along the
 * spine direction instead of around the chord.
 *
 * `_nextNodeId` is kept in the signature for API compatibility; no new
 * nodes are allocated here.
 */
export function resizeLaneletWidth(
  reg: NodeRegistry,
  _nextNodeId: { current: number },
  lanelet: Lanelet,
  newWidth: number
): { reg: NodeRegistry; lanelet: Lanelet } {
  const n = lanelet.leftBoundary.length;
  const half = newWidth / 2;

  // Current centerline (index-by-index midpoint of the two boundary nodes).
  const center: Vec3[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const l = getNode(reg, lanelet.leftBoundary[i]);
    const r = getNode(reg, lanelet.rightBoundary[i]);
    center[i] = midpoint(l, r);
  }

  // Per-index tangent → left-perpendicular. Endpoints use their single
  // adjacent segment; interior indices average the two neighbor segments
  // via the prev→next midpoint direction.
  const tangentLeftPerp = (i: number): [number, number] => {
    const a = center[Math.max(0, i - 1)];
    const b = center[Math.min(n - 1, i + 1)];
    return leftPerpXZ(a, b);
  };

  // Build a single atomic position update — one per NodeId. If both
  // boundaries somehow alias the same NodeId (shouldn't happen, but be
  // safe), the last write wins and that's still a sane outcome.
  const updates: Record<NodeId, Vec3> = {};
  for (let i = 0; i < n; i++) {
    const [lx, lz] = tangentLeftPerp(i);
    const c = center[i];
    const leftId  = lanelet.leftBoundary[i];
    const rightId = lanelet.rightBoundary[i];
    updates[leftId]  = [c[0] + lx * half, c[1], c[2] + lz * half];
    updates[rightId] = [c[0] - lx * half, c[1], c[2] - lz * half];
  }

  return {
    reg: moveNodes(reg, updates),
    lanelet: { ...lanelet, width: newWidth },
  };
}

// ---------------------------------------------------------------------------
// Connector lanelets — "join" A's end to B's start
// ---------------------------------------------------------------------------

export type JointType = "straight" | "left" | "right";

export interface JoinLaneletsParams {
  reg: NodeRegistry;
  nextNodeId: { current: number };
  laneletId: number;
  /** Connector starts at `from`'s end (last boundary pair). */
  from: Lanelet;
  /** Connector ends at `to`'s start (first boundary pair). */
  to: Lanelet;
  /** Turn shape for the connector's interior waypoints. */
  type: JointType;
  /** Width for interior nodes. Defaults to the mean of from/to widths. */
  width?: number;
  snapY?: (x: number, z: number, fallbackY: number) => number;
}

/**
 * Build a connector lanelet running from the END of `from` to the START
 * of `to`. The two ends of the connector adopt the existing corner
 * NodeIds of `from`/`to` as a pair — same pair-attach mechanic as
 * two-click creation — so the three lanelets are structurally joined.
 *
 * The shape is determined by the type AND by the actual tangent
 * directions at `from`'s end and `to`'s start. A `left`/`right`
 * connector leaves `from` along its forward heading and arrives at
 * `to` along `to`'s forward heading — producing a natural L-turn like
 * a road intersection connector, not a perpendicular bulge off the
 * chord. `straight` ignores tangents and draws a direct ribbon.
 *
 * `type` is also stored on the new lanelet as `turnDirection`, so the
 * metadata matches the shape when exporting to Lanelet2 OSM.
 */
export function joinLanelets(
  params: JoinLaneletsParams
): { reg: NodeRegistry; lanelet: Lanelet } {
  const { reg, nextNodeId, laneletId, from, to, type, snapY } = params;

  // Pair-attach snap for FROM's end.
  const fLast = from.leftBoundary.length - 1;
  const fLeftId  = from.leftBoundary[fLast];
  const fRightId = from.rightBoundary[fLast];
  const fLeftP   = getNode(reg, fLeftId);
  const fRightP  = getNode(reg, fRightId);
  const csCenter: Vec3 = midpoint(fLeftP, fRightP);
  const startOverride: LaneletEndSnap = {
    laneletId: from.id,
    end: "end",
    center: csCenter,
    leftId: fLeftId,
    rightId: fRightId,
    leftPos: fLeftP,
    rightPos: fRightP,
  };

  // Pair-attach snap for TO's start.
  const tLeftId  = to.leftBoundary[0];
  const tRightId = to.rightBoundary[0];
  const tLeftP   = getNode(reg, tLeftId);
  const tRightP  = getNode(reg, tRightId);
  const ceCenter: Vec3 = midpoint(tLeftP, tRightP);
  const endOverride: LaneletEndSnap = {
    laneletId: to.id,
    end: "start",
    center: ceCenter,
    leftId: tLeftId,
    rightId: tRightId,
    leftPos: tLeftP,
    rightPos: tRightP,
  };

  // Tangent directions:
  //   dirA = forward direction at A's end (last centerline segment of A)
  //   dirB = forward direction entering B (first centerline segment of B)
  const dirA = lastSegmentTangentXZ(reg, from);
  const dirB = firstSegmentTangentXZ(reg, to);

  const interior = connectorInteriorCenters(
    csCenter, ceCenter, type, dirA, dirB
  );

  const width =
    params.width !== undefined ? params.width : (from.width + to.width) / 2;

  const { reg: regAfter, lanelet } = createLanelet({
    reg,
    nextNodeId,
    laneletId,
    centerStart: csCenter,
    centerEnd:   ceCenter,
    width,
    snapY,
    attachDistance: ATTACH_DISTANCE,
    startOverride,
    endOverride,
    interiorCenters: interior,
  });

  // Persist the user's shape choice as Lanelet2 turn metadata.
  const turnDirection: Lanelet["turnDirection"] =
    type === "straight" ? "straight" :
    type === "left"     ? "left"     :
    "right";

  return {
    reg: regAfter,
    lanelet: { ...lanelet, turnDirection },
  };
}

// Unit vector along A's last centerline segment (forward exit direction).
// Returns [0,0] if the lanelet is degenerate.
function lastSegmentTangentXZ(
  reg: NodeRegistry,
  l: Lanelet
): [number, number] {
  const n = l.leftBoundary.length;
  if (n < 2) return [0, 0];
  const aL = getNode(reg, l.leftBoundary[n - 2]);
  const aR = getNode(reg, l.rightBoundary[n - 2]);
  const bL = getNode(reg, l.leftBoundary[n - 1]);
  const bR = getNode(reg, l.rightBoundary[n - 1]);
  const dx = (bL[0] + bR[0]) * 0.5 - (aL[0] + aR[0]) * 0.5;
  const dz = (bL[2] + bR[2]) * 0.5 - (aL[2] + aR[2]) * 0.5;
  const d  = Math.hypot(dx, dz);
  return d < 1e-6 ? [0, 0] : [dx / d, dz / d];
}

// Unit vector along B's first centerline segment (forward entry direction).
function firstSegmentTangentXZ(
  reg: NodeRegistry,
  l: Lanelet
): [number, number] {
  if (l.leftBoundary.length < 2) return [0, 0];
  const aL = getNode(reg, l.leftBoundary[0]);
  const aR = getNode(reg, l.rightBoundary[0]);
  const bL = getNode(reg, l.leftBoundary[1]);
  const bR = getNode(reg, l.rightBoundary[1]);
  const dx = (bL[0] + bR[0]) * 0.5 - (aL[0] + aR[0]) * 0.5;
  const dz = (bL[2] + bR[2]) * 0.5 - (aL[2] + aR[2]) * 0.5;
  const d  = Math.hypot(dx, dz);
  return d < 1e-6 ? [0, 0] : [dx / d, dz / d];
}

// Evaluate a cubic Bezier curve at parameter t.
function cubicBezier(
  t: number, p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3
): Vec3 {
  const u   = 1 - t;
  const uu  = u * u;
  const uuu = uu * u;
  const tt  = t * t;
  const ttt = tt * t;
  return [
    uuu * p0[0] + 3 * uu * t * p1[0] + 3 * u * tt * p2[0] + ttt * p3[0],
    uuu * p0[1] + 3 * uu * t * p1[1] + 3 * u * tt * p2[1] + ttt * p3[1],
    uuu * p0[2] + 3 * uu * t * p1[2] + 3 * u * tt * p2[2] + ttt * p3[2],
  ];
}

/**
 * Interior centerline waypoints for a connector.
 *
 * For `left`/`right`: we build a cubic Bezier whose control handles
 * extend from cs along dirA and from ce back along dirB (classic
 * "tangent-matching" Bezier). Sampling it at t = 1/4, 2/4, 3/4 gives 3
 * interior waypoints; combined with the endpoints that's 5 control
 * nodes per boundary — enough for the render-time Catmull-Rom spline
 * to produce a visually smooth arc that leaves A along A's heading and
 * enters B along B's heading.
 *
 * For `straight`: ignore tangents, lay 3 interior points on the direct
 * line so a straight connector stays perfectly straight even when A
 * and B aren't collinear.
 *
 * If either tangent degenerates (e.g. a 2-node lanelet with zero-length
 * last segment after surface snap), fall back to the direct A→B
 * direction for that side so the curve stays well-formed.
 */
function connectorInteriorCenters(
  cs: Vec3,
  ce: Vec3,
  type: JointType,
  dirA: [number, number],
  dirB: [number, number]
): Vec3[] {
  const straightInterior: Vec3[] = [
    lerp(cs, ce, 1 / 4),
    lerp(cs, ce, 2 / 4),
    lerp(cs, ce, 3 / 4),
  ];
  if (type === "straight") return straightInterior;

  const dx = ce[0] - cs[0];
  const dz = ce[2] - cs[2];
  const d  = Math.hypot(dx, dz);
  if (d < 1e-6) return straightInterior;

  const hasA = dirA[0] !== 0 || dirA[1] !== 0;
  const hasB = dirB[0] !== 0 || dirB[1] !== 0;
  const fallback: [number, number] = [dx / d, dz / d];
  const dA: [number, number] = hasA ? dirA : fallback;
  const dB: [number, number] = hasB ? dirB : fallback;

  // Bezier handle length — 45% of the gap gives a curvature that looks
  // like a road corner, not too wide, not too tight.
  const h = d * 0.45;

  const b1: Vec3 = [cs[0] + dA[0] * h, cs[1], cs[2] + dA[1] * h];
  const b2: Vec3 = [ce[0] - dB[0] * h, ce[1], ce[2] - dB[1] * h];

  // 3 interior samples from the Bezier. Catmull-Rom at render time will
  // pass through all of them and fair out the curve further.
  return [
    cubicBezier(0.25, cs, b1, b2, ce),
    cubicBezier(0.50, cs, b1, b2, ce),
    cubicBezier(0.75, cs, b1, b2, ce),
  ];
}

// ---------------------------------------------------------------------------
// Editing: translate the whole lanelet
// ---------------------------------------------------------------------------

/**
 * Shift every boundary NodeId by `delta`. Deduped so a NodeId referenced
 * by both sides (pathological) still moves exactly once.
 *
 * Semantic note: shared NodeIds at a junction move too, which pulls any
 * neighboring lanelet's connected corner along with this drag. That is
 * deliberate — the junction stays a junction. The UI flow for "move just
 * this lanelet, detaching it" should handle detachment *before* calling
 * this (e.g. duplicate-out the shared NodeIds into private ones first).
 */
export function translateLanelet(
  reg: NodeRegistry,
  lanelet: Lanelet,
  delta: Vec3
): { reg: NodeRegistry; lanelet: Lanelet } {
  const seen = new Set<NodeId>();
  const updates: Record<NodeId, Vec3> = {};
  const translateId = (id: NodeId) => {
    if (seen.has(id)) return;
    seen.add(id);
    const p = getNode(reg, id);
    updates[id] = [p[0] + delta[0], p[1] + delta[1], p[2] + delta[2]];
  };
  for (const id of lanelet.leftBoundary)  translateId(id);
  for (const id of lanelet.rightBoundary) translateId(id);
  return { reg: moveNodes(reg, updates), lanelet };
}

/**
 * Set every boundary NodeId to `snapshot[id] + delta`. Absolute / stateless
 * variant of `translateLanelet` used by the live drag loop — each pointer
 * move applies `delta` against the pointer-down snapshot, which keeps the
 * result idempotent (no accumulated floating-point drift) even if the drag
 * re-runs many times per second.
 */
export function applyLaneletTranslationFromSnapshot(
  reg: NodeRegistry,
  snapshot: Record<NodeId, Vec3>,
  delta: Vec3
): NodeRegistry {
  const updates: Record<NodeId, Vec3> = {};
  for (const [k, p] of Object.entries(snapshot)) {
    const id = Number(k);
    updates[id] = [p[0] + delta[0], p[1] + delta[1], p[2] + delta[2]];
  }
  return moveNodes(reg, updates);
}

// ---------------------------------------------------------------------------
// Editing: reverse direction of travel
// ---------------------------------------------------------------------------

export function reverseLanelet(l: Lanelet): Lanelet {
  const leftRev  = [...l.rightBoundary].reverse();
  const rightRev = [...l.leftBoundary].reverse();
  return {
    ...l,
    leftBoundary:  leftRev,
    rightBoundary: rightRev,
    leftBoundarySubType:  l.rightBoundarySubType,
    rightBoundarySubType: l.leftBoundarySubType,
  };
}

// ---------------------------------------------------------------------------
// Neighbor duplication — MapToolbox DuplicateLeft / DuplicateRight
// ---------------------------------------------------------------------------

/**
 * New lanelet sits flush against the source's `side`. Same node count per
 * boundary as the source so the two lanelets align index-for-index.
 *   - Inner boundary (the shared edge): reuses the source's NodeIds by
 *     reference, marked "dashed" on both → lane change allowed.
 *   - Outer boundary: each index mirrored across the shared edge
 *     (outer = 2 * shared - opposite). placeCorner so a lanelet slotted
 *     between two existing ones becomes fully connected on both sides.
 */
export interface DuplicateResult {
  reg: NodeRegistry;
  updatedSource: Lanelet;
  neighbor: Omit<Lanelet, "id">;
}

function duplicateImpl(
  reg: NodeRegistry,
  nextNodeId: { current: number },
  src: Lanelet,
  side: "left" | "right",
  attachDistance: number
): DuplicateResult {
  const n = src.leftBoundary.length;
  const pivotIds = side === "left" ? src.leftBoundary  : src.rightBoundary;
  const awayIds  = side === "left" ? src.rightBoundary : src.leftBoundary;

  // Outer boundary positions by mirroring the opposite side across the
  // shared side, per index.
  const outerPos: Vec3[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const pivot = getNode(reg, pivotIds[i]);
    const away  = getNode(reg, awayIds[i]);
    outerPos[i] = mirror(pivot, away);
  }

  // placeCorner lets outer indices auto-attach to existing nodes when the
  // new lanelet slots between neighbors. Taken set prevents collapse onto
  // the shared edge.
  const taken = new Set<NodeId>(pivotIds);

  let reg1 = reg;
  const place = (pos: Vec3): NodeId => {
    const { reg: r2, id } = placeCorner(
      reg1, pos, nextNodeId, attachDistance, taken
    );
    reg1 = r2;
    taken.add(id);
    return id;
  };
  const outerIds: NodeId[] = outerPos.map(place);

  const sharedIds = [...pivotIds];

  let neighbor: Omit<Lanelet, "id">;
  if (side === "left") {
    neighbor = {
      leftBoundary:  outerIds,
      rightBoundary: sharedIds,
      width:         src.width,
      subType:       src.subType,
      turnDirection: src.turnDirection,
      speedLimit:    src.speedLimit,
      leftBoundarySubType:  "solid",
      rightBoundarySubType: "dashed",
    };
  } else {
    neighbor = {
      leftBoundary:  sharedIds,
      rightBoundary: outerIds,
      width:         src.width,
      subType:       src.subType,
      turnDirection: src.turnDirection,
      speedLimit:    src.speedLimit,
      leftBoundarySubType:  "dashed",
      rightBoundarySubType: "solid",
    };
  }

  const updatedSource: Lanelet = {
    ...src,
    leftBoundarySubType:  side === "left"  ? "dashed" : src.leftBoundarySubType,
    rightBoundarySubType: side === "right" ? "dashed" : src.rightBoundarySubType,
  };

  return { reg: reg1, updatedSource, neighbor };
}

export function duplicateLaneletLeft(
  reg: NodeRegistry,
  nextNodeId: { current: number },
  src: Lanelet,
  attachDistance: number = ATTACH_DISTANCE
): DuplicateResult {
  return duplicateImpl(reg, nextNodeId, src, "left", attachDistance);
}

export function duplicateLaneletRight(
  reg: NodeRegistry,
  nextNodeId: { current: number },
  src: Lanelet,
  attachDistance: number = ATTACH_DISTANCE
): DuplicateResult {
  return duplicateImpl(reg, nextNodeId, src, "right", attachDistance);
}

// ---------------------------------------------------------------------------
// Compat re-exports
// ---------------------------------------------------------------------------

export { ATTACH_DISTANCE } from "./registry";
export type { ResolvedLanelet } from "./types";
