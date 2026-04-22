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

/** Cheap ray-down-through-points Y sample for surface snapping. */
export function sampleSurfaceY(
  rc: THREE.Raycaster,
  pointsMesh: THREE.Points | null,
  x: number,
  z: number,
  topY: number,
  fallbackY: number
): number {
  if (!pointsMesh) return fallbackY;
  rc.set(new THREE.Vector3(x, topY, z), new THREE.Vector3(0, -1, 0));
  const hits = rc.intersectObject(pointsMesh, false);
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
   */
  interiorCount?: number;
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
  } = params;

  // Axis: when an end is overridden we snap the relevant endpoint to the
  // adopted end's centerpoint so the interior interpolation is honest.
  const cs: Vec3 = startOverride ? startOverride.center : centerStart;
  const ce: Vec3 = endOverride   ? endOverride.center   : centerEnd;

  const [lx, lz] = leftPerpXZ(cs, ce);
  const half = width / 2;

  // Total boundary length: 2 endpoints + N interior.
  const n = 2 + Math.max(0, interiorCount);

  // Compute raw (unsnapped, unoverridden) positions for every index.
  const leftRaw:  Vec3[] = new Array(n);
  const rightRaw: Vec3[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const t = n === 1 ? 0 : i / (n - 1);
    const c = lerp(cs, ce, t);
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
  const taken = new Set<NodeId>();
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
    subType: "road",
    turnDirection: null,
  };

  return { reg, lanelet };
}

// ---------------------------------------------------------------------------
// Editing: move a centerline node at index i
// ---------------------------------------------------------------------------

/**
 * Translate the boundary pair at `index` so its midpoint becomes `newCenter`.
 * Both left[index] and right[index] move by the same delta — shape stays
 * locally intact, neighbors at other indices untouched.
 *
 * `index` may be 0 (start), last (end), or any interior.
 *
 * Shared nodes move everywhere they're referenced — that's how junctions
 * stay coherent with drags on connected lanelets.
 */
export function moveLaneletNodeAtIndex(
  reg: NodeRegistry,
  lanelet: Lanelet,
  index: number,
  newCenter: Vec3
): { reg: NodeRegistry; lanelet: Lanelet } {
  const leftId  = lanelet.leftBoundary[index];
  const rightId = lanelet.rightBoundary[index];
  if (leftId === undefined || rightId === undefined) {
    return { reg, lanelet };
  }

  const leftOld  = getNode(reg, leftId);
  const rightOld = getNode(reg, rightId);

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
