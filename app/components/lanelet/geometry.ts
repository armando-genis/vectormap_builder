// Lanelet geometry + editing operations.
//
// Every function that creates, moves, resizes, or duplicates lanelets goes
// through the NodeRegistry in `./registry.ts`. Boundary corners are shared
// by reference (NodeId), so:
//   - dragging an endpoint that's part of a junction moves all connected
//     lanelets coherently
//   - deleting a lanelet leaves junction nodes in place while any other
//     lanelet still needs them (caller gc's the registry afterwards)
//   - topology ("A, B → C") is implicit in shared NodeIds — no extra
//     successor/predecessor tables

import * as THREE from "three";
import type {
  Lanelet,
  NodeId,
  NodeRegistry,
  ResolvedLanelet,
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
// Pure geometry helpers (no registry access)
// ---------------------------------------------------------------------------

/**
 * Perpendicular unit vector (in the horizontal XZ plane) pointing to the LEFT
 * of the travel direction start → end, in a three.js right-handed / Y-up frame.
 */
export function leftPerpXZ(start: Vec3, end: Vec3): [number, number] {
  const dx = end[0] - start[0];
  const dz = end[2] - start[2];
  const len = Math.hypot(dx, dz);
  if (len < 1e-6) return [0, 0];
  return [dz / len, -dx / len];
}

export interface LaneletCorners {
  leftStart:  Vec3;
  rightStart: Vec3;
  leftEnd:    Vec3;
  rightEnd:   Vec3;
}

/**
 * Axis-free rectangle corners around the centerline segment (start → end) for
 * a given width. Heights at each corner come from the nearest endpoint; the
 * caller can refine them against the cloud surface afterwards.
 */
export function computeLaneletCorners(
  start: Vec3,
  end: Vec3,
  width: number
): LaneletCorners {
  const [lx, lz] = leftPerpXZ(start, end);
  const half = width / 2;

  const leftStart:  Vec3 = [start[0] + lx * half, start[1], start[2] + lz * half];
  const rightStart: Vec3 = [start[0] - lx * half, start[1], start[2] - lz * half];
  const leftEnd:    Vec3 = [end[0]   + lx * half, end[1],   end[2]   + lz * half];
  const rightEnd:   Vec3 = [end[0]   - lx * half, end[1],   end[2]   - lz * half];

  return { leftStart, rightStart, leftEnd, rightEnd };
}

function midpoint(a: Vec3, b: Vec3): Vec3 {
  return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
}

/** Reflection of `from` across `pivot`: 2*pivot - from. */
function mirror(pivot: Vec3, from: Vec3): Vec3 {
  return [
    2 * pivot[0] - from[0],
    2 * pivot[1] - from[1],
    2 * pivot[2] - from[2],
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
  /** Assigned to the new Lanelet. */
  laneletId: number;
  centerStart: Vec3;
  centerEnd: Vec3;
  width: number;
  /** Per-corner surface-snap; return the Y to use for each corner. */
  snapY?: (x: number, z: number, fallbackY: number) => number;
  attachDistance?: number;

  /**
   * When set, the new lanelet's "start" end does NOT use freshly-computed
   * corners. It reuses the NodeIds / positions described here (typically
   * sourced from `findNearestLaneletEnd`), guaranteeing a full-width pair
   * attach to the existing lanelet's end. The left/right mapping assumes
   * same travel direction as the existing lanelet.
   */
  startOverride?: LaneletEndSnap;
  endOverride?:   LaneletEndSnap;
}

/**
 * Build a lanelet end-to-end: surface-snap corner Ys, attach-snap each corner
 * to a nearby existing node when within `attachDistance` (reusing that node
 * id → structural connection), or create fresh nodes.
 *
 * Returns the updated registry plus the new Lanelet. The laneletId is
 * caller-provided so the caller can keep their own running counter.
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
  } = params;

  // Snap the free centerline ends to the existing end centers so the
  // rectangle's axis passes through them. This matters because downstream
  // corner computations for the non-overridden end use (start → end)
  // direction; keeping the axis honest preserves the user's intended shape.
  const cs: Vec3 = startOverride ? startOverride.center : centerStart;
  const ce: Vec3 = endOverride   ? endOverride.center   : centerEnd;

  const c = computeLaneletCorners(cs, ce, width);

  const snapCorner = (p: Vec3, fbY: number): Vec3 =>
    snapY ? [p[0], snapY(p[0], p[2], fbY), p[2]] : p;

  // Attach each corner to an existing node when within threshold; otherwise
  // allocate a fresh one. Exclude ids already chosen for this lanelet so a
  // zero-length lanelet can't snap all four corners onto the same node.
  const taken = new Set<NodeId>();
  let reg = reg0;

  const place = (pos: Vec3): NodeId => {
    const { reg: r2, id } = placeCorner(
      reg,
      pos,
      nextNodeId,
      attachDistance,
      taken
    );
    reg = r2;
    taken.add(id);
    return id;
  };

  // --- start corners ---
  let lsId: NodeId;
  let rsId: NodeId;
  if (startOverride) {
    // Pair attach: adopt the existing end's corner NodeIds wholesale. No
    // placeCorner call → no accidental duplicate nodes, no distance fuss.
    // The lanelet's "width" at this end becomes whatever the existing
    // lanelet's end had (may differ from the slider — trapezoidal lanelets
    // are fine).
    lsId = startOverride.leftId;
    rsId = startOverride.rightId;
    taken.add(lsId);
    taken.add(rsId);
  } else {
    const leftStart  = snapCorner(c.leftStart,  cs[1]);
    const rightStart = snapCorner(c.rightStart, cs[1]);
    lsId = place(leftStart);
    rsId = place(rightStart);
  }

  // --- end corners ---
  let leId: NodeId;
  let reId: NodeId;
  if (endOverride) {
    leId = endOverride.leftId;
    reId = endOverride.rightId;
    taken.add(leId);
    taken.add(reId);
  } else {
    const leftEnd  = snapCorner(c.leftEnd,  ce[1]);
    const rightEnd = snapCorner(c.rightEnd, ce[1]);
    leId = place(leftEnd);
    reId = place(rightEnd);
  }

  const lanelet: Lanelet = {
    id: laneletId,
    leftBoundary:  [lsId, leId],
    rightBoundary: [rsId, reId],
    width,
    subType: "road",
    turnDirection: null,
  };

  return { reg, lanelet };
}

// ---------------------------------------------------------------------------
// Editing: move / surface-snap
// ---------------------------------------------------------------------------

/**
 * Move the "start" or "end" of a lanelet's centerline — and with it, the two
 * boundary nodes at THAT end, by the same delta. The opposite end is NOT
 * touched, so shared junction nodes on the far side stay put and do not yank
 * the neighbor along (this is exactly how MapToolbox behaves: nodes are
 * independent; moving one moves only that one).
 *
 * Nodes that the moving pair happens to share with other lanelets naturally
 * move in all of them — that's the whole point of sharing. Every connected
 * lanelet's centerline is derived at render time so they follow along
 * without extra bookkeeping.
 *
 * Returns only the updated registry — the Lanelet object itself is
 * unchanged (its NodeIds are; their positions live in the registry).
 *
 * Side-effect: after repeated drags the lanelet may no longer be a perfect
 * rectangle. Use "Resize width" (which reallocates fresh nodes) to snap
 * back to axis-aligned when desired.
 */
export function moveLaneletEndpoint(
  reg: NodeRegistry,
  lanelet: Lanelet,
  which: "start" | "end",
  newCenter: Vec3
): { reg: NodeRegistry; lanelet: Lanelet } {
  const [leftId, rightId]: [NodeId, NodeId] =
    which === "start"
      ? [lanelet.leftBoundary[0], lanelet.rightBoundary[0]]
      : [lanelet.leftBoundary[1], lanelet.rightBoundary[1]];

  const leftOld  = getNode(reg, leftId);
  const rightOld = getNode(reg, rightId);

  // Old center = midpoint of the two boundary nodes at this end.
  const oldCx = (leftOld[0]  + rightOld[0])  * 0.5;
  const oldCy = (leftOld[1]  + rightOld[1])  * 0.5;
  const oldCz = (leftOld[2]  + rightOld[2])  * 0.5;

  const dx = newCenter[0] - oldCx;
  const dy = newCenter[1] - oldCy;
  const dz = newCenter[2] - oldCz;

  const updates: Record<NodeId, Vec3> = {
    [leftId]:  [leftOld[0]  + dx, leftOld[1]  + dy, leftOld[2]  + dz],
    [rightId]: [rightOld[0] + dx, rightOld[1] + dy, rightOld[2] + dz],
  };

  return {
    reg: moveNodes(reg, updates),
    lanelet,
  };
}

/**
 * Re-seat every boundary node of a lanelet onto the cloud surface by
 * down-ray sampling. The centerline is derived from these nodes, so it
 * follows automatically.
 */
export function snapLaneletToSurface(
  reg: NodeRegistry,
  lanelet: Lanelet,
  snapY: (x: number, z: number, fallbackY: number) => number
): { reg: NodeRegistry; lanelet: Lanelet } {
  const updates: Record<NodeId, Vec3> = {};
  for (const id of [
    lanelet.leftBoundary[0],
    lanelet.leftBoundary[1],
    lanelet.rightBoundary[0],
    lanelet.rightBoundary[1],
  ]) {
    const pos = getNode(reg, id);
    updates[id] = [pos[0], snapY(pos[0], pos[2], pos[1]), pos[2]];
  }

  return {
    reg: moveNodes(reg, updates),
    lanelet,
  };
}

// ---------------------------------------------------------------------------
// Editing: resize width
// ---------------------------------------------------------------------------

/**
 * Widen / narrow a lanelet. Because the original boundary nodes may be
 * shared with a neighbor (junction), we don't move them in place — that
 * would warp the neighbor. Instead we **allocate fresh nodes** for the new
 * boundary and detach the lanelet from its neighbors on both sides.
 *
 * Trade-off: simple and predictable. If you want to widen a lanelet chain
 * without breaking junctions, resize each lanelet individually or re-attach
 * after the resize by dragging.
 */
export function resizeLaneletWidth(
  reg: NodeRegistry,
  nextNodeId: { current: number },
  lanelet: Lanelet,
  newWidth: number
): { reg: NodeRegistry; lanelet: Lanelet } {
  // Current centerline is the midpoint of the current boundary pair at each
  // end — same derivation `resolveLanelet` uses for rendering.
  const ls = getNode(reg, lanelet.leftBoundary[0]);
  const le = getNode(reg, lanelet.leftBoundary[1]);
  const rs = getNode(reg, lanelet.rightBoundary[0]);
  const re = getNode(reg, lanelet.rightBoundary[1]);
  const centerStart = midpoint(ls, rs);
  const centerEnd   = midpoint(le, re);

  const c = computeLaneletCorners(centerStart, centerEnd, newWidth);

  let reg1 = reg;
  const alloc = (pos: Vec3): NodeId => {
    const { reg: r2, id } = addNode(reg1, pos, nextNodeId);
    reg1 = r2;
    return id;
  };
  const lsId = alloc(c.leftStart);
  const leId = alloc(c.leftEnd);
  const rsId = alloc(c.rightStart);
  const reId = alloc(c.rightEnd);

  return {
    reg: reg1,
    lanelet: {
      ...lanelet,
      width: newWidth,
      leftBoundary:  [lsId, leId],
      rightBoundary: [rsId, reId],
    },
  };
}

// ---------------------------------------------------------------------------
// Editing: reverse
// ---------------------------------------------------------------------------

/**
 * Flip the direction of travel of a lanelet. "Left" and "right" are defined
 * relative to travel, so we swap the two boundary ways AND reverse each
 * pair. NodeIds are untouched — the nodes don't move, the labels just
 * rearrange; the derived centerline auto-flips at the next resolve.
 */
export function reverseLanelet(l: Lanelet): Lanelet {
  return {
    ...l,
    leftBoundary:  [l.rightBoundary[1], l.rightBoundary[0]],
    rightBoundary: [l.leftBoundary[1],  l.leftBoundary[0]],
    leftBoundarySubType:  l.rightBoundarySubType,
    rightBoundarySubType: l.leftBoundarySubType,
  };
}

// ---------------------------------------------------------------------------
// Neighbor duplication — MapToolbox DuplicateLeft / DuplicateRight
// ---------------------------------------------------------------------------

/**
 * Shape:
 *   - The new lanelet's INNER boundary shares the source's boundary NodeIds
 *     (referentially identical — dragging one drags both).
 *   - The new lanelet's OUTER boundary is the reflection of the source's
 *     opposite boundary across the shared boundary (MapToolbox's
 *     `LineThin.DuplicateNodes`: newPos = 2 * target2 - target1). Those
 *     outer corners are placed through `placeCorner`, so if another
 *     lanelet already has a node there, we reuse it — a new lanelet slotted
 *     between two existing ones becomes fully connected on both sides.
 *   - The shared edge is marked "dashed" on both lanelets (lane change
 *     allowed, per Lanelet2 convention).
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
  const lS = getNode(reg, src.leftBoundary[0]);
  const lE = getNode(reg, src.leftBoundary[1]);
  const rS = getNode(reg, src.rightBoundary[0]);
  const rE = getNode(reg, src.rightBoundary[1]);

  const [sharedStartId, sharedEndId, pivotStart, pivotEnd, awayStart, awayEnd] =
    side === "left"
      ? [src.leftBoundary[0], src.leftBoundary[1], lS, lE, rS, rE]
      : [src.rightBoundary[0], src.rightBoundary[1], rS, rE, lS, lE];

  // Outer boundary positions by mirroring the opposite side across the
  // shared side — same formula as MapToolbox.
  const outerStart = mirror(pivotStart, awayStart);
  const outerEnd   = mirror(pivotEnd,   awayEnd);

  // placeCorner lets new outer corners auto-connect to existing nodes when
  // the new lanelet fits into a pre-existing gap.
  const taken = new Set<NodeId>([sharedStartId, sharedEndId]);

  let reg1 = reg;
  const place = (pos: Vec3): NodeId => {
    const { reg: r2, id } = placeCorner(
      reg1,
      pos,
      nextNodeId,
      attachDistance,
      taken
    );
    reg1 = r2;
    taken.add(id);
    return id;
  };
  const outerStartId = place(outerStart);
  const outerEndId   = place(outerEnd);

  let neighbor: Omit<Lanelet, "id">;
  if (side === "left") {
    neighbor = {
      leftBoundary:  [outerStartId,  outerEndId],
      rightBoundary: [sharedStartId, sharedEndId],
      width:         src.width,
      subType:       src.subType,
      turnDirection: src.turnDirection,
      speedLimit:    src.speedLimit,
      leftBoundarySubType:  "solid",
      rightBoundarySubType: "dashed",
    };
  } else {
    neighbor = {
      leftBoundary:  [sharedStartId, sharedEndId],
      rightBoundary: [outerStartId,  outerEndId],
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
// Compatibility: re-exports so existing imports continue to work.
// ---------------------------------------------------------------------------

export { ATTACH_DISTANCE } from "./registry";
export type { ResolvedLanelet };
