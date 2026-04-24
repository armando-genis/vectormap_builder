// Node registry — the backing store for every Lanelet's boundary corners.
//
// Two lanelets that share a junction literally share a NodeId in their
// boundary slots (mirrors MapToolbox's `Way.Nodes` referencing a common
// `Node` object). All operations that create, move, or delete nodes go
// through this module so that invariants hold in one place.
//
// The registry is immutable at the React layer: every mutator returns a new
// `NodeRegistry`. That keeps it safe to put in React state and trivially
// comparable for re-render decisions.

import * as THREE from "three";
import type {
  Lanelet,
  MapNode,
  NodeId,
  NodeRegistry,
  ResolvedLanelet,
  Vec3,
} from "./types";

/**
 * How many dense samples to emit between consecutive control nodes when
 * smoothing a boundary polyline. 12 segments ≈ no visible crease at typical
 * zoom. Higher = smoother + slower; lower = visible facets.
 */
const SMOOTH_SAMPLES_PER_SEGMENT = 12;

/**
 * Smooth a control polyline with centripetal Catmull–Rom. The curve passes
 * exactly through every control point, doesn't cusp on sharp angles, and
 * degenerates gracefully for length-2 inputs (straight line).
 *
 * Output is a packed `Float32Array` of length `3 * (samplesPerSegment *
 * (ctrl.length - 1) + 1)` for ctrl ≥ 2 — one `xyz` triple per sample. The
 * first and last triples coincide with the first and last control nodes,
 * so two connected lanelets' rendered edges meet exactly at the seam.
 *
 * Typed-array output avoids allocating a `[x,y,z]` tuple per sample, which
 * at ~130 samples × 3 arrays × per-lanelet drag frame dominates GC
 * pressure for large maps.
 */
function smoothPolyline(
  ctrl: Vec3[],
  samplesPerSegment: number = SMOOTH_SAMPLES_PER_SEGMENT
): Float32Array {
  if (ctrl.length < 2) {
    const out = new Float32Array(ctrl.length * 3);
    for (let i = 0; i < ctrl.length; i++) {
      out[i * 3    ] = ctrl[i][0];
      out[i * 3 + 1] = ctrl[i][1];
      out[i * 3 + 2] = ctrl[i][2];
    }
    return out;
  }
  if (ctrl.length === 2) {
    const out = new Float32Array(6);
    out[0] = ctrl[0][0]; out[1] = ctrl[0][1]; out[2] = ctrl[0][2];
    out[3] = ctrl[1][0]; out[4] = ctrl[1][1]; out[5] = ctrl[1][2];
    return out;
  }

  const pts = ctrl.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
  const curve = new THREE.CatmullRomCurve3(pts, false, "centripetal", 0.5);

  const total = samplesPerSegment * (ctrl.length - 1);
  const out = new Float32Array((total + 1) * 3);
  const v = new THREE.Vector3();
  for (let i = 0; i <= total; i++) {
    curve.getPoint(i / total, v);
    out[i * 3    ] = v.x;
    out[i * 3 + 1] = v.y;
    out[i * 3 + 2] = v.z;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

export function createRegistry(): NodeRegistry {
  return { nodes: {} };
}

// ---------------------------------------------------------------------------
// Reads
// ---------------------------------------------------------------------------

export function getNode(reg: NodeRegistry, id: NodeId): Vec3 {
  const p = reg.nodes[id];
  if (!p) {
    // Hard error: a dangling NodeId means the registry and a lanelet fell
    // out of sync — any caller rendering or editing is better off crashing
    // than silently painting at [0,0,0].
    throw new Error(`NodeRegistry: unknown NodeId ${id}`);
  }
  return p;
}

export function getNodeOrNull(
  reg: NodeRegistry,
  id: NodeId
): Vec3 | null {
  return reg.nodes[id] ?? null;
}

export function allNodes(reg: NodeRegistry): MapNode[] {
  return Object.entries(reg.nodes).map(([id, position]) => ({
    id: Number(id),
    position,
  }));
}

// ---------------------------------------------------------------------------
// Mutators (return a new registry)
// ---------------------------------------------------------------------------

/**
 * Create a new node at `pos`. Returns the updated registry and the assigned
 * id so the caller can store it on a boundary.
 *
 * `nextIdRef` is an out-of-state ref to keep ids monotonic across React
 * state updates without having to include the counter in the registry
 * object itself.
 */
export function addNode(
  reg: NodeRegistry,
  pos: Vec3,
  nextIdRef: { current: number }
): { reg: NodeRegistry; id: NodeId } {
  const id = nextIdRef.current++;
  return {
    reg: { nodes: { ...reg.nodes, [id]: pos } },
    id,
  };
}

/** Move an existing node; affects every lanelet that references it. */
export function moveNode(
  reg: NodeRegistry,
  id: NodeId,
  pos: Vec3
): NodeRegistry {
  if (!(id in reg.nodes)) return reg;
  return { nodes: { ...reg.nodes, [id]: pos } };
}

/**
 * Batch move: accepts a partial map of `{id: newPos}` and applies them in a
 * single registry update. Cheaper than calling `moveNode` N times when many
 * nodes change at once (resize/duplicate/surface-snap).
 */
export function moveNodes(
  reg: NodeRegistry,
  updates: Record<NodeId, Vec3>
): NodeRegistry {
  const merged: Record<NodeId, Vec3> = { ...reg.nodes };
  for (const [k, v] of Object.entries(updates)) {
    if (k in merged) merged[Number(k)] = v;
  }
  return { nodes: merged };
}

/** Drop nodes from the registry that no lanelet still references. */
export function gcUnused(
  reg: NodeRegistry,
  lanelets: readonly Lanelet[]
): NodeRegistry {
  const keep = new Set<NodeId>();
  for (const l of lanelets) {
    for (const id of l.leftBoundary)  keep.add(id);
    for (const id of l.rightBoundary) keep.add(id);
  }

  const next: Record<NodeId, Vec3> = {};
  for (const [k, v] of Object.entries(reg.nodes)) {
    const id = Number(k);
    if (keep.has(id)) next[id] = v;
  }
  return { nodes: next };
}

// ---------------------------------------------------------------------------
// Attach / snap
// ---------------------------------------------------------------------------

/** MapToolbox-compatible `attachDistance = 1f`. */
export const ATTACH_DISTANCE = 1;

/**
 * Find the NodeId in the registry whose position is nearest `pos` AND within
 * `maxDistance`. `excludeIds` lets callers skip nodes that belong to the
 * lanelet being constructed (so a corner doesn't snap to its own partner).
 *
 * Returns null when nothing qualifies, meaning the caller should create a
 * fresh node.
 */
export function findNearestNode(
  reg: NodeRegistry,
  pos: Vec3,
  maxDistance: number = ATTACH_DISTANCE,
  excludeIds?: ReadonlySet<NodeId>
): NodeId | null {
  let best: NodeId | null = null;
  let bestD = maxDistance;
  for (const [k, p] of Object.entries(reg.nodes)) {
    const id = Number(k);
    if (excludeIds?.has(id)) continue;
    const dx = p[0] - pos[0];
    const dy = p[1] - pos[1];
    const dz = p[2] - pos[2];
    const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (d <= bestD) {
      best = id;
      bestD = d;
    }
  }
  return best;
}

/**
 * "Place a corner at `pos`": either reuse a nearby existing node (attach
 * snap) or create a new one. Returns the id and updated registry.
 *
 * This is the single workhorse every lanelet-creation flow should call so
 * topology is always maximally shared.
 */
export function placeCorner(
  reg: NodeRegistry,
  pos: Vec3,
  nextIdRef: { current: number },
  attachDistance: number = ATTACH_DISTANCE,
  excludeIds?: ReadonlySet<NodeId>
): { reg: NodeRegistry; id: NodeId; attached: boolean } {
  const hit = findNearestNode(reg, pos, attachDistance, excludeIds);
  if (hit !== null) {
    return { reg, id: hit, attached: true };
  }
  const { reg: reg2, id } = addNode(reg, pos, nextIdRef);
  return { reg: reg2, id, attached: false };
}

// ---------------------------------------------------------------------------
// Resolve (read the world in plain Vec3s)
// ---------------------------------------------------------------------------

export function resolveLanelet(
  reg: NodeRegistry,
  l: Lanelet
): ResolvedLanelet {
  // Boundaries are polylines; same length on both sides by invariant.
  const leftPos:  Vec3[] = l.leftBoundary.map((id)  => getNode(reg, id));
  const rightPos: Vec3[] = l.rightBoundary.map((id) => getNode(reg, id));

  // Derived centerline — per-index midpoint of the two boundary polylines.
  // Recomputed every resolve, so moving a boundary node (even via a
  // neighbor's shared-junction drag) immediately updates this lanelet's
  // centerline, arrow, and drag-handle positions without manual sync.
  const n = leftPos.length;
  const centerline: Vec3[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const lp = leftPos[i];
    const rp = rightPos[i];
    centerline[i] = [
      (lp[0] + rp[0]) * 0.5,
      (lp[1] + rp[1]) * 0.5,
      (lp[2] + rp[2]) * 0.5,
    ];
  }

  // Render-time geometry: each boundary is its own Catmull–Rom spline
  // through the STORED NodeIds. That guarantees seams between connected
  // lanelets blend cleanly — both sides pass exactly through the shared
  // junction NodeIds at their first/last control index, so there's no
  // gap at an L-junction where the two lanelets meet at an angle.
  //
  // The "width drifts when I curve the lanelet" issue is handled at drag
  // time (moveLaneletNodeAtIndex in geometry.ts), which re-anchors the
  // interior boundary pair perpendicular to the new local tangent so the
  // control polylines stay close to true perpendicular offsets.
  //
  // centerSmooth is the per-sample midpoint of the two boundary splines,
  // not an independent Catmull-Rom through the centerline control points.
  // That keeps the rendered arrow/centerline visually consistent with the
  // ribbon edges it lives between.
  const leftSmooth  = smoothPolyline(leftPos);
  const rightSmooth = smoothPolyline(rightPos);
  // leftSmooth/rightSmooth are packed Float32Arrays (xyz per sample). The
  // centerline sampling is just a per-channel average — no per-sample tuple
  // allocation needed.
  const flatLen = leftSmooth.length;
  const centerSmooth = new Float32Array(flatLen);
  for (let k = 0; k < flatLen; k++) {
    centerSmooth[k] = (leftSmooth[k] + rightSmooth[k]) * 0.5;
  }

  return {
    ...l,
    leftBoundaryIds:  l.leftBoundary,
    rightBoundaryIds: l.rightBoundary,
    leftBoundary:  leftPos,
    rightBoundary: rightPos,
    centerline,
    leftSmooth,
    rightSmooth,
    centerSmooth,
    centerStart: centerline[0],
    centerEnd:   centerline[n - 1],
  };
}

export function resolveAll(
  reg: NodeRegistry,
  lanelets: readonly Lanelet[]
): ResolvedLanelet[] {
  return lanelets.map((l) => resolveLanelet(reg, l));
}

// ---------------------------------------------------------------------------
// Pair attach — snap to the whole end of an existing lanelet, not per-corner
// ---------------------------------------------------------------------------

/**
 * A description of "the end of an existing lanelet, ready to be reused as
 * the start (or end) of a new one". Both boundary corners are resolved so
 * the caller can adopt them as a unit.
 */
export interface LaneletEndSnap {
  /** Lanelet whose end we're snapping to. */
  laneletId: number;
  /** Which end of that lanelet — "start" or "end" side. */
  end: "start" | "end";
  /** Midpoint of that end (centerStart or centerEnd of the existing lanelet). */
  center: Vec3;
  /** Left-boundary NodeId at that end, relative to the existing lanelet's
   *  travel direction. */
  leftId: NodeId;
  /** Right-boundary NodeId at that end. */
  rightId: NodeId;
  /** Cached positions so the caller doesn't need the registry again. */
  leftPos: Vec3;
  rightPos: Vec3;
}

/**
 * Find the nearest *end* of any lanelet whose centerpoint is within
 * `threshold` of `click`. This drives a MUCH more forgiving snap than the
 * per-corner attach — a slightly-off click still captures BOTH corners as
 * a pair, producing a full-width connection instead of a one-sided kink.
 *
 * Returns null when nothing qualifies, in which case the caller should
 * fall back to per-corner attach.
 *
 * `subType` restricts the search to lanelets of that kind. Crosswalks
 * sit on top of road lanelets (zebra stripes inside a junction, etc.)
 * so their endpoints often fall within the snap threshold of nearby
 * road endpoints even though they're semantically unrelated — without
 * this filter, placing a crosswalk would silently glue it to the road.
 */
export function findNearestLaneletEnd(
  reg: NodeRegistry,
  lanelets: readonly Lanelet[],
  click: Vec3,
  threshold: number,
  subType?: Lanelet["subType"],
): LaneletEndSnap | null {
  let best: LaneletEndSnap | null = null;
  let bestD = threshold;

  for (const l of lanelets) {
    if (subType !== undefined && l.subType !== subType) continue;
    const last = l.leftBoundary.length - 1;
    for (const end of ["start", "end"] as const) {
      const idx = end === "start" ? 0 : last;
      const leftId  = l.leftBoundary[idx];
      const rightId = l.rightBoundary[idx];
      const leftPos  = getNode(reg, leftId);
      const rightPos = getNode(reg, rightId);

      // The "center" of this end is the midpoint of its two boundary
      // nodes — same derivation as resolveLanelet. No stored centerline.
      const cx = (leftPos[0] + rightPos[0]) * 0.5;
      const cy = (leftPos[1] + rightPos[1]) * 0.5;
      const cz = (leftPos[2] + rightPos[2]) * 0.5;

      const dx = cx - click[0];
      const dy = cy - click[1];
      const dz = cz - click[2];
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (d <= bestD) {
        best = {
          laneletId: l.id,
          end,
          center: [cx, cy, cz],
          leftId,
          rightId,
          leftPos,
          rightPos,
        };
        bestD = d;
      }
    }
  }

  return best;
}

// ---------------------------------------------------------------------------
// Junction analysis
// ---------------------------------------------------------------------------

/**
 * Returns a map { NodeId → # of lanelets referencing it }. Any entry with
 * value ≥ 2 is a structural junction and should be highlighted.
 */
export function nodeRefCount(
  lanelets: readonly Lanelet[]
): Record<NodeId, number> {
  const out: Record<NodeId, number> = {};
  const bump = (id: NodeId) => {
    out[id] = (out[id] ?? 0) + 1;
  };
  for (const l of lanelets) {
    for (const id of l.leftBoundary)  bump(id);
    for (const id of l.rightBoundary) bump(id);
  }
  return out;
}

/** Node ids referenced by 2+ lanelets — where one lanelet meets another. */
export function junctionNodeIds(lanelets: readonly Lanelet[]): NodeId[] {
  const counts = nodeRefCount(lanelets);
  const out: NodeId[] = [];
  for (const [k, n] of Object.entries(counts)) {
    if (n >= 2) out.push(Number(k));
  }
  return out;
}
