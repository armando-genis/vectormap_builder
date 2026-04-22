// Data model for lanelets.
//
// Mirrors the Autocore MapToolbox model closely enough to export cleanly to
// Lanelet2 OSM XML later:
//   - Node    = 3D point (local_x, ele=Y, local_y=Z)
//   - Way     = ordered list of Node refs
//   - Lanelet = relation { left=Way, right=Way } + tags (type=lanelet,
//               subtype, speed_limit, turn_direction, width)
//
// **Connectivity is structural, not declarative.** Two lanelets that meet at
// a junction literally share the same NodeId in their boundaries — just like
// MapToolbox's Way.Nodes list holds references to shared Node objects. Moving
// a shared node moves it in every referencing lanelet; a "T"-merge where two
// lanelets feed one is simply "three lanelets' corner NodeIds coincide".

export type Vec3 = [number, number, number];

/** Stable identifier for a Node in the `NodeRegistry`. */
export type NodeId = number;

/** Stored in the NodeRegistry — the authoritative position for a Node. */
export interface MapNode {
  id: NodeId;
  position: Vec3;
}

/**
 * The global Node registry for a loaded map. Lanelets reference boundaries by
 * NodeId; the registry holds the actual positions. Think of it as the Lanelet2
 * `<node>` set — one entry per unique corner.
 */
export interface NodeRegistry {
  /** Map from NodeId → position. */
  nodes: Record<NodeId, Vec3>;
}

export type LaneletSubType = "road" | "crosswalk";
export type LaneletTurn    = "straight" | "left" | "right" | null;

/**
 * Style of a boundary line. In Lanelet2 OSM the "lane change allowed" /
 * "forbidden" distinction is encoded on the boundary itself:
 *   - solid  → crossing (lane change) is NOT allowed
 *   - dashed → crossing (lane change) IS  allowed
 *
 * Same convention used by MapToolbox's LineThin.SubType.
 */
export type LineSubType = "solid" | "dashed";

/**
 * Persistent lanelet — pure topology over the NodeRegistry.
 *
 * The geometry of this lanelet lives entirely in the registry: boundaries
 * are NodeId pairs, and the centerline (start midpoint, end midpoint,
 * arrow direction) is **derived** from those boundary nodes' positions on
 * demand. Nothing else stores a copy, so dragging a shared junction node
 * in a neighboring lanelet automatically moves this lanelet's centerline
 * and arrow along with it.
 *
 * Two lanelets with `leftBoundary[1]` equal to some other lanelet's
 * `leftBoundary[0]` are structurally connected (they share a corner) with
 * no extra bookkeeping.
 */
export interface Lanelet {
  id: number;

  /**
   * Left boundary: 2 NodeIds [start-side, end-side].
   * "Left" is relative to the direction of travel
   * (start-side → end-side).
   */
  leftBoundary:  [NodeId, NodeId];
  /** Right boundary: 2 NodeIds [start-side, end-side]. */
  rightBoundary: [NodeId, NodeId];

  /** Lateral width in metres, used as a target by resize and by the default
   *  rectangle placement at creation. Not enforced as an invariant — drag
   *  can make the lanelet trapezoidal. */
  width: number;

  subType:       LaneletSubType;
  turnDirection: LaneletTurn;
  speedLimit?:   number;

  /**
   * Boundary line style — doubles as the lane-change flag:
   * "dashed" means a neighbor lanelet on that side can merge into this one.
   * Default for newly-drawn lanelets is "solid".
   */
  leftBoundarySubType?:  LineSubType;
  rightBoundarySubType?: LineSubType;
}

/**
 * Render-side view of a lanelet: NodeIds replaced with their resolved
 * positions, plus centerline midpoints derived from those positions. This
 * is what the rendering and editing layers actually consume.
 *
 * Keeping both `...Ids` AND `...Boundary` lets UI code identify the backing
 * nodes (e.g. for hover / drag callbacks) while still using raw Vec3s for
 * mesh building.
 */
export interface ResolvedLanelet
  extends Omit<Lanelet, "leftBoundary" | "rightBoundary"> {
  leftBoundary:  [Vec3, Vec3];
  rightBoundary: [Vec3, Vec3];
  leftBoundaryIds:  [NodeId, NodeId];
  rightBoundaryIds: [NodeId, NodeId];

  /** Midpoint of (leftBoundary[0], rightBoundary[0]) — derived, not stored. */
  centerStart: Vec3;
  /** Midpoint of (leftBoundary[1], rightBoundary[1]) — derived, not stored. */
  centerEnd:   Vec3;
}
