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
 * Two lanelets whose last boundary NodeId (end side) equals some other
 * lanelet's first boundary NodeId (start side) are structurally connected
 * (they share a corner) with no extra bookkeeping.
 */
export interface Lanelet {
  id: number;

  /**
   * Left boundary as an ordered polyline of NodeIds, `length >= 2`.
   * Index 0 is the start side, last index is the end side, interior
   * indices are shape-controlling nodes the user can drag to bend the
   * lanelet (curves, turns, s-shapes, …) — same model as MapToolbox's
   * `Way.Nodes` list. "Left" is relative to travel direction
   * (start-side → end-side).
   *
   * Invariant: leftBoundary.length === rightBoundary.length. Each index
   * pairs a left node with a right node at the same centerline position.
   */
  leftBoundary:  NodeId[];
  /** Right boundary polyline; same length as `leftBoundary`. */
  rightBoundary: NodeId[];

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

  /**
   * "Rect lock": when true, the lanelet is constrained to a straight
   * rectangle. Only the endpoint drag handles are active and interior
   * control points are re-distributed onto the start→end line after
   * every endpoint move, so the shape can only grow / shrink along
   * its axis. Default (undefined/false) keeps the standard
   * curve-capable editing behaviour.
   */
  straight?: boolean;
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
  /** Left boundary positions, same length & order as the stored NodeIds. */
  leftBoundary:  Vec3[];
  /** Right boundary positions, same length & order. */
  rightBoundary: Vec3[];
  leftBoundaryIds:  NodeId[];
  rightBoundaryIds: NodeId[];

  /**
   * Derived centerline polyline — one point per boundary index, computed
   * as the midpoint of (leftBoundary[i], rightBoundary[i]). Length equals
   * `leftBoundary.length`. This is the "control polyline" — what drag
   * handles clamp to — *not* the rendered shape.
   */
  centerline: Vec3[];

  /**
   * Smooth sampled boundary polylines (centripetal Catmull–Rom through the
   * control boundary nodes). These are what actually get rendered — fill
   * ribbon, boundary lines and centerline — so curves look continuous
   * instead of showing a crease at every interior control node.
   *
   * `leftSmooth` and `rightSmooth` always have the same length; their
   * first sample coincides with the start control node and their last
   * sample coincides with the end control node.
   */
  leftSmooth:   Vec3[];
  rightSmooth:  Vec3[];
  /** Per-index midpoint of (leftSmooth, rightSmooth). */
  centerSmooth: Vec3[];

  /** Shortcut for centerline[0] (and centerSmooth[0]). */
  centerStart: Vec3;
  /** Shortcut for centerline[last] (and centerSmooth[last]). */
  centerEnd:   Vec3;
}
