// Lanelet2 OSM export.
//
// Mirrors the XML layout that Autocore's MapToolbox writes (Lanelet2Map.Save
// + Node/Way/Relation.Save in the reference C# source) so the output can be
// opened back up in MapToolbox, passed to Autoware, or consumed by any
// vanilla Lanelet2 loader.
//
// Shape of the file (simplified):
//
//   <osm generator="VectorMap Builder" version="0.1">
//     <node id="…" lat="0" lon="0">
//       <tag k="ele"     v="…" />
//       <tag k="local_x" v="…" />
//       <tag k="local_y" v="…" />
//     </node>
//     …
//     <way id="…">
//       <nd ref="…" /> …
//       <tag k="type"    v="line_thin" />
//       <tag k="subtype" v="solid" | "dashed" />
//     </way>
//     …
//     <relation id="…">
//       <tag k="type"    v="lanelet" />
//       <tag k="subtype" v="road" | "crosswalk" />
//       <tag k="turn_direction" v="straight|left|right" />   (optional)
//       <tag k="speed_limit"    v="Nkm/h" />                 (optional)
//       <member type="way" ref="…" role="left"  />
//       <member type="way" ref="…" role="right" />
//     </relation>
//     …
//   </osm>
//
// Coordinate frame:
//   Internally the scene is Three.js Y-up (PointCloudViewer.transformZUpToYUp
//   rotates the loaded PCD in). Lanelet2 / Autoware / the original PCD all
//   speak ROS-style Z-up, so we invert that rotation when emitting tag
//   values:
//       local_x =  scene.x
//       local_y = -scene.z
//       ele     =  scene.y
//   That's the exact inverse of the forward transform used at load time.

import type { Lanelet, NodeId, NodeRegistry } from "./types";

/** Mapping used both here and in PointCloudViewer's forward transform. */
function sceneToRobot(p: readonly [number, number, number]): {
  local_x: number;
  local_y: number;
  ele:     number;
} {
  return {
    local_x:  p[0],
    local_y: -p[2],
    ele:      p[1],
  };
}

/**
 * Generators: `version` is bumped when the output format changes in a way a
 * downstream loader might care about. `generator` matches MapToolbox's
 * assembly display name style so files are identifiable in diff tools.
 */
const GENERATOR = "VectorMap Builder";
const VERSION   = "0.1";

/**
 * Round to a stable ~mm precision so the written file doesn't dump 17
 * digits of float noise. 4 decimals ≈ 0.1 mm, plenty for map work and
 * still reproducible across JS runtimes.
 */
const COORD_FIXED = 4;
const fmt = (n: number) => {
  // toFixed produces "-0.0000" for tiny negatives; canonicalise to "0".
  const s = n.toFixed(COORD_FIXED);
  return s === `-${"0".repeat(1)}.${"0".repeat(COORD_FIXED)}` ? `0.${"0".repeat(COORD_FIXED)}` : s;
};

/** XML attribute-safe escaping — the values we emit are numeric or short
 *  enum strings so we only need to guard the meta characters. */
function xmlAttr(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/**
 * Allocate a fresh Way id space that can't collide with any existing
 * NodeId or Lanelet id. We reuse Lanelet.id as the <relation id> and
 * NodeId as the <node id>; each Lanelet gets TWO new way ids (one per
 * boundary), numbered from a counter that starts above every id already
 * in the map.
 */
function nextIdFactory(reg: NodeRegistry, lanelets: readonly Lanelet[]) {
  let max = 0;
  for (const k of Object.keys(reg.nodes)) {
    const n = Number(k);
    if (Number.isFinite(n) && n > max) max = n;
  }
  for (const l of lanelets) {
    if (l.id > max) max = l.id;
  }
  let next = max + 1;
  return () => next++;
}

/**
 * Build the Lanelet2 OSM XML as a string. Deterministic for a given
 * (registry, lanelets) pair: node ids are iterated in numeric order,
 * each lanelet emits exactly two `<way>` elements (left, then right), and
 * relations are iterated in the same order as the input array.
 *
 * Only structurally-complete lanelets (both boundaries have ≥ 2 nodes) are
 * exported — matches MapToolbox's `Valide` gate so an unfinished
 * half-placed lanelet doesn't poison the file.
 */
export function exportToLanelet2Osm(
  registry: NodeRegistry,
  lanelets: readonly Lanelet[]
): string {
  const mkId = nextIdFactory(registry, lanelets);

  // ── 1. collect every NodeId that's actually referenced by a valid lanelet
  //       so we don't emit orphan nodes that snuck past gc.
  const referenced = new Set<NodeId>();
  const valid: Lanelet[] = [];
  for (const l of lanelets) {
    if (l.leftBoundary.length >= 2 && l.rightBoundary.length >= 2) {
      valid.push(l);
      for (const id of l.leftBoundary)  referenced.add(id);
      for (const id of l.rightBoundary) referenced.add(id);
    }
  }

  // ── 2. pre-allocate way ids (left/right) for each lanelet, keyed by
  //       lanelet id so the <relation> section can reference them later.
  const leftWayId:  Record<number, number> = {};
  const rightWayId: Record<number, number> = {};
  for (const l of valid) {
    leftWayId[l.id]  = mkId();
    rightWayId[l.id] = mkId();
  }

  // ── 3. emit the XML. Built as a string array to avoid a DOM dependency
  //       on the client (works identically in SSR and in-browser).
  const out: string[] = [];
  out.push(`<?xml version="1.0" encoding="UTF-8"?>`);
  out.push(
    `<osm generator="${xmlAttr(GENERATOR)}" version="${xmlAttr(VERSION)}">`
  );

  // Nodes — numeric id order for stable diffs.
  const nodeIdsSorted = Array.from(referenced).sort((a, b) => a - b);
  for (const id of nodeIdsSorted) {
    const p = registry.nodes[id];
    if (!p) continue;   // shouldn't happen; skip gracefully.
    const { local_x, local_y, ele } = sceneToRobot(p);
    out.push(`  <node id="${id}" lat="0" lon="0">`);
    out.push(`    <tag k="ele" v="${fmt(ele)}"/>`);
    out.push(`    <tag k="local_x" v="${fmt(local_x)}"/>`);
    out.push(`    <tag k="local_y" v="${fmt(local_y)}"/>`);
    out.push(`  </node>`);
  }

  // Ways — two per lanelet (left, right), each a line_thin with solid /
  // dashed subtype matching our in-memory boundary style. Default solid
  // when unset so downstream loaders don't see a missing subtype tag.
  for (const l of valid) {
    const leftSub  = l.leftBoundarySubType  ?? "solid";
    const rightSub = l.rightBoundarySubType ?? "solid";

    out.push(`  <way id="${leftWayId[l.id]}">`);
    for (const n of l.leftBoundary) out.push(`    <nd ref="${n}"/>`);
    out.push(`    <tag k="type" v="line_thin"/>`);
    out.push(`    <tag k="subtype" v="${xmlAttr(leftSub)}"/>`);
    out.push(`  </way>`);

    out.push(`  <way id="${rightWayId[l.id]}">`);
    for (const n of l.rightBoundary) out.push(`    <nd ref="${n}"/>`);
    out.push(`    <tag k="type" v="line_thin"/>`);
    out.push(`    <tag k="subtype" v="${xmlAttr(rightSub)}"/>`);
    out.push(`  </way>`);
  }

  // Relations — one per lanelet. Tag order matches MapToolbox
  // (type → subtype → turn_direction → speed_limit → members) so a
  // side-by-side diff with a MapToolbox-saved file stays readable.
  for (const l of valid) {
    out.push(`  <relation id="${l.id}">`);
    out.push(`    <tag k="type" v="lanelet"/>`);
    out.push(`    <tag k="subtype" v="${xmlAttr(l.subType)}"/>`);
    if (l.turnDirection) {
      out.push(`    <tag k="turn_direction" v="${xmlAttr(l.turnDirection)}"/>`);
    }
    // Speed_limit is written as "Nkm/h" (MapToolbox's format). Skip for
    // crosswalks — they're pedestrian areas, not travel lanes.
    if (l.speedLimit !== undefined && l.subType !== "crosswalk") {
      out.push(`    <tag k="speed_limit" v="${l.speedLimit}km/h"/>`);
    }
    out.push(`    <member type="way" ref="${leftWayId[l.id]}"  role="left"/>`);
    out.push(`    <member type="way" ref="${rightWayId[l.id]}" role="right"/>`);
    out.push(`  </relation>`);
  }

  out.push(`</osm>`);
  out.push("");   // trailing newline
  return out.join("\n");
}

/**
 * Browser-only helper: turn the XML into a file download. Returns true on
 * success, false when there's nothing to export. Kept in this module so
 * the UI layer doesn't need to know about the DOM blob/URL dance.
 */
export function downloadLanelet2Osm(
  registry: NodeRegistry,
  lanelets: readonly Lanelet[],
  filename: string = "lanelet2_map.osm"
): boolean {
  const xml = exportToLanelet2Osm(registry, lanelets);
  // Header + empty <osm /> = always present, so we gate on "any valid
  // lanelet at all" by counting <relation> lines — cheap & unambiguous.
  if (!xml.includes("<relation ")) return false;

  const blob = new Blob([xml], { type: "application/xml;charset=utf-8" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href     = url;
  a.download = filename;
  a.rel      = "noopener";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  // Give the browser a tick to start the download before revoking.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  return true;
}
