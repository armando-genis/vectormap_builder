"use client";

import { useEffect, useMemo, useRef } from "react";
import { Html, Line } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import type { ResolvedLanelet, Vec3 } from "./types";

// Render-order stack (higher = drawn later = on top)
//   cloud          0 (default)
//   lanelet fill   5
//   centerline     6
//   boundaries    10
//   arrows        11
//   junctions     25
//   edit handles  30+  (always topmost)
const RO_FILL       = 5;
const RO_CENTERLINE = 6;
const RO_LINE       = 10;
const RO_ARROW      = 11;
const RO_DOT        = 20;
const RO_JUNCTION   = 25;
const RO_HANDLE     = 30;

const FILL_COLOR_ROAD       = "#22c55e";
const FILL_COLOR_CROSSWALK  = "#0ea5e9";
const FILL_COLOR_SELECTED   = "#06b6d4";
const FILL_OPACITY          = 0.28;
const FILL_OPACITY_SELECTED = 0.45;
const LINE_COLOR            = "#ffffff";
const LINE_COLOR_SELECTED   = "#22d3ee";

/**
 * Discriminator for what's being dragged:
 *  - `node`: a single centerline control node at `index` (endpoint or
 *    interior shape-control). Width-preserving local edit.
 *  - `move`: the entire lanelet, translated rigidly by the pointer delta.
 */
export type DragHandleState =
  | { kind: "node"; id: number; index: number }
  | { kind: "move"; id: number };

interface LaneletLayerProps {
  lanelets: ResolvedLanelet[];
  selectedIds: Set<number>;
  /** Enables hit-testing on lanelet fills. */
  selectable: boolean;
  onSelect: (id: number, shiftKey: boolean) => void;
  /** Enables drag handles on selected lanelets' centerline indices. */
  editable: boolean;
  onHandlePointerDown: (id: number, index: number) => void;
  /** Enables the green "move whole lanelet" handle on selected lanelets. */
  onMoveHandlePointerDown: (id: number) => void;
  /** Which handle is currently being dragged, if any. */
  activeDragHandle: DragHandleState | null;
  /** Toggle rect-lock (straight) on a lanelet from the in-scene label. */
  onToggleStraight: (id: number) => void;
  /** Toggle position-lock — hides all drag handles and blocks moves. */
  onTogglePositionLocked: (id: number) => void;
  pendingStart: Vec3 | null;
  pendingStartAttached: boolean;
  sceneRadius: number;
  junctionPositions: Vec3[];
}

export function LaneletLayer({
  lanelets,
  selectedIds,
  selectable,
  onSelect,
  editable,
  onHandlePointerDown,
  onMoveHandlePointerDown,
  activeDragHandle,
  onToggleStraight,
  onTogglePositionLocked,
  pendingStart,
  pendingStartAttached,
  sceneRadius,
  junctionPositions,
}: LaneletLayerProps) {
  const arrowSize = Math.min(Math.max(0.3, sceneRadius / 150), 5) * 0.2;

  return (
    <>
      {lanelets.map((l) => {
        const isSelected = selectedIds.has(l.id);
        return (
          <LaneletMesh
            key={l.id}
            lanelet={l}
            selected={isSelected}
            selectable={selectable}
            onSelect={onSelect}
            arrowSize={arrowSize}
            showHandles={editable && isSelected && !l.positionLocked}
            onHandlePointerDown={onHandlePointerDown}
            onMoveHandlePointerDown={onMoveHandlePointerDown}
            activeDragHandle={activeDragHandle}
            onToggleStraight={onToggleStraight}
            onTogglePositionLocked={onTogglePositionLocked}
          />
        );
      })}

      {junctionPositions.length > 0 && (
        <JunctionMarkers positions={junctionPositions} />
      )}

      {pendingStart && (
        <PendingStartMarker
          pos={pendingStart}
          attached={pendingStartAttached}
        />
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// One lanelet = fill strip + two boundary polylines + dashed centerline +
// direction arrow on the last segment. Drag handles on every centerline
// index (endpoints + interior shape controls).
// ---------------------------------------------------------------------------
interface LaneletMeshProps {
  lanelet: ResolvedLanelet;
  selected: boolean;
  selectable: boolean;
  onSelect: (id: number, shiftKey: boolean) => void;
  arrowSize: number;
  showHandles: boolean;
  onHandlePointerDown: (id: number, index: number) => void;
  onMoveHandlePointerDown: (id: number) => void;
  activeDragHandle: DragHandleState | null;
  onToggleStraight: (id: number) => void;
  onTogglePositionLocked: (id: number) => void;
}

function LaneletMesh({
  lanelet,
  selected,
  selectable,
  onSelect,
  arrowSize,
  showHandles,
  onHandlePointerDown,
  onMoveHandlePointerDown,
  activeDragHandle,
  onToggleStraight,
  onTogglePositionLocked,
}: LaneletMeshProps) {
  const {
    leftSmooth,
    rightSmooth,
    centerSmooth,
    centerline,
    subType,
  } = lanelet;
  const n    = centerline.length;        // # of control nodes (= # of drag handles)
  // leftSmooth/rightSmooth/centerSmooth are packed Float32Arrays — xyz per
  // sample, so the sample count is `length / 3`.
  const nSmo = leftSmooth.length / 3;
  const isCrosswalk = subType === "crosswalk";

  // ---- Fill: triangle strip between the smooth left and right polylines.
  // For each smooth segment i we emit 2 triangles: (L[i],R[i],R[i+1]) and
  // (L[i],R[i+1],L[i+1]). Because the samples come from the same spline
  // sweep on both sides with the same parameterisation, the ribbon stays
  // consistent around curves without twist or self-intersection.
  const fillGeo = useMemo(() => {
    const verts = new Float32Array(nSmo * 2 * 3);
    for (let i = 0; i < nSmo; i++) {
      const src = i * 3;
      const dst = i * 6;
      verts[dst + 0] = leftSmooth [src    ];
      verts[dst + 1] = leftSmooth [src + 1];
      verts[dst + 2] = leftSmooth [src + 2];
      verts[dst + 3] = rightSmooth[src    ];
      verts[dst + 4] = rightSmooth[src + 1];
      verts[dst + 5] = rightSmooth[src + 2];
    }
    const indices: number[] = [];
    for (let i = 0; i < nSmo - 1; i++) {
      const li  = i * 2;
      const ri  = i * 2 + 1;
      const li1 = (i + 1) * 2;
      const ri1 = (i + 1) * 2 + 1;
      // Two triangles per smooth segment — winding is arbitrary since we
      // render DoubleSide.
      indices.push(li, ri, ri1);
      indices.push(li, ri1, li1);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(verts, 3));
    geo.setIndex(indices);
    return geo;
  }, [leftSmooth, rightSmooth, nSmo]);

  // Dispose the previous fillGeo as soon as useMemo returns a new one (and on
  // unmount). R3F's `<mesh geometry={...}>` does NOT auto-dispose the
  // *replaced* geometry — only the final one at unmount. Without this cleanup
  // every drag tick leaks a BufferGeometry and its GPU vertex buffer.
  useEffect(() => () => fillGeo.dispose(), [fillGeo]);

  const baseFill = isCrosswalk
    ? FILL_COLOR_CROSSWALK
    : FILL_COLOR_ROAD;

  const fillColor   = selected ? FILL_COLOR_SELECTED   : baseFill;
  const fillOpacity = selected ? FILL_OPACITY_SELECTED : FILL_OPACITY;
  const lineColor   = selected ? LINE_COLOR_SELECTED   : LINE_COLOR;
  const lineWidth   = selected ? 3                     : 2;

  // Dash cadence: proportional to total centerline length so short and
  // long lanelets both get readable dashes. Uses the smooth polyline so
  // curved lanelets aren't under-measured by their chord.
  const centerlineLength = useMemo(() => {
    let len = 0;
    for (let i = 0; i < nSmo - 1; i++) {
      const a = i * 3;
      const b = (i + 1) * 3;
      len += Math.hypot(
        centerSmooth[b    ] - centerSmooth[a    ],
        centerSmooth[b + 2] - centerSmooth[a + 2],
      );
    }
    return len;
  }, [centerSmooth, nSmo]);
  const dash = Math.max(0.3, centerlineLength / 8);

  // Zebra stripes — only built for crosswalks. Alternating white bars laid
  // across the ribbon perpendicular to travel, spaced roughly like a real
  // crossing (~0.5 m stripe / 0.5 m gap, both scaled up a bit with ribbon
  // length so short crosswalks still get a few visible bars). The stripe
  // quads ride between the SAME leftSmooth/rightSmooth samples used by
  // the fill, so they always stay inside the ribbon even on curves.
  const stripeGeo = useMemo(() => {
    if (!isCrosswalk || nSmo < 2 || centerlineLength < 0.1) return null;

    // Cumulative XZ arc length per smooth sample — needed to place stripe
    // boundaries at physical metre offsets regardless of spline parameter.
    const arc = new Float32Array(nSmo);
    for (let i = 1; i < nSmo; i++) {
      const a = (i - 1) * 3;
      const b = i * 3;
      arc[i] = arc[i - 1] + Math.hypot(
        centerSmooth[b    ] - centerSmooth[a    ],
        centerSmooth[b + 2] - centerSmooth[a + 2],
      );
    }
    const total = arc[nSmo - 1];

    const stripeW = 0.5;                                 // metres of paint
    const gapW    = 0.5;                                 // metres of gap
    const period  = stripeW + gapW;
    const count   = Math.max(2, Math.floor(total / period));
    // Re-centre so the first half-gap and last half-gap are equal and the
    // stripes sit symmetrically on the crossing.
    const used    = count * period - gapW;               // metres of stripe+gap+...+stripe
    const offset  = Math.max(0, (total - used) / 2);

    // Walk arc-length from index 0 upward; write the interpolated left/right
    // xyz pair at `positions[off .. off+5]` (l.xyz then r.xyz). No tuple
    // allocations in the inner loop.
    const writeSample = (s: number, positions: Float32Array, off: number) => {
      let i = 0;
      while (i < nSmo - 1 && arc[i + 1] < s) i++;
      const j = Math.min(nSmo - 1, i + 1);
      const a = arc[i];
      const b = arc[j];
      const t = b > a ? (s - a) / (b - a) : 0;
      const iOff = i * 3, jOff = j * 3;
      for (let k = 0; k < 3; k++) {
        positions[off + k    ] =
          leftSmooth [iOff + k] + (leftSmooth [jOff + k] - leftSmooth [iOff + k]) * t;
        positions[off + k + 3] =
          rightSmooth[iOff + k] + (rightSmooth[jOff + k] - rightSmooth[iOff + k]) * t;
      }
    };

    // Worst-case size — each stripe contributes 4 vertices × 3 floats.
    const positions = new Float32Array(count * 12);
    const indices:   number[] = [];
    let base = 0, floatIdx = 0;
    for (let s = 0; s < count; s++) {
      const sStart = offset + s * period;
      const sEnd   = Math.min(total, sStart + stripeW);
      if (sEnd <= sStart) continue;
      writeSample(sStart, positions, floatIdx);
      writeSample(sEnd,   positions, floatIdx + 6);
      // Two triangles per stripe quad (doubled winding, DoubleSide render).
      indices.push(base + 0, base + 1, base + 3);
      indices.push(base + 0, base + 3, base + 2);
      base += 4;
      floatIdx += 12;
    }

    if (indices.length === 0) return null;
    // Trim the tail if any stripes were skipped (rare — only when period
    // over-counts at the very end of a short crossing).
    const positionAttr =
      floatIdx === positions.length ? positions : positions.slice(0, floatIdx);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positionAttr, 3));
    geo.setIndex(indices);
    return geo;
  }, [isCrosswalk, leftSmooth, rightSmooth, centerSmooth, nSmo, centerlineLength]);

  useEffect(() => {
    if (!stripeGeo) return;
    return () => stripeGeo.dispose();
  }, [stripeGeo]);

  const hoverHandlers = selectable
    ? {
        onPointerOver: (e: { stopPropagation: () => void }) => {
          e.stopPropagation();
          document.body.style.cursor = "pointer";
        },
        onPointerOut: () => {
          document.body.style.cursor = "";
        },
        onClick: (e: { stopPropagation: () => void; shiftKey: boolean }) => {
          e.stopPropagation();
          onSelect(lanelet.id, e.shiftKey);
        },
      }
    : {};

  // Arrow sits at the last smooth sample, pointing along the final
  // smooth segment — so curved lanelets get a heading that tracks the
  // actual tangent at the tip, not the chord between control nodes.
  // Boxing the xyz triples once per buffer change (cached via resolve
  // cache in SceneViewer) keeps `DirectionArrow` and the handle/label
  // consumers' prop identities stable across unrelated re-renders.
  const arrowStart = useMemo<Vec3>(() => {
    const i = Math.max(0, nSmo - 2) * 3;
    return [centerSmooth[i], centerSmooth[i + 1], centerSmooth[i + 2]];
  }, [centerSmooth, nSmo]);
  const arrowEnd = useMemo<Vec3>(() => {
    const i = (nSmo - 1) * 3;
    return [centerSmooth[i], centerSmooth[i + 1], centerSmooth[i + 2]];
  }, [centerSmooth, nSmo]);
  // Midpoint — shared by the green move handle and the #id / lock-toggle
  // label. Same cache story as arrow endpoints.
  const midPos = useMemo<Vec3>(() => {
    const i = Math.floor(nSmo / 2) * 3;
    return [centerSmooth[i], centerSmooth[i + 1], centerSmooth[i + 2]];
  }, [centerSmooth, nSmo]);
  // drei's `<Line>` for the dashed centerline needs an array of 3-tuples,
  // so box the packed buffer once per buffer change (same cache story as
  // `BoundaryLine`'s `useVec3View`).
  const centerSmoothVec3 = useVec3View(centerSmooth);

  return (
    <>
      <mesh
        geometry={fillGeo}
        renderOrder={RO_FILL}
        {...hoverHandlers}
      >
        <meshBasicMaterial
          color={fillColor}
          transparent
          opacity={fillOpacity}
          side={THREE.DoubleSide}
          depthTest={false}
          depthWrite={false}
        />
      </mesh>

      {stripeGeo && (
        // White zebra bars rendered just above the fill (RO_FILL+1) so
        // they don't z-fight with it but still sit below boundary lines.
        <mesh geometry={stripeGeo} renderOrder={RO_FILL + 1}>
          <meshBasicMaterial
            color="#ffffff"
            transparent
            opacity={selected ? 0.95 : 0.85}
            side={THREE.DoubleSide}
            depthTest={false}
            depthWrite={false}
          />
        </mesh>
      )}

      <BoundaryLine
        points={leftSmooth}
        color={lineColor}
        lineWidth={lineWidth}
        dashed={lanelet.leftBoundarySubType === "dashed"}
      />
      <BoundaryLine
        points={rightSmooth}
        color={lineColor}
        lineWidth={lineWidth}
        dashed={lanelet.rightBoundarySubType === "dashed"}
      />

      {/* Crosswalks aren't directional, so the dashed centerline and the
          direction arrow would be misleading — only road lanelets get
          them. The zebra stripes above already convey the "this is a
          crossing" read on their own. */}
      {!isCrosswalk && (
        <>
          <Line
            points={centerSmoothVec3}
            color={lineColor}
            lineWidth={selected ? 1.6 : 1.2}
            dashed
            dashSize={dash}
            gapSize={dash * 0.8}
            transparent
            opacity={selected ? 0.9 : 0.75}
            depthTest={false}
            renderOrder={RO_CENTERLINE}
          />

          <DirectionArrow
            start={arrowStart}
            end={arrowEnd}
            size={arrowSize}
            color={lineColor}
          />
        </>
      )}

      {showHandles && centerline.map((pos, i) => {
        const isInterior = i !== 0 && i !== n - 1;
        // Rect-locked lanelets expose only their endpoints — hiding
        // interior handles makes the "length only" contract visible
        // and prevents the user from trying to curve a shape that
        // would snap back on the next endpoint drag anyway.
        if (lanelet.straight && isInterior) return null;
        return (
          <LaneletHandle
            key={i}
            pos={pos}
            active={
              activeDragHandle?.kind === "node" &&
              activeDragHandle.id === lanelet.id &&
              activeDragHandle.index === i
            }
            /* Endpoints (first/last) get the cyan palette; interior shape
               controls are a slightly different pink so they're easier to
               tell apart visually. */
            interior={isInterior}
            onPointerDown={() => onHandlePointerDown(lanelet.id, i)}
          />
        );
      })}

      {showHandles && (
        <MoveHandle
          pos={midPos}
          active={
            activeDragHandle?.kind === "move" &&
            activeDragHandle.id === lanelet.id
          }
          onPointerDown={() => onMoveHandlePointerDown(lanelet.id)}
        />
      )}

      {selected && (
        <LaneletIdLabel
          pos={midPos}
          id={lanelet.id}
          rectLocked={!!lanelet.straight}
          positionLocked={!!lanelet.positionLocked}
          onToggleRectLock={() => onToggleStraight(lanelet.id)}
          onTogglePositionLock={() => onTogglePositionLocked(lanelet.id)}
        />
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// ID label shown when the lanelet is selected. Uses drei's <Html> instead of
// drei's <Text> because <Text> needs to fetch its default font from a CDN —
// in a sandboxed / offline dev environment that fails and the component
// paints white glyph-box fallbacks, which looked like a stray white face
// over the ribbon.
// ---------------------------------------------------------------------------
interface LaneletIdLabelProps {
  pos:                  Vec3;
  id:                   number;
  rectLocked:           boolean;
  positionLocked:       boolean;
  onToggleRectLock:     () => void;
  onTogglePositionLock: () => void;
}

function LaneletIdLabel({
  pos, id, rectLocked, positionLocked, onToggleRectLock, onTogglePositionLock,
}: LaneletIdLabelProps) {
  const htmlPos = useMemo<[number, number, number]>(
    () => [pos[0], pos[1] + 0.05, pos[2]],
    [pos[0], pos[1], pos[2]]
  );

  // Straight-lane icon: two parallel horizontal lines = shape locked to straight.
  const StraightIcon = () => (
    <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth={2.2} viewBox="0 0 24 24">
      <path strokeLinecap="round" d="M3 8h18M3 16h18" />
    </svg>
  );

  // Padlock icon: full position freeze.
  const PadlockIcon = ({ locked }: { locked: boolean }) => (
    <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
      {locked ? (
        <path strokeLinecap="round" strokeLinejoin="round"
          d="M16.5 10.5V6.75a4.5 4.5 0 1 0-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 0 0 2.25-2.25v-6.75a2.25 2.25 0 0 0-2.25-2.25H6.75a2.25 2.25 0 0 0-2.25 2.25v6.75a2.25 2.25 0 0 0 2.25 2.25Z" />
      ) : (
        <path strokeLinecap="round" strokeLinejoin="round"
          d="M13.5 10.5V6.75a4.5 4.5 0 1 1 9 0v3.75M3.75 21.75h10.5a2.25 2.25 0 0 0 2.25-2.25v-6.75a2.25 2.25 0 0 0-2.25-2.25H3.75a2.25 2.25 0 0 0-2.25 2.25v6.75a2.25 2.25 0 0 0 2.25 2.25Z" />
      )}
    </svg>
  );

  return (
    <Html position={htmlPos} center zIndexRange={[100, 0]}>
      <div
        style={{
          display:       "flex",
          alignItems:    "center",
          gap:           3,
          padding:       "3px 7px 3px 10px",
          borderRadius:  9999,
          background:    "rgba(0,0,0,0.70)",
          border:        "1px solid rgba(255,255,255,0.12)",
          color:         "#ffffff",
          fontSize:      16,
          fontFamily:    "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
          fontWeight:    600,
          letterSpacing: 0.2,
          whiteSpace:    "nowrap",
          userSelect:    "none",
          lineHeight:    1,
          pointerEvents: "auto",
        }}
      >
        {`#${id}`}

        {/* Rect-lock (amber) — two parallel lines = straight shape constraint */}
        <button
          onClick={(e) => { e.stopPropagation(); onToggleRectLock(); }}
          title={rectLocked ? "Straight lock ON — click to allow curves" : "Curves allowed — click to lock straight"}
          style={{
            background: rectLocked ? "rgba(217,119,6,0.75)" : "none",
            border:     rectLocked ? "1px solid rgba(251,191,36,0.5)" : "1px solid transparent",
            borderRadius: 9999,
            padding:    "1px 4px",
            cursor:     "pointer",
            display:    "flex",
            alignItems: "center",
            opacity:    rectLocked ? 1 : 0.4,
            color:      rectLocked ? "#fde68a" : "#ffffff",
          }}
        >
          <StraightIcon />
        </button>

        {/* Position-lock (red) — padlock = full freeze (no drag, no fit-on-plane) */}
        <button
          onClick={(e) => { e.stopPropagation(); onTogglePositionLock(); }}
          title={positionLocked ? "Position locked — click to allow moving" : "Unlocked — click to freeze position"}
          style={{
            background: positionLocked ? "rgba(185,28,28,0.75)" : "none",
            border:     positionLocked ? "1px solid rgba(248,113,113,0.5)" : "1px solid transparent",
            borderRadius: 9999,
            padding:    "1px 4px",
            cursor:     "pointer",
            display:    "flex",
            alignItems: "center",
            opacity:    positionLocked ? 1 : 0.4,
            color:      positionLocked ? "#fca5a5" : "#ffffff",
          }}
        >
          <PadlockIcon locked={positionLocked} />
        </button>
      </div>
    </Html>
  );
}

// ---------------------------------------------------------------------------
// Boundary polyline — solid by default, dashed when lane change allowed.
// ---------------------------------------------------------------------------
interface BoundaryLineProps {
  /** Packed xyz Float32Array (xyz per sample, `length = 3 * N`). */
  points:    Float32Array;
  color:     string;
  lineWidth: number;
  dashed:    boolean;
}

function BoundaryLine({ points, color, lineWidth, dashed }: BoundaryLineProps) {
  const n = points.length / 3;

  const dashSize = useMemo(() => {
    let len = 0;
    for (let i = 0; i < n - 1; i++) {
      const a = i * 3;
      const b = (i + 1) * 3;
      len += Math.hypot(points[b] - points[a], points[b + 2] - points[a + 2]);
    }
    return Math.max(0.2, len / 12);
  }, [points, n]);

  // drei's `<Line>` needs an array of 3-tuples (or `Vector3` instances),
  // so the Float32Array has to be boxed for the interface. We do it once
  // per buffer change via useMemo: the per-lanelet resolve cache in
  // SceneViewer keeps `points`' identity stable across unrelated
  // re-renders, so this short-circuits and allocates no tuples during
  // idle frames.
  const pointsVec3 = useVec3View(points);

  return (
    <Line
      points={pointsVec3}
      color={color}
      lineWidth={lineWidth}
      transparent
      depthTest={false}
      renderOrder={RO_LINE}
      dashed={dashed}
      {...(dashed ? { dashSize, gapSize: dashSize * 0.6 } : {})}
    />
  );
}

/**
 * Materialise a packed xyz `Float32Array` (length `3 * N`) as a
 * `[x, y, z][]` once per buffer identity. Thanks to the resolve cache,
 * unchanged lanelets keep the same buffer reference across renders, so
 * the boxed view is only built when the smooth samples actually changed.
 */
function useVec3View(buf: Float32Array): Vec3[] {
  return useMemo(() => {
    const n = buf.length / 3;
    const out: Vec3[] = new Array(n);
    for (let i = 0; i < n; i++) {
      const k = i * 3;
      out[i] = [buf[k], buf[k + 1], buf[k + 2]];
    }
    return out;
  }, [buf]);
}

// ---------------------------------------------------------------------------
// Draggable centerline node handle.
// ---------------------------------------------------------------------------
interface LaneletHandleProps {
  pos: Vec3;
  active: boolean;
  interior: boolean;
  onPointerDown: () => void;
}

const HANDLE_HIT_PIXELS = 14;

function LaneletHandle({ pos, active, interior, onPointerDown }: LaneletHandleProps) {
  const dotGeo = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(pos), 3)
    );
    return g;
  }, [pos]);

  useEffect(() => () => dotGeo.dispose(), [dotGeo]);

  const hitRef = useRef<THREE.Mesh>(null);
  const tmpVec = useMemo(() => new THREE.Vector3(), []);

  useFrame(({ camera, size }) => {
    const mesh = hitRef.current;
    if (!mesh) return;
    mesh.getWorldPosition(tmpVec);
    const distance = camera.position.distanceTo(tmpVec);
    const cam = camera as THREE.PerspectiveCamera;
    const fovRad = (cam.fov * Math.PI) / 180;
    const worldPerPixel = (2 * distance * Math.tan(fovRad / 2)) / size.height;
    mesh.scale.setScalar(worldPerPixel * HANDLE_HIT_PIXELS);
  });

  // Two palettes: endpoints (cyan) vs interior shape controls (magenta).
  // Active drag is always amber so the current manipulator pops.
  const haloColor = active
    ? "#f59e0b"
    : interior
      ? "#ec4899"
      : "#22d3ee";
  const coreColor = active
    ? "#fef3c7"
    : interior
      ? "#fce7f3"
      : "#ffffff";

  const haloSize  = interior ? 16 : 20;
  const coreSize  = interior ? 8  : 10;

  return (
    <>
      <points geometry={dotGeo} renderOrder={RO_HANDLE}>
        <pointsMaterial
          size={haloSize}
          sizeAttenuation={false}
          color={haloColor}
          transparent
          opacity={0.55}
          depthTest={false}
          depthWrite={false}
        />
      </points>
      <points geometry={dotGeo} renderOrder={RO_HANDLE + 1}>
        <pointsMaterial
          size={coreSize}
          sizeAttenuation={false}
          color={coreColor}
          transparent
          depthTest={false}
          depthWrite={false}
        />
      </points>
      <mesh
        ref={hitRef}
        position={pos}
        renderOrder={RO_HANDLE + 2}
        onPointerDown={(e) => {
          e.stopPropagation();
          onPointerDown();
        }}
        onPointerOver={(e) => {
          e.stopPropagation();
          document.body.style.cursor = active ? "grabbing" : "grab";
        }}
        onPointerOut={() => {
          if (!active) document.body.style.cursor = "";
        }}
      >
        <sphereGeometry args={[1, 10, 10]} />
        <meshBasicMaterial
          transparent
          opacity={0}
          depthTest={false}
          depthWrite={false}
        />
      </mesh>
    </>
  );
}

// ---------------------------------------------------------------------------
// Move-lanelet handle — green, larger, at the mid-point of the smooth
// centerline. Dragging it translates every boundary NodeId of this lanelet
// by the pointer delta (connected neighbors follow at shared junctions).
// ---------------------------------------------------------------------------
interface MoveHandleProps {
  pos: Vec3;
  active: boolean;
  onPointerDown: () => void;
}

const MOVE_HANDLE_HIT_PIXELS = 30;

function MoveHandle({ pos, active, onPointerDown }: MoveHandleProps) {
  const dotGeo = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(pos), 3)
    );
    return g;
  }, [pos]);

  useEffect(() => () => dotGeo.dispose(), [dotGeo]);

  const hitRef = useRef<THREE.Mesh>(null);
  const tmpVec = useMemo(() => new THREE.Vector3(), []);

  useFrame(({ camera, size }) => {
    const mesh = hitRef.current;
    if (!mesh) return;
    mesh.getWorldPosition(tmpVec);
    const distance = camera.position.distanceTo(tmpVec);
    const cam = camera as THREE.PerspectiveCamera;
    const fovRad = (cam.fov * Math.PI) / 180;
    const worldPerPixel = (2 * distance * Math.tan(fovRad / 2)) / size.height;
    mesh.scale.setScalar(worldPerPixel * MOVE_HANDLE_HIT_PIXELS);
  });

  const haloColor = active ? "#f59e0b" : "#10b981"; // amber when dragging, emerald otherwise
  const coreColor = active ? "#fef3c7" : "#d1fae5";

  return (
    <>
      <points geometry={dotGeo} renderOrder={RO_HANDLE}>
        <pointsMaterial
          size={40}
          sizeAttenuation={false}
          color={haloColor}
          transparent
          opacity={0.55}
          depthTest={false}
          depthWrite={false}
        />
      </points>
      <points geometry={dotGeo} renderOrder={RO_HANDLE + 1}>
        <pointsMaterial
          size={22}
          sizeAttenuation={false}
          color={coreColor}
          transparent
          depthTest={false}
          depthWrite={false}
        />
      </points>
      <mesh
        ref={hitRef}
        position={pos}
        renderOrder={RO_HANDLE + 2}
        onPointerDown={(e) => {
          e.stopPropagation();
          onPointerDown();
        }}
        onPointerOver={(e) => {
          e.stopPropagation();
          document.body.style.cursor = active ? "grabbing" : "move";
        }}
        onPointerOut={() => {
          if (!active) document.body.style.cursor = "";
        }}
      >
        <sphereGeometry args={[1, 10, 10]} />
        <meshBasicMaterial
          transparent
          opacity={0}
          depthTest={false}
          depthWrite={false}
        />
      </mesh>
    </>
  );
}

// ---------------------------------------------------------------------------
// Junction markers — NodeIds referenced by ≥ 2 lanelets.
// ---------------------------------------------------------------------------
interface JunctionMarkersProps { positions: Vec3[]; }

function JunctionMarkers({ positions }: JunctionMarkersProps) {
  const geo = useMemo(() => {
    const flat = new Float32Array(positions.length * 3);
    for (let i = 0; i < positions.length; i++) {
      flat[i * 3 + 0] = positions[i][0];
      flat[i * 3 + 1] = positions[i][1];
      flat[i * 3 + 2] = positions[i][2];
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(flat, 3));
    return g;
  }, [positions]);

  useEffect(() => () => geo.dispose(), [geo]);

  return (
    <>
      <points geometry={geo} renderOrder={RO_JUNCTION}>
        <pointsMaterial
          size={18}
          sizeAttenuation={false}
          color="#f59e0b"
          transparent
          opacity={0.5}
          depthTest={false}
          depthWrite={false}
        />
      </points>
      <points geometry={geo} renderOrder={RO_JUNCTION + 1}>
        <pointsMaterial
          size={8}
          sizeAttenuation={false}
          color="#fde68a"
          transparent
          depthTest={false}
          depthWrite={false}
        />
      </points>
    </>
  );
}

// ---------------------------------------------------------------------------
// Pending start marker (first click of two-click create).
// ---------------------------------------------------------------------------
function PendingStartMarker({
  pos,
  attached,
}: {
  pos: Vec3;
  attached: boolean;
}) {
  const geo = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(pos), 3)
    );
    return g;
  }, [pos]);

  useEffect(() => () => geo.dispose(), [geo]);

  const halo = attached ? "#f59e0b" : "#22d3ee";
  const core = attached ? "#fde68a" : "#ffffff";

  return (
    <>
      <points geometry={geo} renderOrder={RO_DOT}>
        <pointsMaterial
          size={22}
          sizeAttenuation={false}
          color={halo}
          transparent
          opacity={0.4}
          depthTest={false}
        />
      </points>
      <points geometry={geo} renderOrder={RO_DOT + 1}>
        <pointsMaterial
          size={10}
          sizeAttenuation={false}
          color={core}
          transparent
          depthTest={false}
        />
      </points>
    </>
  );
}

// ---------------------------------------------------------------------------
// Direction arrow — flat triangle in the XZ plane at `end`, pointing
// start → end. Used at the last segment of a lanelet's centerline.
// ---------------------------------------------------------------------------
interface ArrowProps {
  start: Vec3;
  end:   Vec3;
  size:  number;
  color: string;
}

function DirectionArrow({ start, end, size, color }: ArrowProps) {
  const { geometry, rotY } = useMemo(() => {
    const h = size;
    const w = size * 0.55;
    const verts = new Float32Array([
       0, 0,  0,
      -h, 0, -w,
      -h, 0,  w,
    ]);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(verts, 3));
    geo.setIndex([0, 1, 2]);

    const dx = end[0] - start[0];
    const dz = end[2] - start[2];
    const angle = -Math.atan2(dz, dx);
    return { geometry: geo, rotY: angle };
  }, [start, end, size]);

  useEffect(() => () => geometry.dispose(), [geometry]);

  return (
    <mesh
      geometry={geometry}
      position={end}
      rotation={[0, rotY, 0]}
      renderOrder={RO_ARROW}
    >
      <meshBasicMaterial
        color={color}
        side={THREE.DoubleSide}
        depthTest={false}
        transparent
      />
    </mesh>
  );
}
