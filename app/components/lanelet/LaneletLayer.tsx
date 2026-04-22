"use client";

import { useMemo, useRef } from "react";
import { Line } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import type { ResolvedLanelet, Vec3 } from "./types";

// Render-order stack (higher = drawn later = on top)
//   cloud          0 (default)
//   lanelet fill   5
//   centerline     6
//   boundaries    10
//   arrows/dots   11+
//   edit handles  30+ (always on top of everything lanelet-related)
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

interface LaneletLayerProps {
  lanelets: ResolvedLanelet[];
  selectedIds: Set<number>;
  /** Enables hit-testing on lanelet fills (true in "view" tool only). */
  selectable: boolean;
  onSelect: (id: number, shiftKey: boolean) => void;
  /**
   * Enables drag handles on selected lanelets' centerStart / centerEnd.
   * Handles are only clickable when this is true.
   */
  editable: boolean;
  onHandlePointerDown: (id: number, which: "start" | "end") => void;
  /** Which lanelet/endpoint, if any, is currently being dragged. */
  activeDragHandle: { id: number; which: "start" | "end" } | null;
  /** If set, the user is mid-lanelet: show a preview marker at the start. */
  pendingStart: Vec3 | null;
  /** True when `pendingStart` was snapped to an existing lanelet's end — we
   *  draw the marker amber (junction color) to signal "this will connect". */
  pendingStartAttached: boolean;
  /** Scene scale for sensibly-sized direction arrows. */
  sceneRadius: number;
  /**
   * Positions of registry nodes referenced by ≥2 lanelets — structural
   * junctions. Drawn as small amber markers so users can SEE the topology.
   */
  junctionPositions: Vec3[];
}

export function LaneletLayer({
  lanelets,
  selectedIds,
  selectable,
  onSelect,
  editable,
  onHandlePointerDown,
  activeDragHandle,
  pendingStart,
  pendingStartAttached,
  sceneRadius,
  junctionPositions,
}: LaneletLayerProps) {
  // 40% of the previous size — matches the "60% smaller arrow" request.
  const arrowSize = Math.min(Math.max(0.3, sceneRadius / 150), 5) * 0.4;

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
            showHandles={editable && isSelected}
            onHandlePointerDown={onHandlePointerDown}
            activeDragHandle={activeDragHandle}
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
// One lanelet = fill quad + two boundary lines + dashed centerline + arrow.
// ---------------------------------------------------------------------------
interface LaneletMeshProps {
  lanelet: ResolvedLanelet;
  selected: boolean;
  selectable: boolean;
  onSelect: (id: number, shiftKey: boolean) => void;
  arrowSize: number;
  showHandles: boolean;
  onHandlePointerDown: (id: number, which: "start" | "end") => void;
  activeDragHandle: { id: number; which: "start" | "end" } | null;
}

function LaneletMesh({
  lanelet,
  selected,
  selectable,
  onSelect,
  arrowSize,
  showHandles,
  onHandlePointerDown,
  activeDragHandle,
}: LaneletMeshProps) {
  const [lStart, lEnd] = lanelet.leftBoundary;
  const [rStart, rEnd] = lanelet.rightBoundary;
  const { centerStart, centerEnd, subType } = lanelet;

  const fillGeo = useMemo(() => {
    const v = new Float32Array([
      lStart[0], lStart[1], lStart[2],
      rStart[0], rStart[1], rStart[2],
      rEnd[0],   rEnd[1],   rEnd[2],
      lEnd[0],   lEnd[1],   lEnd[2],
    ]);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(v, 3));
    geo.setIndex([0, 1, 2, 0, 2, 3]);
    return geo;
  }, [lStart, rStart, rEnd, lEnd]);

  const baseFill = subType === "crosswalk"
    ? FILL_COLOR_CROSSWALK
    : FILL_COLOR_ROAD;

  const fillColor   = selected ? FILL_COLOR_SELECTED   : baseFill;
  const fillOpacity = selected ? FILL_OPACITY_SELECTED : FILL_OPACITY;
  const lineColor   = selected ? LINE_COLOR_SELECTED   : LINE_COLOR;
  const lineWidth   = selected ? 3                     : 2;

  const dash = useMemo(() => {
    const dx = centerEnd[0] - centerStart[0];
    const dz = centerEnd[2] - centerStart[2];
    const len = Math.hypot(dx, dz);
    return Math.max(0.3, len / 8);
  }, [centerStart, centerEnd]);

  // Cursor feedback for clickability (only when selectable).
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

      {/* Boundary lines — dashed when the lanelet allows lane change on
          that side (Lanelet2 line_thin subtype="dashed"). */}
      <BoundaryLine
        start={lStart}
        end={lEnd}
        color={lineColor}
        lineWidth={lineWidth}
        dashed={lanelet.leftBoundarySubType === "dashed"}
      />
      <BoundaryLine
        start={rStart}
        end={rEnd}
        color={lineColor}
        lineWidth={lineWidth}
        dashed={lanelet.rightBoundarySubType === "dashed"}
      />

      {/* Dashed centerline */}
      <Line
        points={[centerStart, centerEnd]}
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
        start={centerStart}
        end={centerEnd}
        size={arrowSize}
        color={lineColor}
      />

      {showHandles && (
        <>
          <LaneletHandle
            pos={centerStart}
            active={
              activeDragHandle?.id === lanelet.id &&
              activeDragHandle?.which === "start"
            }
            onPointerDown={() => onHandlePointerDown(lanelet.id, "start")}
          />
          <LaneletHandle
            pos={centerEnd}
            active={
              activeDragHandle?.id === lanelet.id &&
              activeDragHandle?.which === "end"
            }
            onPointerDown={() => onHandlePointerDown(lanelet.id, "end")}
          />
        </>
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// Draggable centerline endpoint.
//
// Consists of three layers:
//   1. Pixel-sized halo  (visual only, no hit-testing)
//   2. Pixel-sized core  (visual only, no hit-testing)
//   3. World-space sphere whose diameter is rescaled every frame so it
//      stays ≈ 22 px on screen — regardless of camera FOV / distance.
//      The sphere is what actually catches the mouse.
// ---------------------------------------------------------------------------
interface LaneletHandleProps {
  pos: Vec3;
  active: boolean;
  onPointerDown: () => void;
}

// ~pixel-radius we aim for when rescaling the invisible hit sphere.
const HANDLE_HIT_PIXELS = 14;

function LaneletHandle({ pos, active, onPointerDown }: LaneletHandleProps) {
  const dotGeo = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(pos), 3)
    );
    return g;
  }, [pos]);

  const hitRef = useRef<THREE.Mesh>(null);
  const tmpVec = useMemo(() => new THREE.Vector3(), []);

  // Keep the click target at a roughly constant pixel size — otherwise the
  // 2D top-down view (camera very far, fov ≈ 5°) yields a microscopic
  // sphere that is impossible to grab.
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

  const haloColor = active ? "#f59e0b" : "#22d3ee";
  const coreColor = active ? "#fef3c7" : "#ffffff";

  return (
    <>
      {/* Halo (visual only) */}
      <points geometry={dotGeo} renderOrder={RO_HANDLE}>
        <pointsMaterial
          size={20}
          sizeAttenuation={false}
          color={haloColor}
          transparent
          opacity={0.55}
          depthTest={false}
          depthWrite={false}
        />
      </points>

      {/* Core (visual only) */}
      <points geometry={dotGeo} renderOrder={RO_HANDLE + 1}>
        <pointsMaterial
          size={10}
          sizeAttenuation={false}
          color={coreColor}
          transparent
          depthTest={false}
          depthWrite={false}
        />
      </points>

      {/* Invisible hit target. `scale` gets updated every frame. */}
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
// Junction markers — one per NodeId referenced by two or more lanelets.
//
// These are the topological connection points between lanelets (e.g. where
// two feeder lanelets merge into one): if you move a junction node, every
// connected lanelet follows automatically — that's the whole point.
//
// Drawn as amber pixel-dots, sitting above the boundary lines but beneath
// the edit handles so a selected lanelet's handles stay grabbable on top.
// ---------------------------------------------------------------------------
interface JunctionMarkersProps {
  positions: Vec3[];
}

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

  return (
    <>
      {/* Halo */}
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
      {/* Core */}
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
// Pending start: highlighted marker at the first click.
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

  // Amber when the start snapped to an existing lanelet's end (full-width
  // pair attach coming on click-2); cyan otherwise.
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
// One side of a lanelet's boundary (left or right).
// Renders solid by default; dashed when the lanelet has lane-change allowed
// on that side (LineThin subtype = "dashed").
// ---------------------------------------------------------------------------
interface BoundaryLineProps {
  start:     Vec3;
  end:       Vec3;
  color:     string;
  lineWidth: number;
  dashed:    boolean;
}

function BoundaryLine({ start, end, color, lineWidth, dashed }: BoundaryLineProps) {
  // Dash cadence derived from segment length so short boundaries still get
  // a few visible dashes.
  const dashSize = useMemo(() => {
    const dx = end[0] - start[0];
    const dz = end[2] - start[2];
    return Math.max(0.2, Math.hypot(dx, dz) / 12);
  }, [start, end]);

  return (
    <Line
      points={[start, end]}
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

// ---------------------------------------------------------------------------
// Flat XZ-plane triangle placed at `end`, tip pointing start → end.
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
