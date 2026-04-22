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

interface LaneletLayerProps {
  lanelets: ResolvedLanelet[];
  selectedIds: Set<number>;
  /** Enables hit-testing on lanelet fills. */
  selectable: boolean;
  onSelect: (id: number, shiftKey: boolean) => void;
  /** Enables drag handles on selected lanelets' centerline indices. */
  editable: boolean;
  onHandlePointerDown: (id: number, index: number) => void;
  /** Which lanelet/index is being dragged, if any. */
  activeDragHandle: { id: number; index: number } | null;
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
  activeDragHandle,
  pendingStart,
  pendingStartAttached,
  sceneRadius,
  junctionPositions,
}: LaneletLayerProps) {
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
  activeDragHandle: { id: number; index: number } | null;
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
  const {
    leftSmooth,
    rightSmooth,
    centerSmooth,
    centerline,
    subType,
  } = lanelet;
  const n    = centerline.length;        // # of control nodes (= # of drag handles)
  const nSmo = leftSmooth.length;        // # of smooth samples (>> n for n ≥ 3)

  // ---- Fill: triangle strip between the smooth left and right polylines.
  // For each smooth segment i we emit 2 triangles: (L[i],R[i],R[i+1]) and
  // (L[i],R[i+1],L[i+1]). Because the samples come from the same spline
  // sweep on both sides with the same parameterisation, the ribbon stays
  // consistent around curves without twist or self-intersection.
  const fillGeo = useMemo(() => {
    const verts = new Float32Array(nSmo * 2 * 3);
    for (let i = 0; i < nSmo; i++) {
      const l = leftSmooth[i];
      const r = rightSmooth[i];
      verts[i * 6 + 0] = l[0];
      verts[i * 6 + 1] = l[1];
      verts[i * 6 + 2] = l[2];
      verts[i * 6 + 3] = r[0];
      verts[i * 6 + 4] = r[1];
      verts[i * 6 + 5] = r[2];
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

  const baseFill = subType === "crosswalk"
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
      const a = centerSmooth[i];
      const b = centerSmooth[i + 1];
      len += Math.hypot(b[0] - a[0], b[2] - a[2]);
    }
    return len;
  }, [centerSmooth, nSmo]);
  const dash = Math.max(0.3, centerlineLength / 8);

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
  const arrowStart = centerSmooth[nSmo - 2] ?? centerSmooth[0];
  const arrowEnd   = centerSmooth[nSmo - 1];

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

      {/* Dashed centerline — follow the smooth spine. */}
      <Line
        points={centerSmooth}
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

      {showHandles && centerline.map((pos, i) => (
        <LaneletHandle
          key={i}
          pos={pos}
          active={
            activeDragHandle?.id === lanelet.id &&
            activeDragHandle?.index === i
          }
          /* Endpoints (first/last) get the cyan palette; interior shape
             controls are a slightly different pink so they're easier to
             tell apart visually. */
          interior={i !== 0 && i !== n - 1}
          onPointerDown={() => onHandlePointerDown(lanelet.id, i)}
        />
      ))}
    </>
  );
}

// ---------------------------------------------------------------------------
// Boundary polyline — solid by default, dashed when lane change allowed.
// ---------------------------------------------------------------------------
interface BoundaryLineProps {
  points:    Vec3[];
  color:     string;
  lineWidth: number;
  dashed:    boolean;
}

function BoundaryLine({ points, color, lineWidth, dashed }: BoundaryLineProps) {
  const dashSize = useMemo(() => {
    let len = 0;
    for (let i = 0; i < points.length - 1; i++) {
      const a = points[i];
      const b = points[i + 1];
      len += Math.hypot(b[0] - a[0], b[2] - a[2]);
    }
    return Math.max(0.2, len / 12);
  }, [points]);

  return (
    <Line
      points={points}
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
