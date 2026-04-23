"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls, Grid, Stats } from "@react-three/drei";
import * as THREE from "three";
import { PointCloud } from "./PointCloud";
import type { CropRegion } from "./PointCloud";
import type { PointCloudChunk, PointCloudStats } from "./types";
import { LaneletLayer } from "./lanelet/LaneletLayer";
import type { DragHandleState } from "./lanelet/LaneletLayer";
import { LaneletProperties } from "./lanelet/LaneletProperties";
import {
  applyLaneletTranslationFromSnapshot,
  applyPlaneToLanelet,
  collectNodeIdsOfOtherSubType,
  createLanelet,
  duplicateLaneletLeft,
  duplicateLaneletRight,
  fitPlaneRansac,
  joinLanelets,
  moveLaneletNodeAtIndex,
  resizeLaneletWidth,
  reverseLanelet,
  sampleSurfaceY,
  snapLaneletToSurface,
  straightenLanelet,
  ATTACH_DISTANCE,
} from "./lanelet/geometry";
import type { JointType } from "./lanelet/geometry";
import {
  addNode,
  createRegistry,
  findNearestLaneletEnd,
  gcUnused,
  getNode,
  getNodeOrNull,
  junctionNodeIds,
  resolveAll,
} from "./lanelet/registry";
import { downloadLanelet2Osm } from "./lanelet/export";
import type { LaneletEndSnap } from "./lanelet/registry";
import type {
  Lanelet,
  NodeId,
  NodeRegistry,
  ResolvedLanelet,
  Vec3,
} from "./lanelet/types";

type CameraMode = "3d" | "2d";
// "view"        = selection/edit.
// "lanelet"     = draw a road lanelet.
// "crosswalk"   = draw a crosswalk (no direction arrow, zebra-striped fill).
// "crop-center" = one-shot pick mode: next click on the PCD sets the Z-clip
//                 region's center; the tool auto-returns to "view".
type Tool       = "view" | "lanelet" | "crosswalk" | "crop-center";

/**
 * Per-position ceiling height, mirroring the fragment shader in PointCloud.
 * When cropRegion is null the ceiling is flat at zCeiling globally.
 * When inside the crop rectangle the ceiling is the tilted plane.
 * When outside the rectangle (crop enabled, no clip there) returns zCeiling
 * so the raycast still finds real surface points.
 */
function effectiveCeilingAt(
  x: number,
  z: number,
  zCeiling: number,
  crop: CropRegion | null
): number {
  if (!crop) return zCeiling;

  const dx  = x - crop.cx;
  const dz  = z - crop.cz;
  const cosA = Math.cos(crop.angle);
  const sinA = Math.sin(crop.angle);
  // Rotate world delta into crop-local frame (matches shader uCropRot logic)
  const lx =  dx * cosA + dz * sinA;
  const lz = -dx * sinA + dz * cosA;

  if (Math.abs(lx) > crop.halfW || Math.abs(lz) > crop.halfL) {
    // Outside the rectangle: no ceiling is applied by the shader.
    return zCeiling;
  }

  const MAX_TILT = (85 * Math.PI) / 180;
  const pitch = Math.max(-MAX_TILT, Math.min(MAX_TILT, crop.pitch));
  const roll  = Math.max(-MAX_TILT, Math.min(MAX_TILT, crop.roll));
  return zCeiling + Math.tan(roll) * lx + Math.tan(pitch) * lz;
}

interface SceneProps {
  geometry: THREE.BufferGeometry | null;
  /** Rendered cloud tiles. Produced by the worker owned by the outer
   *  `PointCloudViewer`; this component is a pure renderer of them. */
  chunks: PointCloudChunk[];
  pointSize: number;
  zCeiling: number;
  cameraMode: CameraMode;
  tool: Tool;
  width: number;

  /** Active Z-clip region (null → global Y-clip). */
  cropRegion: CropRegion | null;
  /** Show the translucent amber fill on the crop box's tilted top face. */
  showCropCap: boolean;
  /** Invoked when `tool === "crop-center"` and the user clicks a point on
   *  the cloud. The handler should update the crop center and switch the
   *  tool back to "view". */
  onPickCropCenter: (p: Vec3) => void;

  registry: NodeRegistry;
  nextNodeIdRef: React.MutableRefObject<number>;

  lanelets: Lanelet[];
  resolvedLanelets: ResolvedLanelet[];
  junctionPositions: Vec3[];

  selectedIds: Set<number>;
  onSelect: (id: number, shiftKey: boolean) => void;
  pendingStart: Vec3 | null;
  /** Set iff the pending start attached to an existing lanelet's end. */
  pendingStartSnap: LaneletEndSnap | null;
  onStartLanelet: (p: Vec3, snap: LaneletEndSnap | null) => void;
  onFinishLanelet: (reg: NodeRegistry, l: Lanelet) => void;

  /** Called after every in-place edit (drag, surface-snap). Updates both
   *  registry and the single lanelet atomically. */
  onUpsertLanelet: (reg: NodeRegistry, l: Lanelet) => void;
  onToggleStraight: (id: number) => void;
  onTogglePositionLocked: (id: number) => void;

  nextLaneletIdRef: React.MutableRefObject<number>;
  /** Ref to the group of all chunk <points> — owned by SceneViewer so
   *  operations outside the Canvas (e.g. fit-plane) can read raw positions. */
  pointsRef: React.RefObject<THREE.Group | null>;
}

function Scene({
  geometry,
  chunks,
  pointSize,
  zCeiling,
  cameraMode,
  tool,
  width,
  cropRegion,
  showCropCap,
  onPickCropCenter,
  registry,
  nextNodeIdRef,
  lanelets,
  resolvedLanelets,
  junctionPositions,
  selectedIds,
  onSelect,
  pendingStart,
  pendingStartSnap,
  onStartLanelet,
  onFinishLanelet,
  onUpsertLanelet,
  onToggleStraight,
  onTogglePositionLocked,
  nextLaneletIdRef,
  pointsRef,
}: SceneProps) {
  const { camera, raycaster, gl } = useThree();
  const controlsRef = useRef<any>(null);

  // Always-current reads inside long-running drag handlers.
  const laneletsRef = useRef(lanelets);
  const registryRef = useRef(registry);
  const onUpsertRef = useRef(onUpsertLanelet);
  useEffect(() => { laneletsRef.current = lanelets;   }, [lanelets]);
  useEffect(() => { registryRef.current = registry;   }, [registry]);
  useEffect(() => { onUpsertRef.current = onUpsertLanelet; }, [onUpsertLanelet]);

  // Two flavors of drag:
  //   kind "node": a single centerline control node (index 0 = start,
  //                last = end, otherwise an interior shape-control) —
  //                width-preserving local edit.
  //   kind "move": the whole lanelet is translated rigidly; shared
  //                junction nodes pull connected neighbors along.
  const [dragHandle, setDragHandle] =
    useState<DragHandleState | null>(null);

  const surfaceRc = useMemo(() => new THREE.Raycaster(), []);

  // Tune pick raycaster threshold to scene scale.
  useEffect(() => {
    const r = geometry?.boundingSphere?.radius ?? 50;
    raycaster.params.Points = {
      ...(raycaster.params.Points ?? {}),
      threshold: Math.max(0.15, r / 400),
    };
  }, [geometry, raycaster]);

  useEffect(() => {
    if (!geometry?.boundingSphere) return;
    const { radius: r, center: c } = geometry.boundingSphere;
    const cam = camera as THREE.PerspectiveCamera;

    cam.near = r * 0.001;
    cam.far  = r * 200;

    if (cameraMode === "2d") {
      cam.fov = 5;
      cam.up.set(0, 0, -1);
      camera.position.set(c.x, c.y + r * 25, c.z);
    } else {
      cam.fov = 60;
      cam.up.set(0, 1, 0);
      const dist = r * 2.8;
      camera.position.set(
        c.x + dist * 0.6,
        c.y + dist * 0.5,
        c.z + dist * 0.8
      );
    }

    cam.updateProjectionMatrix();

    if (controlsRef.current) {
      controlsRef.current.target.copy(c);
      controlsRef.current.update();
    }
  }, [geometry, cameraMode, camera]);

  // --------------------------------------------------------------------
  // Drag handles.
  //
  // Two flavors — "node" (single control-node edit, keeps the lanelet's
  // width) and "move" (whole-lanelet rigid translation, pulls connected
  // neighbors along at shared junctions). Both share the same drag-plane
  // infrastructure: a horizontal XZ plane at the handle's current Y, with
  // OrbitControls disabled for the duration of the drag.
  //
  // "move" uses a pointer-down snapshot of every boundary NodeId's
  // position; each pointermove recomputes absolute positions as
  // `snapshot[id] + delta`, which keeps the drag idempotent (no drift).
  // --------------------------------------------------------------------
  const handleHandlePointerDown = (id: number, index: number) => {
    if (laneletsRef.current.find((l) => l.id === id)?.positionLocked) return;
    if (controlsRef.current) controlsRef.current.enabled = false;
    setDragHandle({ kind: "node", id, index });
  };

  const handleMoveHandlePointerDown = (id: number) => {
    if (laneletsRef.current.find((l) => l.id === id)?.positionLocked) return;
    if (controlsRef.current) controlsRef.current.enabled = false;
    setDragHandle({ kind: "move", id });
  };

  useEffect(() => {
    if (!dragHandle) return;

    const controls = controlsRef.current;
    if (controls) controls.enabled = false;

    const prevCursor = document.body.style.cursor;
    document.body.style.cursor = "grabbing";

    const initial = laneletsRef.current.find((l) => l.id === dragHandle.id);
    if (!initial) {
      if (controls) controls.enabled = true;
      document.body.style.cursor = prevCursor;
      setDragHandle(null);
      return;
    }

    const reg0 = registryRef.current;

    // Drag plane Y — and, for "move", also a snapshot of every boundary
    // node's starting position (used for absolute drag arithmetic).
    let planeY = 0;
    const snapshot: Record<NodeId, Vec3> = {};

    if (dragHandle.kind === "node") {
      const leftId  = initial.leftBoundary[dragHandle.index];
      const rightId = initial.rightBoundary[dragHandle.index];
      const leftP   = leftId  !== undefined ? reg0.nodes[leftId]  : undefined;
      const rightP  = rightId !== undefined ? reg0.nodes[rightId] : undefined;
      planeY = leftP && rightP ? (leftP[1] + rightP[1]) * 0.5 : 0;
    } else {
      // "move" — plane at centroid Y, snapshot every node we're about to
      // translate. Dedupe via the object so aliased NodeIds land once.
      let sumY = 0;
      let count = 0;
      const allIds = [...initial.leftBoundary, ...initial.rightBoundary];
      for (const id of allIds) {
        if (snapshot[id]) continue;
        const p = reg0.nodes[id];
        if (!p) continue;
        snapshot[id] = [p[0], p[1], p[2]];
        sumY += p[1];
        count++;
      }
      planeY = count > 0 ? sumY / count : 0;
    }

    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -planeY);
    const dragRc = new THREE.Raycaster();
    const ndc = new THREE.Vector2();
    const hit = new THREE.Vector3();

    const mouseToPlane = (clientX: number, clientY: number): Vec3 | null => {
      const rect = gl.domElement.getBoundingClientRect();
      ndc.x = ((clientX - rect.left) / rect.width) * 2 - 1;
      ndc.y = -((clientY - rect.top) / rect.height) * 2 + 1;
      dragRc.setFromCamera(ndc, camera);
      const p = dragRc.ray.intersectPlane(plane, hit);
      if (!p) return null;
      return [p.x, planeY, p.z];
    };

    // "move" uses the first pointermove as its anchor — set lazily so we
    // don't need access to the original pointerdown coordinates.
    let moveAnchor: Vec3 | null = null;

    const onMove = (e: PointerEvent) => {
      document.body.style.cursor = "grabbing";

      const np = mouseToPlane(e.clientX, e.clientY);
      if (!np) return;

      const curr = laneletsRef.current.find((l) => l.id === dragHandle.id);
      if (!curr) return;

      if (dragHandle.kind === "node") {
        const { reg, lanelet } = moveLaneletNodeAtIndex(
          registryRef.current,
          curr,
          dragHandle.index,
          np
        );
        onUpsertRef.current(reg, lanelet);
      } else {
        if (!moveAnchor) {
          moveAnchor = np;
          return;
        }
        const delta: Vec3 = [
          np[0] - moveAnchor[0],
          np[1] - moveAnchor[1],
          np[2] - moveAnchor[2],
        ];
        const newReg = applyLaneletTranslationFromSnapshot(
          registryRef.current,
          snapshot,
          delta
        );
        // Lanelet object itself is unchanged — only node positions moved.
        onUpsertRef.current(newReg, curr);
      }
    };

    const onUp = () => {
      const curr = laneletsRef.current.find((l) => l.id === dragHandle.id);
      if (curr) {
        surfaceRc.params.Points = {
          ...(surfaceRc.params.Points ?? {}),
          threshold: Math.max(0.1, Math.min(curr.width / 6, 0.8)),
        };
        const snapY = (x: number, z: number, fb: number) =>
          sampleSurfaceY(
            surfaceRc, pointsRef.current, x, z,
            effectiveCeilingAt(x, z, zCeiling, cropRegion) + 0.5,
            fb
          );

        const { reg, lanelet } = snapLaneletToSurface(
          registryRef.current,
          curr,
          snapY
        );
        onUpsertRef.current(reg, lanelet);
      }

      if (controls) controls.enabled = true;
      document.body.style.cursor = prevCursor;
      setDragHandle(null);
    };

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);

    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      if (controls) controls.enabled = true;
      document.body.style.cursor = prevCursor;
    };
  }, [dragHandle, camera, gl, surfaceRc, geometry]);

  // --------------------------------------------------------------------
  // Two-click create flow.
  //
  // Each click runs a pair-attach check first (findNearestLaneletEnd): if
  // the click lands near the centerpoint of an existing lanelet's end, the
  // new lanelet adopts that end's two NodeIds as a unit — guaranteeing a
  // full-width connection instead of a kink where only one corner reaches.
  //
  // If pair-attach doesn't fire, createLanelet falls back to per-corner
  // attach with ATTACH_DISTANCE, then surface-snaps any free corners.
  // --------------------------------------------------------------------

  /**
   * Threshold for pair-attach: generous enough that a click that's clearly
   * "aiming at" an existing endpoint snaps, tight enough not to merge two
   * lanelets the user meant to keep separate. Scales with current width so
   * wide lanelets have proportionally larger grab radius.
   */
  const endSnapThreshold = Math.max(ATTACH_DISTANCE, width * 0.75);

  const handlePick = (p: Vec3) => {
    // Crop-center picker: consume the click, hand the world point back to
    // the parent, done. Lanelet creation paths never run in this mode
    // because `pendingStart` is local to the lanelet flow and gets
    // cleared when entering "crop-center" via `pickTool`.
    if (tool === "crop-center") {
      onPickCropCenter(p);
      return;
    }

    // The drawing tool determines the kind we're creating. Crosswalks
    // must NEVER attach to road lanelets (they're physically on top of
    // roads but semantically separate), so both snap paths filter by
    // subType and we reserve every other-kind NodeId from per-corner
    // attach too.
    const newSubType: "road" | "crosswalk" =
      tool === "crosswalk" ? "crosswalk" : "road";
    const reservedNodeIds = collectNodeIdsOfOtherSubType(
      laneletsRef.current,
      newSubType,
    );

    if (!pendingStart) {
      const snap = findNearestLaneletEnd(
        registryRef.current,
        laneletsRef.current,
        p,
        endSnapThreshold,
        newSubType,
      );
      // When snapping, use the existing lanelet's centerpoint as the start
      // so the centerline is truly axis-aligned with the adopted corners.
      onStartLanelet(snap ? snap.center : p, snap);
      return;
    }

    const endSnap = findNearestLaneletEnd(
      registryRef.current,
      laneletsRef.current,
      p,
      endSnapThreshold,
      newSubType,
    );

    surfaceRc.params.Points = {
      ...(surfaceRc.params.Points ?? {}),
      threshold: Math.max(0.1, Math.min(width / 6, 0.8)),
    };

    const snapY = (x: number, z: number, fallback: number) =>
      sampleSurfaceY(
        surfaceRc, pointsRef.current, x, z,
        effectiveCeilingAt(x, z, zCeiling, cropRegion) + 0.5,
        fallback
      );

    const { reg, lanelet } = createLanelet({
      reg: registryRef.current,
      nextNodeId: nextNodeIdRef,
      laneletId: nextLaneletIdRef.current++,
      centerStart: pendingStart,
      centerEnd: endSnap ? endSnap.center : p,
      width,
      snapY,
      attachDistance: ATTACH_DISTANCE,
      startOverride: pendingStartSnap ?? undefined,
      endOverride:   endSnap ?? undefined,
      subType: newSubType,
      reservedNodeIds,
    });

    onFinishLanelet(reg, lanelet);
  };

  const r        = geometry?.boundingSphere?.radius ?? 50;
  const gridY    = geometry?.boundingBox ? geometry.boundingBox.min.y - 0.01 : 0;
  const gridSize = Math.ceil(r * 4);

  return (
    <>
      <color attach="background" args={["#050810"]} />
      <ambientLight intensity={0.4} />

      <OrbitControls
        ref={controlsRef}
        makeDefault
        enableDamping
        dampingFactor={0.08}
        rotateSpeed={0.6}
        panSpeed={0.8}
        zoomSpeed={1.2}
        minDistance={0.01}
        maxDistance={100000}
        enableRotate={cameraMode === "3d"}
        mouseButtons={{
          LEFT:   cameraMode === "2d" ? THREE.MOUSE.PAN : THREE.MOUSE.ROTATE,
          MIDDLE: THREE.MOUSE.DOLLY,
          RIGHT:  THREE.MOUSE.PAN,
        }}
      />

      <Grid
        position={[0, gridY, 0]}
        args={[gridSize, gridSize]}
        cellSize={gridSize / 20}
        cellThickness={0.4}
        cellColor="#1a3a5c"
        sectionSize={gridSize / 4}
        sectionThickness={0.8}
        sectionColor="#0e4f7a"
        fadeDistance={gridSize * 1.5}
        fadeStrength={1}
        infiniteGrid
      />

      {geometry && (
        <PointCloud
          chunks={chunks}
          pointSize={pointSize}
          cameraMode={cameraMode}
          zCeiling={zCeiling}
          cropRegion={cropRegion}
          pickEnabled={
            tool === "lanelet" ||
            tool === "crosswalk" ||
            tool === "crop-center"
          }
          onPick={handlePick}
          pointsRef={pointsRef}
        />
      )}

      {cropRegion && geometry?.boundingBox && (
        <CropBox
          cropRegion={cropRegion}
          zCeiling={zCeiling}
          floorY={geometry.boundingBox.min.y - 0.02}
          showCap={showCropCap}
        />
      )}

      <LaneletLayer
        lanelets={resolvedLanelets}
        selectedIds={selectedIds}
        selectable={tool === "view"}
        onSelect={onSelect}
        editable={tool === "view"}
        onHandlePointerDown={handleHandlePointerDown}
        onMoveHandlePointerDown={handleMoveHandlePointerDown}
        activeDragHandle={dragHandle}
        onToggleStraight={onToggleStraight}
        onTogglePositionLocked={onTogglePositionLocked}
        pendingStart={pendingStart}
        pendingStartAttached={pendingStartSnap !== null}
        sceneRadius={r}
        junctionPositions={junctionPositions}
      />
    </>
  );
}

// ---------------------------------------------------------------------------
// Wireframe box outlining the active Z-clip region. The top face sits
// exactly at `zCeiling` so it doubles as a visual read-out of the current
// clip plane, and the bottom face sits slightly under the cloud's
// bounding-box floor so it's clearly visible on inclined terrain.
//
// Lines are rendered with depthTest disabled (like the lanelet overlays)
// so the box stays visible through the PCD in 3D mode.
// ---------------------------------------------------------------------------
interface CropBoxProps {
  cropRegion: CropRegion;
  zCeiling:   number;
  floorY:     number;
  /** When false, the translucent clip-plane fill is skipped — only the
   *  yellow wireframe outline renders. Useful when the colored cap
   *  obscures cloud detail the user wants to inspect. */
  showCap?:   boolean;
}

/**
 * Wireframe outline of the crop volume.
 *
 * Bottom face: axis-aligned rectangle on the cloud floor at `floorY`.
 * Top face:    the *tilted* clip plane — corners lifted to whatever
 *              height the fragment shader would discard at that XZ.
 *
 * We build 8 vertices imperatively (4 bottom + 4 top) and connect them
 * with 12 line segments. Can't reuse BoxGeometry/EdgesGeometry because
 * pitch & roll give us a parallelepiped, not a box, and we want the
 * floor face to stay flat so the visualization communicates
 * "bottom grounded, top tilted to follow the slope".
 *
 * Coord convention matches the shader exactly:
 *   localClipHeight(lx, lz) = zCeiling + tan(roll)·lx + tan(pitch)·lz
 *   worldCorner(lx, lz)     = (cx, cz) + Rz(angle) · (lx, lz)
 * so what you see is what gets clipped.
 */
function CropBox({ cropRegion, zCeiling, floorY, showCap = true }: CropBoxProps) {
  const { cx, cz, halfW, halfL, angle, pitch, roll } = cropRegion;

  // We build TWO geometries from the same 8 corner vertices:
  //  - an edge wireframe (lineSegments) to show the crop volume
  //  - a filled mesh of the top quad, so the user can actually see
  //    the tilted clip plane and how it moves through the cloud
  //    when they change yaw / pitch / roll. The 4-vertex mesh
  //    re-uses only indices 4..7 of the corner array, which is why
  //    we build a separate small BufferGeometry for it.
  const { lineGeometry, capGeometry } = useMemo(() => {
    const cosA = Math.cos(angle);
    const sinA = Math.sin(angle);
    const slopeX = Math.tan(roll);
    const slopeZ = Math.tan(pitch);

    // Local corners CCW in the rectangle's XZ frame.
    const localCorners: Array<[number, number]> = [
      [-halfW, -halfL],
      [ halfW, -halfL],
      [ halfW,  halfL],
      [-halfW,  halfL],
    ];

    // 8 vertices, floats packed xyz·8.
    const positions = new Float32Array(8 * 3);
    for (let i = 0; i < 4; i++) {
      const [lx, lz] = localCorners[i];
      const wx = cx + cosA * lx - sinA * lz;
      const wz = cz + sinA * lx + cosA * lz;

      positions[i * 3]     = wx;
      positions[i * 3 + 1] = floorY;
      positions[i * 3 + 2] = wz;

      const topY = zCeiling + slopeX * lx + slopeZ * lz;
      const ti = (i + 4) * 3;
      positions[ti]     = wx;
      positions[ti + 1] = topY;
      positions[ti + 2] = wz;
    }

    const indices = new Uint16Array([
      0, 1, 1, 2, 2, 3, 3, 0,   // bottom edges
      4, 5, 5, 6, 6, 7, 7, 4,   // top edges (= the clip plane outline)
      0, 4, 1, 5, 2, 6, 3, 7,   // verticals
    ]);

    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    lineGeo.setIndex(new THREE.BufferAttribute(indices, 1));

    // Clip-plane cap — a single quad using the 4 top corners. We pack
    // just those four vertices (so no index remapping is needed) and
    // draw two CCW triangles covering the rectangle.
    const capPositions = new Float32Array(4 * 3);
    for (let i = 0; i < 4; i++) {
      capPositions[i * 3]     = positions[(i + 4) * 3];
      capPositions[i * 3 + 1] = positions[(i + 4) * 3 + 1];
      capPositions[i * 3 + 2] = positions[(i + 4) * 3 + 2];
    }
    const capIndices = new Uint16Array([0, 1, 2, 0, 2, 3]);
    const capGeo = new THREE.BufferGeometry();
    capGeo.setAttribute("position", new THREE.BufferAttribute(capPositions, 3));
    capGeo.setIndex(new THREE.BufferAttribute(capIndices, 1));

    return { lineGeometry: lineGeo, capGeometry: capGeo };
  }, [cx, cz, halfW, halfL, angle, pitch, roll, zCeiling, floorY]);

  // Free GPU buffers when the component unmounts or geometry is
  // rebuilt — React's ref-cleanup runs before the next useMemo result
  // is committed, so this catches every rebuild.
  useEffect(() => {
    return () => {
      lineGeometry.dispose();
      capGeometry.dispose();
    };
  }, [lineGeometry, capGeometry]);

  return (
    <group>
      {/* Filled top face — translucent amber so you can still see
          through it to the cloud below, but obvious enough to read
          at a glance where points are being clipped. Double-sided
          so the camera orientation in 3D doesn't hide it. */}
      {showCap && (
        <mesh renderOrder={8}>
          <primitive object={capGeometry} attach="geometry" />
          <meshBasicMaterial
            color="#facc15"
            transparent
            opacity={0.18}
            side={THREE.DoubleSide}
            depthTest={false}
            depthWrite={false}
          />
        </mesh>
      )}
      {/* Edge wireframe — the full parallelepiped outline. */}
      <lineSegments renderOrder={9}>
        <primitive object={lineGeometry} attach="geometry" />
        <lineBasicMaterial
          color="#facc15"
          transparent
          opacity={0.9}
          depthTest={false}
          depthWrite={false}
        />
      </lineSegments>
    </group>
  );
}

/**
 * Thin wrapper around `<input type="number">` that keeps its own
 * string buffer so intermediate keystrokes (`-`, `1.`, empty) don't
 * get clobbered by a round-trip through the numeric prop.
 *
 * Commits valid numbers to `onChange` immediately on every keystroke,
 * but only clamps to [min, max] on blur — letting the user type a
 * number that's temporarily below `min` (e.g. typing "3" before "30")
 * without the field fighting them. If the blur value is not a number
 * at all, the display re-syncs to the last committed `value`.
 */
function NumericInput({
  value,
  step,
  min,
  max,
  onChange,
  fmt = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(2)),
  className,
  disabled,
}: {
  value: number;
  step?: number;
  min?: number;
  max?: number;
  onChange: (v: number) => void;
  fmt?: (v: number) => string;
  className?: string;
  disabled?: boolean;
}) {
  const [text, setText] = useState(() => fmt(value));
  useEffect(() => {
    setText(fmt(value));
  }, [value, fmt]);
  return (
    <input
      type="number"
      value={text}
      step={step}
      min={min}
      max={max}
      disabled={disabled}
      onChange={(e) => {
        setText(e.target.value);
        const n = parseFloat(e.target.value);
        if (!Number.isNaN(n)) onChange(n);
      }}
      onBlur={() => {
        const n = parseFloat(text);
        if (Number.isNaN(n)) {
          setText(fmt(value));
          return;
        }
        let clamped = n;
        if (min !== undefined) clamped = Math.max(min, clamped);
        if (max !== undefined) clamped = Math.min(max, clamped);
        if (clamped !== n) {
          onChange(clamped);
          setText(fmt(clamped));
        }
      }}
      className={className}
    />
  );
}

/**
 * Slider + editable numeric input for a single angle (degrees in the
 * UI, radians in state). Both controls write to the same `onChangeRad`
 * so dragging the slider and typing a number stay in sync.
 *
 * Double-clicking the slider resets to 0° — cheap "snap to flat"
 * without needing a per-field reset button.
 */
function AngleField({
  label,
  title,
  icon,
  min,
  max,
  step,
  valueRad,
  onChangeRad,
}: {
  label: string;
  title: string;
  icon: React.ReactNode;
  min: number;
  max: number;
  step: number;
  valueRad: number;
  onChangeRad: (rad: number) => void;
}) {
  const DEG = (r: number) => (r * 180) / Math.PI;
  const RAD = (d: number) => (d * Math.PI) / 180;
  // The text input carries its own string state so intermediate
  // keystrokes like "-" or "1." are not clobbered by the radian
  // round-trip. It re-syncs whenever the external radian value
  // changes (e.g. slider drag, Reset button).
  const [text, setText] = useState(() => DEG(valueRad).toFixed(1));
  useEffect(() => {
    setText(DEG(valueRad).toFixed(1));
  }, [valueRad]);

  return (
    <div className="flex items-center gap-1.5" title={title}>
      <svg className="w-3 h-3 text-white/40" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
        {icon}
      </svg>
      <span className="text-[10px] font-mono text-white/40">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={Number(DEG(valueRad).toFixed(1))}
        onChange={(e) => onChangeRad(RAD(parseFloat(e.target.value)))}
        onDoubleClick={() => onChangeRad(0)}
        className="w-24 accent-amber-300 cursor-pointer"
      />
      <input
        type="number"
        value={text}
        step={step}
        onChange={(e) => {
          setText(e.target.value);
          const n = parseFloat(e.target.value);
          if (!Number.isNaN(n)) onChangeRad(RAD(n));
        }}
        onBlur={() => {
          const n = parseFloat(text);
          if (Number.isNaN(n)) setText(DEG(valueRad).toFixed(1));
        }}
        className="w-14 bg-white/5 border border-white/10 rounded px-1.5 py-0.5 text-[10px] font-mono text-amber-200/90 text-right focus:outline-none focus:border-amber-300/60"
      />
      <span className="text-[10px] font-mono text-white/40">°</span>
    </div>
  );
}

/**
 * Slider + numeric input pair for the crop-region scalar fields
 * (center X, center Z, width, length). Shares the same state
 * between both controls via `value` / `onChange`, so dragging and
 * typing are interchangeable.
 */
function CropScalarField({
  label,
  unit,
  min,
  max,
  step,
  value,
  onChange,
  fmt,
}: {
  label: string;
  unit?: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
  fmt: (v: number) => string;
}) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[10px] font-mono text-white/40">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-20 accent-amber-300 cursor-pointer"
      />
      <NumericInput
        value={value}
        step={step}
        min={min}
        onChange={onChange}
        fmt={fmt}
        className="w-14 bg-white/5 border border-white/10 rounded px-1.5 py-0.5 text-[10px] font-mono text-amber-200/90 text-right focus:outline-none focus:border-amber-300/60"
      />
      {unit && <span className="text-[10px] font-mono text-white/40">{unit}</span>}
    </div>
  );
}

interface SceneViewerProps {
  geometry: THREE.BufferGeometry | null;
  /** Rendered cloud tiles. Owned/produced by the parent's worker. */
  chunks: PointCloudChunk[];
  fileName: string;
  /** Voxel downsample size controlled by the slider below. Lives at the
   *  parent level so voxel changes post directly to the parent's worker
   *  without re-transferring the cloud. */
  voxelSize: number;
  onVoxelSizeChange: (v: number) => void;
  /** Live status from the parent's worker pipeline. */
  cloudStats: PointCloudStats | null;
  onReset: () => void;
}

export function SceneViewer({
  geometry,
  chunks,
  fileName,
  voxelSize,
  onVoxelSizeChange,
  cloudStats,
  onReset,
}: SceneViewerProps) {
  const [cameraMode,    setCameraMode]    = useState<CameraMode>("3d");
  const [pointSize3d,   setPointSize3d]   = useState(0.03);
  const [pointSize2d,   setPointSize2d]   = useState(2);
  const [showStats,     setShowStats]     = useState(false);
  const [tool,          setTool]          = useState<Tool>("view");
  const [width,         setWidth]         = useState(3);

  const [lanelets,     setLanelets]     = useState<Lanelet[]>([]);
  const [registry,     setRegistry]     = useState<NodeRegistry>(createRegistry());
  const [pendingStart, setPendingStart] = useState<Vec3 | null>(null);
  /** Remembered between click-1 and click-2 so the first click's pair-attach
   *  flows into createLanelet even though creation only happens on click-2. */
  const [pendingStartSnap, setPendingStartSnap] =
    useState<LaneletEndSnap | null>(null);
  const [selectedIds,  setSelectedIds]  = useState<Set<number>>(new Set());

  const nextLaneletIdRef = useRef(1);
  const nextNodeIdRef    = useRef(1);
  const pointsRef        = useRef<THREE.Group | null>(null);

  const pointSize = cameraMode === "3d" ? pointSize3d : pointSize2d;

  const bbox  = geometry?.boundingBox ?? null;
  const yMin  = bbox ? bbox.min.y : 0;
  const yMax  = bbox ? bbox.max.y : 1;
  const ySpan = Math.max(yMax - yMin, 1e-6);
  const yStep = ySpan / 200;
  const [zCeiling, setZCeiling] = useState<number>(yMax);

  // Oriented XZ crop region the Y-clip is scoped to. When disabled, the
  // slider behaves globally (original behaviour).
  //
  // `angle` is the yaw of the rectangle's local X axis away from world X,
  // in radians, right-handed around +Y (matching Three.js convention and
  // the shader's uCropRot sign). Exposed to the user in degrees.
  //
  // Motivation: a hillside street that runs diagonally needs a rotated
  // rectangle, otherwise the user has to oversize W and L to cover the
  // road, which brings back the same diagonal-cut problem we're trying
  // to avoid.
  interface CropState {
    enabled: boolean;
    cx:      number;
    cz:      number;
    width:   number;   // size along the rectangle's local X
    length:  number;   // size along the rectangle's local Z
    angle:   number;   // yaw   around +Y,       radians
    pitch:   number;   // tilt  around local X,  radians (street uphill)
    roll:    number;   // tilt  around local Z,  radians (camber / side)
  }
  const [crop, setCrop] = useState<CropState>({
    enabled: false,
    cx:      0,
    cz:      0,
    width:   20,
    length:  20,
    angle:   0,
    pitch:   0,
    roll:    0,
  });
  // Independent of the crop volume itself — whether the tilted top
  // face of the crop box renders as a translucent colored quad.
  // Wireframe always stays on; this just toggles the fill.
  const [showCropCap, setShowCropCap] = useState(true);

  // Reset crop center to the cloud's footprint centroid whenever a new
  // cloud is loaded — the old center would almost certainly sit outside
  // the new map. Width/length/angle keep the user's last choice so
  // reloading a similar map doesn't stomp on their setup.
  useEffect(() => {
    if (!geometry?.boundingBox) return;
    const b = geometry.boundingBox;
    setCrop((c) => ({
      ...c,
      enabled: false,
      cx: (b.min.x + b.max.x) * 0.5,
      cz: (b.min.z + b.max.z) * 0.5,
    }));
  }, [geometry]);

  // Packed uniform-ready region handed down to Scene / PointCloud.
  const cropRegion = useMemo<CropRegion | null>(
    () =>
      crop.enabled
        ? {
            cx:    crop.cx,
            cz:    crop.cz,
            halfW: crop.width  * 0.5,
            halfL: crop.length * 0.5,
            angle: crop.angle,
            pitch: crop.pitch,
            roll:  crop.roll,
          }
        : null,
    [
      crop.enabled,
      crop.cx,
      crop.cz,
      crop.width,
      crop.length,
      crop.angle,
      crop.pitch,
      crop.roll,
    ]
  );

  // Center X/Z slider ranges follow the map footprint, padded by the
  // rectangle's worst-case diagonal so the center can still slide past
  // the bbox edge even when the rectangle is rotated.
  const diag = Math.hypot(crop.width, crop.length);
  const xMin = bbox ? bbox.min.x - diag : -50;
  const xMax = bbox ? bbox.max.x + diag :  50;
  const zMin = bbox ? bbox.min.z - diag : -50;
  const zMax = bbox ? bbox.max.z + diag :  50;
  const xzStep = Math.max(0.01, (ySpan + Math.max(xMax - xMin, zMax - zMin)) / 2000);
  // Hard caps per user request — W/L range is simply 0–100 m regardless
  // of the underlying map size. Below 0.5 m the rectangle is effectively
  // a line (and the shader guard clamps to 1 mm anyway), so that's the
  // slider floor; step 0.5 gives decimetre precision across the range.
  const WL_MIN  = 0.5;
  const WL_MAX  = 100;
  const WL_STEP = 0.5;

  // Reset on new geometry
  useEffect(() => {
    if (geometry?.boundingBox) {
      setZCeiling(geometry.boundingBox.max.y);
    }
    setLanelets([]);
    setRegistry(createRegistry());
    setPendingStart(null);
    setPendingStartSnap(null);
    setSelectedIds(new Set());
    setTool("view");
    nextLaneletIdRef.current = 1;
    nextNodeIdRef.current    = 1;
  }, [geometry]);

  useEffect(() => {
    // Any drawing tool clears the current selection so clicks don't
    // accidentally toggle selection while you're trying to place points.
    if (tool !== "view") setSelectedIds(new Set());
  }, [tool]);

  // --------------------------------------------------------------------
  // Derived: resolved lanelets + junction markers
  // --------------------------------------------------------------------
  const resolvedLanelets = useMemo(
    () => resolveAll(registry, lanelets),
    [registry, lanelets]
  );

  const junctionPositions = useMemo<Vec3[]>(() => {
    const ids = junctionNodeIds(lanelets);
    const out: Vec3[] = [];
    for (const id of ids) {
      const p = getNodeOrNull(registry, id);
      if (p) out.push(p);
    }
    return out;
  }, [registry, lanelets]);

  // --------------------------------------------------------------------
  // Mutations
  // --------------------------------------------------------------------
  const deselectAll = () => setSelectedIds(new Set());

  // ─── Clipboard for Ctrl/Cmd + C / V on lanelets ──────────────
  //
  // The clipboard holds a self-contained snapshot: every NodeId
  // referenced by the copied lanelets has its position recorded,
  // keyed by the ORIGINAL NodeId. On paste we allocate a fresh
  // NodeId per original and rewrite the boundary arrays to use
  // the new IDs. Because shared NodeIds in the original group
  // map through the same entry in `idMap`, a copied pair of
  // connected lanelets stays connected in the paste — but never
  // attaches back to the source lanelets (no existing registry
  // IDs are reused).
  interface LaneletClipboard {
    lanelets: Omit<Lanelet, "id">[];
    nodes:    Record<NodeId, Vec3>;
  }
  const clipboardRef = useRef<LaneletClipboard | null>(null);

  const copySelectedLanelets = () => {
    if (selectedIds.size === 0) return;
    const selected: Lanelet[] = lanelets.filter((l) => selectedIds.has(l.id));
    if (selected.length === 0) return;

    const nodes: Record<NodeId, Vec3> = {};
    for (const l of selected) {
      for (const id of l.leftBoundary)  nodes[id] = getNode(registry, id);
      for (const id of l.rightBoundary) nodes[id] = getNode(registry, id);
    }

    // Drop the id and snapshot the boundaries so future edits to
    // the source lanelets don't mutate what's on the clipboard.
    const lanSnap: Omit<Lanelet, "id">[] = selected.map((l) => {
      const { id: _id, leftBoundary, rightBoundary, ...rest } = l;
      void _id;
      return {
        ...rest,
        leftBoundary:  [...leftBoundary],
        rightBoundary: [...rightBoundary],
      };
    });
    clipboardRef.current = { lanelets: lanSnap, nodes };
  };

  const pasteClipboard = () => {
    const clip = clipboardRef.current;
    if (!clip || clip.lanelets.length === 0) return;

    // Offset XZ so the paste doesn't overlap the source. Scale with
    // the scene size when known so huge maps don't end up with a
    // 3m nudge you can barely see, while small test scenes don't
    // get a 50m nudge off the cloud.
    const spanHint = geometry?.boundingSphere?.radius ?? 0;
    const OFF = Math.max(3, Math.min(12, spanHint * 0.02));

    let reg = registry;
    const idMap = new Map<NodeId, NodeId>();
    for (const [oldIdStr, pos] of Object.entries(clip.nodes)) {
      const oldId = Number(oldIdStr);
      const newPos: Vec3 = [pos[0] + OFF, pos[1], pos[2] + OFF];
      const { reg: reg2, id: newId } = addNode(reg, newPos, nextNodeIdRef);
      reg = reg2;
      idMap.set(oldId, newId);
    }

    const newLanelets: Lanelet[] = clip.lanelets.map((l) => ({
      ...l,
      id: nextLaneletIdRef.current++,
      leftBoundary:  l.leftBoundary.map((id)  => idMap.get(id) ?? id),
      rightBoundary: l.rightBoundary.map((id) => idMap.get(id) ?? id),
    }));

    setRegistry(reg);
    setLanelets((ls) => [...ls, ...newLanelets]);
    setSelectedIds(new Set(newLanelets.map((l) => l.id)));
  };

  const handleSelect = (id: number, shiftKey: boolean) => {
    setSelectedIds((prev) => {
      if (shiftKey) {
        const next = new Set(prev);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        return next;
      }
      return new Set([id]);
    });
  };

  const handleFinishLanelet = (reg: NodeRegistry, l: Lanelet) => {
    setRegistry(reg);
    setLanelets((ls) => [...ls, l]);
    setPendingStart(null);
    setPendingStartSnap(null);
  };

  /** Replace a single lanelet AND swap in the new registry atomically. */
  const handleUpsertLanelet = (reg: NodeRegistry, l: Lanelet) => {
    setRegistry(reg);
    setLanelets((ls) => ls.map((x) => (x.id === l.id ? l : x)));
  };

  const deleteIds = (ids: number[]) => {
    const idSet = new Set(ids);
    setLanelets((ls) => {
      const remaining = ls.filter((l) => !idSet.has(l.id));
      // GC orphaned nodes from the registry. Uses the CURRENT registry
      // snapshot — fine for sync user actions.
      setRegistry((reg) => gcUnused(reg, remaining));
      return remaining;
    });
    setSelectedIds((prev) => {
      const next = new Set(prev);
      for (const id of ids) next.delete(id);
      return next;
    });
  };

  const updateIds = (ids: number[], patch: Partial<Lanelet>) => {
    const idSet = new Set(ids);
    setLanelets((ls) =>
      ls.map((l) => (idSet.has(l.id) ? { ...l, ...patch } : l))
    );
  };

  const resizeWidthIds = (ids: number[], newWidth: number) => {
    const idSet = new Set(ids);
    let nextReg = registry;
    const nextLanelets = lanelets.map((l) => {
      if (!idSet.has(l.id)) return l;
      const { reg, lanelet } = resizeLaneletWidth(
        nextReg,
        nextNodeIdRef,
        l,
        newWidth
      );
      nextReg = reg;
      return lanelet;
    });
    setRegistry(gcUnused(nextReg, nextLanelets));
    setLanelets(nextLanelets);
  };

  const reverseIds = (ids: number[]) => {
    const idSet = new Set(ids);
    setLanelets((ls) =>
      ls.map((l) => (idSet.has(l.id) ? reverseLanelet(l) : l))
    );
  };

  // Toggle "rect lock" on a selection. Turning it ON also collapses
  // any existing curvature onto the straight start→end axis immediately
  // so the UI reflects the locked shape without waiting for a drag.
  // Turning it OFF just clears the flag — the current geometry stays.
  const handleToggleStraight = (id: number) => {
    const lanelet = lanelets.find((l) => l.id === id);
    if (!lanelet) return;
    setStraightIds([id], !lanelet.straight);
  };

  const handleTogglePositionLocked = (id: number) => {
    setLanelets((ls) =>
      ls.map((l) => (l.id === id ? { ...l, positionLocked: !l.positionLocked } : l))
    );
  };

  const setStraightIds = (ids: number[], straight: boolean) => {
    const idSet = new Set(ids);
    if (!straight) {
      setLanelets((ls) =>
        ls.map((l) => (idSet.has(l.id) ? { ...l, straight: false } : l))
      );
      return;
    }
    let nextReg = registry;
    const nextLanelets = lanelets.map((l) => {
      if (!idSet.has(l.id)) return l;
      const { reg } = straightenLanelet(nextReg, l);
      nextReg = reg;
      return { ...l, straight: true };
    });
    setRegistry(nextReg);
    setLanelets(nextLanelets);
  };

  /**
   * Fit-on-plane: collect PCD points that fall inside each selected lanelet's
   * XZ bounding box (+ half-width margin), run RANSAC to find the dominant
   * road plane while rejecting car-roof / noise outliers, then project every
   * boundary node onto that plane.
   */
  const fitPlaneIds = (ids: number[]) => {
    const group = pointsRef.current;
    if (!group) return;

    let reg = registry;

    for (const id of ids) {
      const lanelet = lanelets.find((l) => l.id === id);
      if (!lanelet) continue;

      // XZ bounding box of all boundary nodes + half-width margin
      const allNodeIds = new Set([
        ...lanelet.leftBoundary,
        ...lanelet.rightBoundary,
      ]);
      let xMin = Infinity, xMax = -Infinity;
      let zMin = Infinity, zMax = -Infinity;
      for (const nid of allNodeIds) {
        const p = getNode(reg, nid);
        if (p[0] < xMin) xMin = p[0];
        if (p[0] > xMax) xMax = p[0];
        if (p[2] < zMin) zMin = p[2];
        if (p[2] > zMax) zMax = p[2];
      }
      const margin = lanelet.width * 0.5;
      xMin -= margin; xMax += margin;
      zMin -= margin; zMax += margin;

      // Collect PCD points inside the box that are below the crop ceiling
      const raw: number[] = [];
      group.traverse((obj) => {
        if (!(obj instanceof THREE.Points)) return;
        const pos = obj.geometry.attributes.position as THREE.BufferAttribute;
        for (let i = 0; i < pos.count; i++) {
          const x = pos.getX(i);
          const z = pos.getZ(i);
          if (x < xMin || x > xMax || z < zMin || z > zMax) continue;
          const y = pos.getY(i);
          if (y > effectiveCeilingAt(x, z, zCeiling, cropRegion)) continue;
          raw.push(x, y, z);
        }
      });

      const count = raw.length / 3;
      if (count < 3) continue;

      const plane = fitPlaneRansac(new Float32Array(raw), count);
      if (!plane) continue;

      reg = applyPlaneToLanelet(reg, lanelet, plane.a, plane.b, plane.c);
    }

    setRegistry(reg);
  };

  const duplicateNeighbor = (sourceId: number, side: "left" | "right") => {
    const src = lanelets.find((l) => l.id === sourceId);
    if (!src) return;

    const { reg: newReg, updatedSource, neighbor } =
      side === "left"
        ? duplicateLaneletLeft(registry, nextNodeIdRef, src)
        : duplicateLaneletRight(registry, nextNodeIdRef, src);

    const newId = nextLaneletIdRef.current++;
    const newLanelet: Lanelet = { ...neighbor, id: newId };

    setRegistry(newReg);
    setLanelets((ls) =>
      ls.map((l) => (l.id === sourceId ? updatedSource : l)).concat(newLanelet)
    );
    setSelectedIds(new Set([newId]));
  };

  // Build a "connector" lanelet from `fromId`'s end to `toId`'s start. The
  // two end pairs of the connector adopt the existing corner NodeIds via
  // pair-attach, so A, connector and B become one connected chain with
  // shared junction NodeIds at both seams.
  //
  // We don't surface-snap interior Y at creation time: the endpoints
  // inherit their (already-snapped) Y from the existing lanelets, and the
  // interior waypoints average those Ys — close enough to the ground for
  // a first draft. Dragging any interior handle afterwards re-runs
  // `snapLaneletToSurface` on pointer-up.
  const createJoint = (fromId: number, toId: number, type: JointType) => {
    if (fromId === toId) return;
    const from = lanelets.find((l) => l.id === fromId);
    const to   = lanelets.find((l) => l.id === toId);
    if (!from || !to) return;

    const width = (from.width + to.width) / 2;

    const { reg, lanelet } = joinLanelets({
      reg: registry,
      nextNodeId: nextNodeIdRef,
      laneletId: nextLaneletIdRef.current++,
      from,
      to,
      type,
      width,
    });

    setRegistry(reg);
    setLanelets((ls) => [...ls, lanelet]);
    setSelectedIds(new Set([lanelet.id]));
  };

  // Keyboard shortcuts
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)) return;

      if (e.key === "Escape") {
        if (tool === "crop-center") {
          setTool("view");
        } else if (pendingStart) {
          setPendingStart(null);
          setPendingStartSnap(null);
        }
        else if (selectedIds.size > 0) deselectAll();
      } else if (e.key === "Delete" || e.key === "Backspace") {
        if (selectedIds.size > 0) {
          e.preventDefault();
          deleteIds(Array.from(selectedIds));
        }
      } else if ((e.ctrlKey || e.metaKey) && !e.shiftKey && !e.altKey) {
        // Copy / paste. Lower-case check covers both "c" / "v" (the
        // plain key) and works uniformly across modifiers.
        const k = e.key.toLowerCase();
        if (k === "c" && selectedIds.size > 0) {
          e.preventDefault();
          copySelectedLanelets();
        } else if (k === "v" && clipboardRef.current) {
          e.preventDefault();
          pasteClipboard();
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingStart, selectedIds, tool, lanelets, registry]);

  /**
   * Toolbar button behaviour: clicking the same tool that's already active
   * turns it off (back to "view"); clicking a different tool switches to
   * it and discards any half-drawn pending start — otherwise the first
   * click of a crosswalk would connect back to a lanelet's pending start.
   */
  const pickTool = (next: Tool) => {
    if (tool === next) {
      setTool("view");
      setPendingStart(null);
      setPendingStartSnap(null);
      return;
    }
    setTool(next);
    setPendingStart(null);
    setPendingStartSnap(null);
  };

  /**
   * Consume a "set crop center" click from the PCD: recenter the rectangle
   * on the clicked point, enable the crop if it was off (otherwise the
   * click would do nothing visible), and return to "view" mode.
   *
   * Only X and Z are used — the rectangle is always axis-aligned and
   * sits vertically from scene floor to zCeiling, so the click's Y
   * doesn't matter.
   */
  const handlePickCropCenter = (p: Vec3) => {
    setCrop((c) => ({ ...c, enabled: true, cx: p[0], cz: p[2] }));
    setTool("view");
  };

  const handleStartLanelet = (p: Vec3, snap: LaneletEndSnap | null) => {
    setPendingStart(p);
    setPendingStartSnap(snap);
  };

  const clearAll = () => {
    setLanelets([]);
    setRegistry(createRegistry());
    setPendingStart(null);
    setPendingStartSnap(null);
    setSelectedIds(new Set());
    nextLaneletIdRef.current = 1;
    nextNodeIdRef.current    = 1;
  };

  /**
   * Export the current map as a Lanelet2 OSM file. The output filename
   * piggybacks on the loaded PCD name when available so the exported map
   * lives next to its source cloud ("foo.pcd" → "foo.osm"); falls back to
   * MapToolbox's default ("lanelet2_map.osm") otherwise.
   */
  const exportOsm = () => {
    const base =
      fileName && fileName.toLowerCase().endsWith(".pcd")
        ? fileName.slice(0, -4)
        : fileName || "lanelet2_map";
    downloadLanelet2Osm(registry, lanelets, `${base}.osm`);
  };

  return (
    <div className="relative w-full h-full">
      <Canvas
        camera={{ fov: 60, near: 0.01, far: 100000, position: [50, 30, 50] }}
        gl={{ antialias: true, alpha: false }}
        dpr={[1, 1.5]}
        onPointerMissed={() => {
          if (tool === "view" && selectedIds.size > 0) deselectAll();
        }}
      >
        <Scene
          geometry={geometry}
          chunks={chunks}
          pointSize={pointSize}
          zCeiling={zCeiling}
          cameraMode={cameraMode}
          tool={tool}
          width={width}
          cropRegion={cropRegion}
          showCropCap={showCropCap}
          onPickCropCenter={handlePickCropCenter}
          registry={registry}
          nextNodeIdRef={nextNodeIdRef}
          lanelets={lanelets}
          resolvedLanelets={resolvedLanelets}
          junctionPositions={junctionPositions}
          selectedIds={selectedIds}
          onSelect={handleSelect}
          pendingStart={pendingStart}
          pendingStartSnap={pendingStartSnap}
          onStartLanelet={handleStartLanelet}
          onFinishLanelet={handleFinishLanelet}
          onUpsertLanelet={handleUpsertLanelet}
          onToggleStraight={handleToggleStraight}
          onTogglePositionLocked={handleTogglePositionLocked}
          nextLaneletIdRef={nextLaneletIdRef}
          pointsRef={pointsRef}
        />
        {showStats && <Stats className="!left-auto !right-4 !top-4" />}
      </Canvas>

      {/* ── Top bar ─────────────────────────────────────────── */}
      <div className="absolute top-0 left-0 right-0 flex items-center justify-between px-5 py-3 bg-gradient-to-b from-black/60 to-transparent pointer-events-none">

        <div className="flex items-center gap-4 pointer-events-auto">
          <div className="flex items-center gap-2.5">
            <div className="h-2 w-2 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(0,220,255,0.8)]" />
            <span className="text-xs font-semibold tracking-widest text-white/50 uppercase font-mono">
              VectorMap Builder
            </span>
          </div>

          <div className="flex items-center rounded-lg border border-white/10 overflow-hidden bg-black/50 backdrop-blur-sm">
            <button
              onClick={() => setCameraMode("3d")}
              className={`flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-mono transition-colors cursor-pointer border-r border-white/10
                ${cameraMode === "3d"
                  ? "bg-cyan-500/20 text-cyan-300"
                  : "text-white/40 hover:text-white/70"}`}
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth={1.8} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="m21 7.5-9-5.25L3 7.5m18 0-9 5.25m9-5.25v9l-9 5.25M3 7.5l9 5.25M3 7.5v9l9 5.25m0-9v9" />
              </svg>
              3D
            </button>
            <button
              onClick={() => setCameraMode("2d")}
              className={`flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-mono transition-colors cursor-pointer
                ${cameraMode === "2d"
                  ? "bg-cyan-500/20 text-cyan-300"
                  : "text-white/40 hover:text-white/70"}`}
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth={1.8} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M9 6.75V15m6-6v8.25m.503 3.498 4.875-2.437c.381-.19.622-.58.622-1.006V4.82c0-.836-.88-1.38-1.628-1.006l-3.869 1.934c-.317.159-.69.159-1.006 0L9.503 3.252a1.125 1.125 0 0 0-1.006 0L3.622 5.689C3.24 5.88 3 6.27 3 6.695V19.18c0 .836.88 1.38 1.628 1.006l3.869-1.934c.317-.159.69-.159 1.006 0l4.994 2.497c.317.158.69.158 1.006 0Z" />
              </svg>
              2D
            </button>
          </div>
        </div>

        {fileName && (
          <div className="flex items-center gap-3 pointer-events-auto">
            <div className="flex items-center gap-2 rounded-md border border-white/10 bg-white/5 px-3 py-1.5 backdrop-blur-sm">
              <svg className="w-3.5 h-3.5 text-cyan-400" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z" />
              </svg>
              <span className="text-xs text-white/70 font-mono max-w-[160px] truncate">{fileName}</span>
            </div>
            {/* Cloud stats. Shows live progress while the worker
                is voxeling / chunking a freshly loaded cloud, then
                collapses to "raw → voxel · chunk count" when ready. */}
            <div className="rounded-md border border-white/10 bg-white/5 px-3 py-1.5 backdrop-blur-sm">
              {cloudStats?.phase === "processing" ? (
                <div className="flex items-center gap-2">
                  <div className="relative h-2 w-2">
                    <div className="absolute inset-0 rounded-full bg-amber-400 animate-pulse" />
                  </div>
                  <span className="text-[11px] text-amber-200/90 font-mono tracking-wide">
                    {cloudStats.message ?? "Processing…"}
                  </span>
                  <span className="text-[11px] text-amber-200/70 font-mono">
                    {cloudStats.progress != null
                      ? `${Math.round(cloudStats.progress * 100)}%`
                      : ""}
                  </span>
                </div>
              ) : cloudStats?.phase === "ready" ? (
                <span
                  className="text-xs text-white/50 font-mono"
                  title={`Raw points: ${cloudStats.rawPoints.toLocaleString()}`}
                >
                  {cloudStats.rawPoints.toLocaleString()} →{" "}
                  <span className="text-cyan-300/90">
                    {cloudStats.voxelPoints.toLocaleString()}
                  </span>{" "}
                  pts · {cloudStats.chunkCount} tiles
                </span>
              ) : cloudStats?.phase === "error" ? (
                <span className="text-xs text-rose-300/90 font-mono" title={cloudStats.message}>
                  cloud err — fallback
                </span>
              ) : (
                <span className="text-xs text-white/50 font-mono">
                  cloud
                </span>
              )}
            </div>
          </div>
        )}
      </div>

      {/* ── Properties panel (when something is selected) ───── */}
      <LaneletProperties
        lanelets={lanelets}
        selectedIds={selectedIds}
        onUpdate={updateIds}
        onResizeWidth={resizeWidthIds}
        onReverse={reverseIds}
        onDelete={deleteIds}
        onDeselectAll={deselectAll}
        onDuplicateNeighbor={duplicateNeighbor}
        onCreateJoint={createJoint}
        onSetStraight={setStraightIds}
        onFitPlane={fitPlaneIds}
      />

      {/* ── Tool panel ──────────────────────────────────────── */}
      <div className="absolute top-20 left-5 flex flex-col gap-2 rounded-xl border border-white/10 bg-black/60 p-3 backdrop-blur-md w-56">
        <div className="text-[10px] font-mono text-white/40 uppercase tracking-wider">Tool</div>

        <button
          onClick={() => pickTool("lanelet")}
          className={`flex items-center justify-between px-3 py-2 rounded-md text-xs font-mono transition-colors cursor-pointer border
            ${tool === "lanelet"
              ? "bg-cyan-500/20 text-cyan-300 border-cyan-400/40"
              : "bg-white/5 text-white/60 border-white/10 hover:text-white/80 hover:bg-white/10"}`}
        >
          <span className="flex items-center gap-2">
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth={1.8} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M4 19 L10 5 L14 5 L20 19 Z" />
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M12 6 L12 18" strokeDasharray="1.5 1.5" />
            </svg>
            Lanelet
          </span>
          <span className="text-[10px] text-white/40">{tool === "lanelet" ? "ON" : "OFF"}</span>
        </button>

        <button
          onClick={() => pickTool("crosswalk")}
          className={`flex items-center justify-between px-3 py-2 rounded-md text-xs font-mono transition-colors cursor-pointer border
            ${tool === "crosswalk"
              ? "bg-sky-500/20 text-sky-300 border-sky-400/40"
              : "bg-white/5 text-white/60 border-white/10 hover:text-white/80 hover:bg-white/10"}`}
        >
          <span className="flex items-center gap-2">
            {/* Zebra-stripe glyph — a rectangle with vertical bars, matching
                the on-canvas crosswalk look. */}
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth={1.6} viewBox="0 0 24 24">
              <rect x="4" y="5" width="16" height="14" rx="1" />
              <path d="M8 5v14M12 5v14M16 5v14" />
            </svg>
            Crosswalk
          </span>
          <span className="text-[10px] text-white/40">{tool === "crosswalk" ? "ON" : "OFF"}</span>
        </button>

        <div className="flex flex-col gap-1.5 pt-1">
          <div className="flex items-center justify-between gap-2">
            <span className="text-[10px] font-mono text-white/40 uppercase tracking-wider">Width</span>
            <div className="flex items-center gap-1">
              <NumericInput
                value={width}
                step={0.1}
                min={0.5}
                max={10}
                onChange={setWidth}
                fmt={(v) => v.toFixed(1)}
                className="w-14 bg-white/5 border border-white/10 rounded px-1.5 py-0.5 text-[11px] font-mono text-white/80 text-right focus:outline-none focus:border-cyan-400/50"
              />
              <span className="text-[10px] font-mono text-white/40">m</span>
            </div>
          </div>
          <input
            type="range"
            min={0.5}
            max={10}
            step={0.1}
            value={width}
            onChange={(e) => setWidth(parseFloat(e.target.value))}
            className="w-full accent-cyan-400 cursor-pointer"
          />
        </div>

        <div className="flex items-center justify-between pt-1">
          <span className="text-[10px] font-mono text-white/40 uppercase">Lanelets</span>
          <span className="text-xs font-mono text-white/70">{lanelets.length}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-[10px] font-mono text-white/40 uppercase">Junctions</span>
          <span className="text-xs font-mono text-white/70">{junctionPositions.length}</span>
        </div>

        {pendingStart && (
          <div className={`rounded-md px-2.5 py-1.5 text-[10px] font-mono leading-4 border
            ${tool === "crosswalk"
              ? "bg-sky-500/10 border-sky-400/30 text-sky-300"
              : "bg-cyan-500/10 border-cyan-400/30 text-cyan-300"}`}>
            Click the end point…
            <br />
            <span className="text-white/50">Esc to cancel</span>
          </div>
        )}

        <button
          onClick={exportOsm}
          disabled={lanelets.length === 0}
          title="Export current map as Lanelet2 OSM"
          className="flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-md text-[11px] font-mono text-emerald-300 border border-emerald-400/40 bg-emerald-500/10 hover:bg-emerald-500/20 transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-emerald-500/10"
        >
          <svg className="w-3 h-3" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round"
              d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3" />
          </svg>
          Download .osm
        </button>

        <button
          onClick={clearAll}
          disabled={lanelets.length === 0 && !pendingStart}
          className="flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-md text-[11px] font-mono text-white/50 border border-white/10 hover:text-white/80 hover:bg-white/5 transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <svg className="w-3 h-3" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round"
              d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0" />
          </svg>
          Clear all
        </button>
      </div>

      {/* ── Z-clip region panel (appears when region is on) ─── */}
      {crop.enabled && bbox && (
        <div className="absolute bottom-[72px] left-1/2 -translate-x-1/2 flex flex-col gap-2 rounded-xl border border-amber-300/30 bg-black/70 px-4 py-2 backdrop-blur-md shadow-[0_0_20px_rgba(251,191,36,0.08)]">
          {/* ── Row 1: placement + footprint ─────────────────── */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5">
              <div className="h-1.5 w-1.5 rounded-full bg-amber-300 shadow-[0_0_6px_rgba(251,191,36,0.9)]" />
              <span className="text-[10px] font-mono uppercase tracking-wider text-amber-200/80">Z-clip region</span>
            </div>

            <div className="w-px h-5 bg-white/10" />

            {/* Toggle the translucent amber fill on the clip plane.
                The wireframe stays visible either way — this just
                hides the colored face for users who want an
                uncluttered view of the points being clipped. */}
            <button
              onClick={() => setShowCropCap((v) => !v)}
              title={showCropCap ? "Hide colored clip face" : "Show colored clip face"}
              className={`flex items-center gap-1.5 text-[10px] font-mono px-2 py-1 rounded-md border transition-colors cursor-pointer
                ${showCropCap
                  ? "bg-amber-400/20 border-amber-300/50 text-amber-100"
                  : "bg-white/5 border-white/10 text-white/60 hover:text-white hover:bg-white/10"}`}
            >
              <svg className="w-3 h-3" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                {showCropCap ? (
                  <>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M2.036 12.322a1.012 1.012 0 0 1 0-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178Z" />
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
                  </>
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3.98 8.223A10.477 10.477 0 0 0 1.934 12c1.292 4.338 5.31 7.5 10.066 7.5.993 0 1.953-.138 2.863-.395M6.228 6.228A10.451 10.451 0 0 1 12 4.5c4.756 0 8.773 3.162 10.065 7.498a10.522 10.522 0 0 1-4.293 5.774M6.228 6.228 3 3m3.228 3.228 3.65 3.65m7.894 7.894L21 21m-3.228-3.228-3.65-3.65m0 0a3 3 0 1 0-4.243-4.243m4.242 4.242L9.88 9.88" />
                )}
              </svg>
              {showCropCap ? "Face on" : "Face off"}
            </button>

            <button
              onClick={() => setTool(tool === "crop-center" ? "view" : "crop-center")}
              className={`flex items-center gap-1.5 text-[10px] font-mono px-2 py-1 rounded-md border transition-colors cursor-pointer
                ${tool === "crop-center"
                  ? "bg-amber-400/30 border-amber-300/60 text-amber-100"
                  : "bg-white/5 border-white/10 text-white/70 hover:text-white hover:bg-white/10"}`}
            >
              <svg className="w-3 h-3" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                <circle cx="12" cy="12" r="3" />
                <path strokeLinecap="round" d="M12 2v4M12 18v4M2 12h4M18 12h4" />
              </svg>
              {tool === "crop-center" ? "Click on the cloud…" : "Pick center"}
            </button>

            <button
              onClick={() => {
                const b = geometry!.boundingBox!;
                setCrop((c) => ({
                  ...c,
                  cx: (b.min.x + b.max.x) * 0.5,
                  cz: (b.min.z + b.max.z) * 0.5,
                }));
              }}
              title="Recenter on the cloud"
              className="flex items-center gap-1 text-[10px] font-mono px-2 py-1 rounded-md border bg-white/5 border-white/10 text-white/70 hover:text-white hover:bg-white/10 cursor-pointer"
            >
              <svg className="w-3 h-3" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                <circle cx="12" cy="12" r="8" />
                <circle cx="12" cy="12" r="2" fill="currentColor" />
              </svg>
              Recenter
            </button>

            <div className="w-px h-5 bg-white/10" />

            {/* Each of X / Z / W / L has a slider for quick drags
                and a numeric input right next to it for typing an
                exact value. Both write to the same piece of state
                so the two controls stay in lock-step. */}
            <CropScalarField
              label="X"
              min={xMin} max={xMax} step={xzStep}
              value={crop.cx}
              onChange={(v) => setCrop((c) => ({ ...c, cx: v }))}
              fmt={(v) => v.toFixed(1)}
            />

            {/* Center Z (ground-plane second axis; label as "Z" to match
                Three.js world coords). */}
            <CropScalarField
              label="Z"
              min={zMin} max={zMax} step={xzStep}
              value={crop.cz}
              onChange={(v) => setCrop((c) => ({ ...c, cz: v }))}
              fmt={(v) => v.toFixed(1)}
            />

            <div className="w-px h-5 bg-white/10" />

            {/* Rectangle width (local X) */}
            <CropScalarField
              label="W"
              unit="m"
              min={WL_MIN} max={WL_MAX} step={WL_STEP}
              value={crop.width}
              onChange={(v) => setCrop((c) => ({ ...c, width: v }))}
              fmt={(v) => v.toFixed(1)}
            />

            {/* Rectangle length (local Z) */}
            <CropScalarField
              label="L"
              unit="m"
              min={WL_MIN} max={WL_MAX} step={WL_STEP}
              value={crop.length}
              onChange={(v) => setCrop((c) => ({ ...c, length: v }))}
              fmt={(v) => v.toFixed(1)}
            />
          </div>

          {/* ── Row 2: clip-plane angles (yaw / pitch / roll) ───
              Each angle has both a slider (for dragging) and a text
              input (for punching in an exact value). Editing either
              immediately writes back to the same state. */}
          <div className="flex items-center gap-3 pl-[110px]">
            <span className="text-[9px] font-mono uppercase tracking-wider text-amber-200/60">Angle</span>

            <button
              onClick={() =>
                setCrop((c) => ({ ...c, angle: 0, pitch: 0, roll: 0 }))
              }
              title="Reset yaw · pitch · roll"
              className="text-[10px] font-mono px-2 py-0.5 rounded border bg-white/5 border-white/10 text-white/60 hover:text-white hover:bg-white/10 cursor-pointer"
            >
              Reset
            </button>

            <div className="w-px h-5 bg-white/10" />

            {/* Yaw — spin the rectangle around world +Y. Lets the
                user line the crop axis up with a diagonal road. */}
            <AngleField
              label="Yaw"
              title="Yaw · rotate rectangle around vertical axis"
              icon={
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M4 12a8 8 0 1 0 16 0M4 12l3-3M4 12l3 3" />
              }
              min={-180}
              max={180}
              step={0.5}
              valueRad={crop.angle}
              onChangeRad={(rad) => setCrop((c) => ({ ...c, angle: rad }))}
            />

            {/* Pitch — tilt the clip plane along the rectangle's length.
                Positive = plane rises toward local +Z (street going
                uphill along its length). Shader clamps to ±85°. */}
            <AngleField
              label="Pitch"
              title="Pitch · tilt clip plane along length (street uphill)"
              icon={
                <>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 17 L21 7" />
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 17 L21 17" strokeDasharray="2 2" opacity="0.5" />
                </>
              }
              min={-60}
              max={60}
              step={0.5}
              valueRad={crop.pitch}
              onChangeRad={(rad) => setCrop((c) => ({ ...c, pitch: rad }))}
            />

            {/* Roll — tilt the clip plane across the rectangle's width.
                Positive = plane rises toward local +X (road camber). */}
            <AngleField
              label="Roll"
              title="Roll · tilt clip plane across width (camber)"
              icon={
                <>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M21 17 L3 7" />
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 17 L21 17" strokeDasharray="2 2" opacity="0.5" />
                </>
              }
              min={-60}
              max={60}
              step={0.5}
              valueRad={crop.roll}
              onChangeRad={(rad) => setCrop((c) => ({ ...c, roll: rad }))}
            />
          </div>
        </div>
      )}

      {/* ── Bottom toolbar ──────────────────────────────────── */}
      <div className="absolute bottom-5 left-1/2 -translate-x-1/2 flex items-center gap-2 rounded-xl border border-white/10 bg-black/60 px-4 py-2.5 backdrop-blur-md">

        <div className="flex items-center gap-2.5">
          <svg className="w-3.5 h-3.5 text-white/40 shrink-0" fill="currentColor" viewBox="0 0 20 20">
            <circle cx="10" cy="10" r="2" />
          </svg>
          {cameraMode === "3d" ? (
            <input
              key="size-3d"
              type="range" min="0.005" max="0.2" step="0.005"
              value={pointSize3d}
              onChange={(e) => setPointSize3d(parseFloat(e.target.value))}
              className="w-24 accent-cyan-400 cursor-pointer"
            />
          ) : (
            <input
              key="size-2d"
              type="range" min="0.5" max="10" step="0.5"
              value={pointSize2d}
              onChange={(e) => setPointSize2d(parseFloat(e.target.value))}
              className="w-24 accent-cyan-400 cursor-pointer"
            />
          )}
          <span className="text-xs text-white/30 font-mono w-14">
            {cameraMode === "3d"
              ? `${pointSize3d.toFixed(3)} m`
              : `${pointSize2d.toFixed(1)} px`}
          </span>
        </div>

        <div className="w-px h-5 bg-white/10 mx-1" />

        <div className="flex items-center gap-2.5">
          <span className="text-[10px] font-mono text-white/40 uppercase tracking-wider shrink-0">Voxel</span>
          <input
            type="range" min="0" max="0.5" step="0.01"
            value={voxelSize}
            onChange={(e) => onVoxelSizeChange(parseFloat(e.target.value))}
            className="w-24 accent-cyan-400 cursor-pointer"
          />
          <span className={`text-xs font-mono w-10 ${voxelSize > 0 ? "text-cyan-400" : "text-white/30"}`}>
            {voxelSize > 0 ? voxelSize.toFixed(2) : "OFF"}
          </span>
        </div>

        <div className="w-px h-5 bg-white/10 mx-1" />

        <div className="flex items-center gap-2.5">
          <svg className="w-3.5 h-3.5 text-white/40 shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 19.5v-15m0 0-6.75 6.75M12 4.5l6.75 6.75" />
          </svg>
          <span className="text-[10px] font-mono text-white/40 uppercase tracking-wider shrink-0">Z&nbsp;clip</span>
          <input
            type="range"
            min={yMin}
            max={yMax}
            step={yStep}
            value={zCeiling}
            onChange={(e) => setZCeiling(parseFloat(e.target.value))}
            disabled={!bbox}
            className="w-24 accent-cyan-400 cursor-pointer disabled:opacity-40"
          />
          <NumericInput
            value={zCeiling}
            step={yStep}
            min={yMin}
            max={yMax}
            onChange={setZCeiling}
            disabled={!bbox}
            fmt={(v) => v.toFixed(2)}
            className={`w-16 bg-white/5 border rounded px-1.5 py-0.5 text-[11px] font-mono text-right focus:outline-none disabled:opacity-40
              ${zCeiling < yMax
                ? "border-cyan-400/40 text-cyan-300 focus:border-cyan-300"
                : "border-white/10 text-white/40 focus:border-white/30"}`}
          />
          <span className="text-[10px] font-mono text-white/40">m</span>
          <button
            onClick={() => setCrop((c) => ({ ...c, enabled: !c.enabled }))}
            disabled={!bbox}
            title={
              crop.enabled
                ? "Z-clip only affects the yellow region"
                : "Limit Z-clip to a rectangular region"
            }
            className={`flex items-center gap-1 text-[10px] font-mono px-2 py-1 rounded-md border transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed
              ${crop.enabled
                ? "bg-amber-400/20 border-amber-300/40 text-amber-200"
                : "bg-white/5 border-white/10 text-white/50 hover:text-white/80"}`}
          >
            <svg className="w-3 h-3" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M9 4.5h10.5v10.5M14.5 19.5H4V9" />
            </svg>
            Region
          </button>
        </div>

        <div className="w-px h-5 bg-white/10 mx-1" />

        <button
          onClick={() => setShowStats((s) => !s)}
          className={`text-xs font-mono px-2.5 py-1 rounded-md transition-colors cursor-pointer
            ${showStats ? "bg-cyan-500/20 text-cyan-300" : "text-white/40 hover:text-white/70"}`}
        >
          FPS
        </button>

        <div className="w-px h-5 bg-white/10 mx-1" />

        <button
          onClick={onReset}
          className="flex items-center gap-1.5 text-xs font-mono text-white/40 hover:text-white/70 transition-colors px-1 cursor-pointer"
        >
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round"
              d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5" />
          </svg>
          Load new
        </button>
      </div>

      {/* ── Hint ─────────────────────────────────────────────── */}
      <div className="absolute bottom-5 right-5 text-right pointer-events-none">
        {tool === "crop-center" ? (
          <p className="text-[10px] font-mono leading-5 text-amber-200/80">
            Click on the cloud · set region center · Esc to cancel
          </p>
        ) : tool !== "view" ? (
          pendingStart ? (
            <p className={`text-[10px] font-mono leading-5 ${tool === "crosswalk" ? "text-sky-300/80" : "text-cyan-300/80"}`}>
              Click the end of the {tool} · Esc to cancel
            </p>
          ) : (
            <p className={`text-[10px] font-mono leading-5 ${tool === "crosswalk" ? "text-sky-300/80" : "text-cyan-300/80"}`}>
              Click the start of the {tool} ({width.toFixed(1)} m wide)
            </p>
          )
        ) : selectedIds.size > 0 ? (
          <>
            <p className="text-[10px] text-cyan-300/80 font-mono leading-5">
              {selectedIds.size === 1 ? "1 lanelet selected" : `${selectedIds.size} lanelets selected`}
            </p>
            <p className="text-[10px] text-white/40 font-mono leading-5">Drag cyan dot · move endpoint</p>
            <p className="text-[10px] text-white/40 font-mono leading-5">Shift-click · add to selection</p>
            <p className="text-[10px] text-white/40 font-mono leading-5">Del · delete · Esc · deselect</p>
          </>
        ) : cameraMode === "3d" ? (
          <>
            <p className="text-[10px] text-white/20 font-mono leading-5">Left drag · rotate</p>
            <p className="text-[10px] text-white/20 font-mono leading-5">Right drag · pan</p>
            <p className="text-[10px] text-white/20 font-mono leading-5">Click a lanelet · select</p>
          </>
        ) : (
          <>
            <p className="text-[10px] text-white/20 font-mono leading-5">Drag · pan · Scroll · zoom</p>
            <p className="text-[10px] text-white/20 font-mono leading-5">Click a lanelet · select</p>
          </>
        )}
      </div>
    </div>
  );
}
