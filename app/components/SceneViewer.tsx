"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls, Grid, Stats } from "@react-three/drei";
import * as THREE from "three";
import { PointCloud } from "./PointCloud";
import { LaneletLayer } from "./lanelet/LaneletLayer";
import type { DragHandleState } from "./lanelet/LaneletLayer";
import { LaneletProperties } from "./lanelet/LaneletProperties";
import {
  applyLaneletTranslationFromSnapshot,
  createLanelet,
  duplicateLaneletLeft,
  duplicateLaneletRight,
  joinLanelets,
  moveLaneletNodeAtIndex,
  resizeLaneletWidth,
  reverseLanelet,
  sampleSurfaceY,
  snapLaneletToSurface,
  ATTACH_DISTANCE,
} from "./lanelet/geometry";
import type { JointType } from "./lanelet/geometry";
import {
  createRegistry,
  findNearestLaneletEnd,
  gcUnused,
  junctionNodeIds,
  resolveAll,
  getNodeOrNull,
} from "./lanelet/registry";
import type { LaneletEndSnap } from "./lanelet/registry";
import type {
  Lanelet,
  NodeId,
  NodeRegistry,
  ResolvedLanelet,
  Vec3,
} from "./lanelet/types";

type CameraMode = "3d" | "2d";
// "view"      = selection/edit.
// "lanelet"   = draw a road lanelet.
// "crosswalk" = draw a crosswalk (no direction arrow, zebra-striped fill).
type Tool       = "view" | "lanelet" | "crosswalk";

interface SceneProps {
  geometry: THREE.BufferGeometry | null;
  pointSize: number;
  voxelSize: number;
  zCeiling: number;
  cameraMode: CameraMode;
  tool: Tool;
  width: number;

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

  nextLaneletIdRef: React.MutableRefObject<number>;
}

function Scene({
  geometry,
  pointSize,
  voxelSize,
  zCeiling,
  cameraMode,
  tool,
  width,
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
  nextLaneletIdRef,
}: SceneProps) {
  const { camera, raycaster, gl } = useThree();
  const controlsRef = useRef<any>(null);
  const pointsRef   = useRef<THREE.Points | null>(null);

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
    if (controlsRef.current) controlsRef.current.enabled = false;
    setDragHandle({ kind: "node", id, index });
  };

  const handleMoveHandlePointerDown = (id: number) => {
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
      const bbox = geometry?.boundingBox;
      const topY = (bbox?.max.y ?? 1e4) + 10;

      const curr = laneletsRef.current.find((l) => l.id === dragHandle.id);
      if (curr) {
        surfaceRc.params.Points = {
          ...(surfaceRc.params.Points ?? {}),
          threshold: Math.max(0.1, Math.min(curr.width / 6, 0.8)),
        };
        const snapY = (x: number, z: number, fb: number) =>
          sampleSurfaceY(surfaceRc, pointsRef.current, x, z, topY, fb);

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
    if (!pendingStart) {
      const snap = findNearestLaneletEnd(
        registryRef.current,
        laneletsRef.current,
        p,
        endSnapThreshold
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
      endSnapThreshold
    );

    const bbox = geometry?.boundingBox;
    const topY = (bbox?.max.y ?? 1e4) + 10;

    surfaceRc.params.Points = {
      ...(surfaceRc.params.Points ?? {}),
      threshold: Math.max(0.1, Math.min(width / 6, 0.8)),
    };

    const snapY = (x: number, z: number, fallback: number) =>
      sampleSurfaceY(surfaceRc, pointsRef.current, x, z, topY, fallback);

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
      // Tool chooses the kind of lanelet being drawn.
      subType: tool === "crosswalk" ? "crosswalk" : "road",
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
          geometry={geometry}
          pointSize={pointSize}
          voxelSize={voxelSize}
          cameraMode={cameraMode}
          zCeiling={zCeiling}
          pickEnabled={tool === "lanelet" || tool === "crosswalk"}
          onPick={handlePick}
          pointsRef={pointsRef}
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
        pendingStart={pendingStart}
        pendingStartAttached={pendingStartSnap !== null}
        sceneRadius={r}
        junctionPositions={junctionPositions}
      />
    </>
  );
}

interface SceneViewerProps {
  geometry: THREE.BufferGeometry | null;
  fileName: string;
  pointCount: number;
  onReset: () => void;
}

export function SceneViewer({ geometry, fileName, pointCount, onReset }: SceneViewerProps) {
  const [cameraMode,    setCameraMode]    = useState<CameraMode>("3d");
  const [pointSize3d,   setPointSize3d]   = useState(0.03);
  const [pointSize2d,   setPointSize2d]   = useState(2);
  const [voxelSize,     setVoxelSize]     = useState(0);
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

  const pointSize = cameraMode === "3d" ? pointSize3d : pointSize2d;

  const bbox  = geometry?.boundingBox ?? null;
  const yMin  = bbox ? bbox.min.y : 0;
  const yMax  = bbox ? bbox.max.y : 1;
  const ySpan = Math.max(yMax - yMin, 1e-6);
  const yStep = ySpan / 200;
  const [zCeiling, setZCeiling] = useState<number>(yMax);

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
        if (pendingStart) {
          setPendingStart(null);
          setPendingStartSnap(null);
        }
        else if (selectedIds.size > 0) deselectAll();
      } else if (e.key === "Delete" || e.key === "Backspace") {
        if (selectedIds.size > 0) {
          e.preventDefault();
          deleteIds(Array.from(selectedIds));
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingStart, selectedIds]);

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
          pointSize={pointSize}
          voxelSize={voxelSize}
          zCeiling={zCeiling}
          cameraMode={cameraMode}
          tool={tool}
          width={width}
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
          nextLaneletIdRef={nextLaneletIdRef}
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
            <div className="rounded-md border border-white/10 bg-white/5 px-3 py-1.5 backdrop-blur-sm">
              <span className="text-xs text-white/50 font-mono">{pointCount.toLocaleString()} pts</span>
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
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono text-white/40 uppercase tracking-wider">Width</span>
            <span className="text-xs font-mono text-white/70">{width.toFixed(1)} m</span>
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
            onChange={(e) => setVoxelSize(parseFloat(e.target.value))}
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
          <span className={`text-xs font-mono w-14 ${zCeiling < yMax ? "text-cyan-400" : "text-white/30"}`}>
            {bbox ? `${zCeiling.toFixed(2)} m` : "—"}
          </span>
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
        {tool !== "view" ? (
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
