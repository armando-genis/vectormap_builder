"use client";

import { useEffect, useRef, useState } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls, Grid, Stats } from "@react-three/drei";
import * as THREE from "three";
import { PointCloud } from "./PointCloud";
import { PointMarkers, type Marker } from "./PointMarkers";

type CameraMode = "3d" | "2d";
type Tool       = "view" | "place";

interface SceneProps {
  geometry: THREE.BufferGeometry | null;
  pointSize: number;
  voxelSize: number;
  zCeiling: number;
  cameraMode: CameraMode;
  tool: Tool;
  markers: Marker[];
  onPick: (pos: Marker) => void;
}

function Scene({
  geometry,
  pointSize,
  voxelSize,
  zCeiling,
  cameraMode,
  tool,
  markers,
  onPick,
}: SceneProps) {
  const { camera, raycaster } = useThree();
  const controlsRef = useRef<any>(null);

  // Tune the points raycast so clicks reliably hit a nearby cloud point.
  // Threshold is in world units; scale it with the scene so big maps still pick.
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
          pickEnabled={tool === "place"}
          onPick={onPick}
        />
      )}

      <PointMarkers markers={markers} />
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
  const [cameraMode,  setCameraMode]  = useState<CameraMode>("3d");
  const [pointSize3d, setPointSize3d] = useState(0.03);
  const [pointSize2d, setPointSize2d] = useState(2);
  const [voxelSize,   setVoxelSize]   = useState(0);
  const [showStats,   setShowStats]   = useState(false);
  const [tool,        setTool]        = useState<Tool>("view");
  const [markers,     setMarkers]     = useState<Marker[]>([]);

  const pointSize = cameraMode === "3d" ? pointSize3d : pointSize2d;

  // Z-clip (height ceiling)
  const bbox  = geometry?.boundingBox ?? null;
  const yMin  = bbox ? bbox.min.y : 0;
  const yMax  = bbox ? bbox.max.y : 1;
  const ySpan = Math.max(yMax - yMin, 1e-6);
  const yStep = ySpan / 200;
  const [zCeiling, setZCeiling] = useState<number>(yMax);

  // Whenever the cloud changes: reset markers and show-all Z-clip.
  useEffect(() => {
    if (geometry?.boundingBox) {
      setZCeiling(geometry.boundingBox.max.y);
    }
    setMarkers([]);
    setTool("view");
  }, [geometry]);

  const handlePick = (pos: Marker) => {
    setMarkers((m) => [...m, pos]);
  };

  return (
    <div className="relative w-full h-full">
      <Canvas
        camera={{ fov: 60, near: 0.01, far: 100000, position: [50, 30, 50] }}
        gl={{ antialias: true, alpha: false }}
        dpr={[1, 1.5]}
      >
        <Scene
          geometry={geometry}
          pointSize={pointSize}
          voxelSize={voxelSize}
          zCeiling={zCeiling}
          cameraMode={cameraMode}
          tool={tool}
          markers={markers}
          onPick={handlePick}
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

          {/* Camera mode toggle */}
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

      {/* ── Tool panel (top-left under header) ──────────────── */}
      <div className="absolute top-20 left-5 flex flex-col gap-2 rounded-xl border border-white/10 bg-black/60 p-3 backdrop-blur-md w-52">
        <div className="text-[10px] font-mono text-white/40 uppercase tracking-wider">Tool</div>

        <button
          onClick={() => setTool(tool === "place" ? "view" : "place")}
          className={`flex items-center justify-between px-3 py-2 rounded-md text-xs font-mono transition-colors cursor-pointer border
            ${tool === "place"
              ? "bg-cyan-500/20 text-cyan-300 border-cyan-400/40"
              : "bg-white/5 text-white/60 border-white/10 hover:text-white/80 hover:bg-white/10"}`}
        >
          <span className="flex items-center gap-2">
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M15 10.5a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M19.5 10.5c0 7.142-7.5 11.25-7.5 11.25S4.5 17.642 4.5 10.5a7.5 7.5 0 1 1 15 0Z" />
            </svg>
            Place point
          </span>
          <span className="text-[10px] text-white/40">{tool === "place" ? "ON" : "OFF"}</span>
        </button>

        <div className="flex items-center justify-between pt-1">
          <span className="text-[10px] font-mono text-white/40 uppercase">Markers</span>
          <span className="text-xs font-mono text-white/70">{markers.length}</span>
        </div>

        <button
          onClick={() => setMarkers([])}
          disabled={markers.length === 0}
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

        {/* Z-clip slider: drag LEFT to peel points off the top. */}
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

      {/* ── Navigation hint ─────────────────────────────────── */}
      <div className="absolute bottom-5 right-5 text-right pointer-events-none">
        {tool === "place" ? (
          <p className="text-[10px] text-cyan-300/80 font-mono leading-5">Click a point on the cloud to drop a marker</p>
        ) : cameraMode === "3d" ? (
          <>
            <p className="text-[10px] text-white/20 font-mono leading-5">Left drag · rotate</p>
            <p className="text-[10px] text-white/20 font-mono leading-5">Right drag · pan</p>
            <p className="text-[10px] text-white/20 font-mono leading-5">Scroll · zoom</p>
          </>
        ) : (
          <>
            <p className="text-[10px] text-white/20 font-mono leading-5">Drag · pan</p>
            <p className="text-[10px] text-white/20 font-mono leading-5">Scroll · zoom</p>
          </>
        )}
      </div>
    </div>
  );
}
