"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { FileDropZone } from "./FileDropZone";
import { SceneViewer } from "./SceneViewer";
import type { PointCloudChunk, PointCloudStats } from "./types";
import type { ChunkOut, InMsg, OutMsg } from "./pointCloudWorker";

type ViewerState = "idle" | "loading" | "loaded" | "error";

/** Debounce the voxel slider so a rapid drag doesn't thrash the worker. */
const VOXEL_DEBOUNCE_MS = 120;

/** Target points per spatial tile. Tuned so mid-size clouds (~1 M pts)
 *  end up with a few dozen chunks. Frustum-cull and ray-picking both
 *  scale linearly with chunk count, so a handful-to-hundreds is ideal. */
const DEFAULT_TARGET_CHUNK_POINTS = 25_000;

/**
 * Pick a sensible starting voxel size for a freshly dropped file.
 *
 * For multi-GB clouds the unvoxelised pipeline OOMs V8 (positions
 * alone is ~2 GB, plus colors, plus per-chunk buffers — the allocator
 * gives up with "Array buffer allocation failed"). Plus, even if
 * memory weren't a concern, no GPU can usefully render 100 M+ points.
 *
 * Thresholds are coarse: they only trigger when the user left the
 * slider at 0. Anyone who already dialed in a voxel size keeps it.
 */
function autoVoxelFor(fileSize: number, currentVoxel: number): number {
  if (currentVoxel > 0) return currentVoxel;
  if (fileSize > 2_000_000_000) return 0.2;   // >2 GB → ~25 cm cells
  if (fileSize > 1_000_000_000) return 0.15;  // 1–2 GB
  if (fileSize >   500_000_000) return 0.08;  // 500 MB – 1 GB
  if (fileSize >   200_000_000) return 0.04;  // 200–500 MB
  return 0;
}

// Human-readable byte size for the loading overlay.
function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

/**
 * Build a placeholder `BufferGeometry` that's empty of points but
 * carries the aggregate bbox/sphere. SceneViewer only reads those two
 * fields off `geometry`, never the position attribute, so this is
 * enough to drive its camera-init / Z-clip / crop-center logic without
 * copying the full cloud to the main thread.
 */
function bboxGeometry(min: THREE.Vector3, max: THREE.Vector3): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry();
  // Zero-length position attribute: `geometry.attributes.position.count`
  // is 0, but `boundingBox` / `boundingSphere` are set explicitly below.
  g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(0), 3));
  g.boundingBox    = new THREE.Box3(min.clone(), max.clone());
  g.boundingSphere = new THREE.Sphere();
  g.boundingBox.getBoundingSphere(g.boundingSphere);
  return g;
}

/** Turn the worker's transferable chunks into GPU-ready BufferGeometries. */
function chunkOutToRenderable(c: ChunkOut, i: number): PointCloudChunk {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(c.positions, 3));
  g.setAttribute("color",    new THREE.BufferAttribute(c.colors,    3));
  const min = new THREE.Vector3(...c.min);
  const max = new THREE.Vector3(...c.max);
  g.boundingBox    = new THREE.Box3(min, max);
  g.boundingSphere = new THREE.Sphere();
  g.boundingBox.getBoundingSphere(g.boundingSphere);
  return { key: `${c.cellKey}-${i}`, geo: g };
}

export default function PointCloudViewer() {
  const [state, setState] = useState<ViewerState>("idle");
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null);
  const [chunks, setChunks] = useState<PointCloudChunk[]>([]);
  const [fileName, setFileName] = useState("");
  const [fileSize, setFileSize] = useState(0);
  const [error, setError] = useState("");
  const [voxelSize, setVoxelSize] = useState(0);
  const [cloudStats, setCloudStats] = useState<PointCloudStats | null>(null);

  // --------------------------------------------------------------------
  // Worker lives at this level, not inside PointCloud. The critical bit:
  // for files above ~2 GB, Chrome's main-thread `file.arrayBuffer()`
  // rejects with `NotReadableError`. Passing the File into the worker
  // via postMessage (File is structured-cloneable — the bytes aren't
  // copied, just a reference) sidesteps that cap entirely; the worker
  // reads the file itself.
  // --------------------------------------------------------------------
  const workerRef     = useRef<Worker | null>(null);
  const chunksRef     = useRef<PointCloudChunk[]>([]);
  const geometryRef   = useRef<THREE.BufferGeometry | null>(null);
  const ingestedRef   = useRef({ rawPoints: 0, voxelPoints: 0, chunkCount: 0 });

  useEffect(() => { chunksRef.current   = chunks;   }, [chunks]);
  useEffect(() => { geometryRef.current = geometry; }, [geometry]);

  const ensureWorker = useCallback(() => {
    if (workerRef.current) return workerRef.current;

    const w = new Worker(
      new URL("./pointCloudWorker.ts", import.meta.url),
      { type: "module" }
    );

    w.onmessage = (e: MessageEvent<OutMsg>) => {
      const msg = e.data;
      switch (msg.type) {
        case "progress": {
          const total = msg.total || 1;
          const defaultLabel =
            msg.phase === "reading"      ? "Reading file…"
          : msg.phase === "parsing"      ? "Parsing PCD…"
          : msg.phase === "transforming" ? "Transforming to Y-up…"
          : msg.phase === "voxel"        ? "Voxel downsampling…"
                                         : "Chunking cloud…";
          setCloudStats({
            phase:       "processing",
            rawPoints:   ingestedRef.current.rawPoints,
            voxelPoints: ingestedRef.current.voxelPoints,
            chunkCount:  ingestedRef.current.chunkCount,
            progress:    Math.min(1, msg.done / total),
            message:     msg.message ?? defaultLabel,
          });
          break;
        }
        case "chunks": {
          // Dispose previous chunks before swapping — React will remount
          // <points> elements and we don't want stale GPU buffers.
          for (const c of chunksRef.current) c.geo.dispose();

          const next = msg.chunks.map(chunkOutToRenderable);
          setChunks(next);

          // Build / refresh the placeholder geometry so SceneViewer's
          // camera init + crop defaults pick up the correct bbox. Voxel
          // re-runs produce a very slightly different bbox (rounding to
          // grid centers), so we always rebuild it here.
          const prevGeo = geometryRef.current;
          const newGeo = bboxGeometry(
            new THREE.Vector3(...msg.min),
            new THREE.Vector3(...msg.max),
          );
          // If we already had a geometry *identity* (not first ingest),
          // keep it stable so SceneViewer's `[geometry]` effects don't
          // nuke lanelets and selection every time the voxel slider
          // moves. Copy the bbox over instead.
          if (prevGeo) {
            prevGeo.boundingBox    = newGeo.boundingBox;
            prevGeo.boundingSphere = newGeo.boundingSphere;
            newGeo.dispose();
          } else {
            setGeometry(newGeo);
          }

          ingestedRef.current = {
            rawPoints:   msg.rawPoints,
            voxelPoints: msg.voxelPoints,
            chunkCount:  next.length,
          };
          setCloudStats({
            phase:       "ready",
            rawPoints:   msg.rawPoints,
            voxelPoints: msg.voxelPoints,
            chunkCount:  next.length,
            progress:    null,
          });
          setState("loaded");
          break;
        }
        case "error": {
          setCloudStats({
            phase:       "error",
            rawPoints:   ingestedRef.current.rawPoints,
            voxelPoints: ingestedRef.current.voxelPoints,
            chunkCount:  ingestedRef.current.chunkCount,
            progress:    null,
            message:     msg.message,
          });
          // Only flip the app to the error screen if we hadn't already
          // shown the scene. A voxel re-run that fails shouldn't lose
          // the user's work — the error goes to the stats pill only.
          if (!geometryRef.current) {
            setError(msg.message);
            setState("error");
          }
          break;
        }
      }
    };

    workerRef.current = w;
    return w;
  }, []);

  // Terminate on unmount (page navigate, HMR). The scene-reset path
  // reuses the same worker.
  useEffect(() => {
    return () => {
      workerRef.current?.postMessage({ type: "clear" } as InMsg);
      workerRef.current?.terminate();
      workerRef.current = null;
    };
  }, []);

  const handleFile = useCallback(async (file: File) => {
    setState("loading");
    setError("");
    setFileName(file.name);
    setFileSize(file.size);
    setCloudStats({
      phase:       "processing",
      rawPoints:   0,
      voxelPoints: 0,
      chunkCount:  0,
      progress:    0,
      message:     "Starting…",
    });

    // Drop the previous cloud before kicking off a new parse — prevents
    // double memory pressure while the worker loads a fresh file.
    for (const c of chunksRef.current) c.geo.dispose();
    setChunks([]);
    if (geometryRef.current) {
      geometryRef.current.dispose();
      setGeometry(null);
    }
    ingestedRef.current = { rawPoints: 0, voxelPoints: 0, chunkCount: 0 };

    try {
      // For a multi-GB cloud, voxelSize=0 would force the worker to
      // ingest hundreds of millions of points into GPU memory and
      // allocate several 1-2 GB intermediate arrays during chunking —
      // both the allocator and the renderer would choke. Start the
      // slider at a sensible default based on file size; the user can
      // still dial it down (slowly — the worker re-voxels on every
      // debounced change).
      const initialVoxel = autoVoxelFor(file.size, voxelSize);
      if (initialVoxel !== voxelSize) setVoxelSize(initialVoxel);

      const w = ensureWorker();
      // Hand the File off to the worker. File is structured-cloneable,
      // so this postMessage is essentially free (no byte copy) — and,
      // crucially, the worker's own arrayBuffer()/stream() calls aren't
      // subject to the main-thread NotReadableError cap.
      const msg: InMsg = {
        type:              "parseFile",
        file,
        voxelSize:         initialVoxel,
        targetChunkPoints: DEFAULT_TARGET_CHUNK_POINTS,
      };
      w.postMessage(msg);
    } catch (err) {
      console.error("PCD load failed:", err);
      const fallback =
        "Failed to load PCD file. Make sure it is a valid .pcd format.";
      const m = err instanceof Error && err.message ? err.message : fallback;
      setError(m);
      setState("error");
    }
  }, [voxelSize, ensureWorker]);

  const handleReset = useCallback(() => {
    workerRef.current?.postMessage({ type: "clear" } as InMsg);
    for (const c of chunksRef.current) c.geo.dispose();
    setChunks([]);
    if (geometryRef.current) {
      geometryRef.current.dispose();
    }
    setGeometry(null);
    setFileName("");
    setFileSize(0);
    setError("");
    setCloudStats(null);
    ingestedRef.current = { rawPoints: 0, voxelPoints: 0, chunkCount: 0 };
    setState("idle");
  }, []);

  // Voxel slider: debounced re-run. Uses the worker's cached raw buffers
  // so we don't re-transfer the cloud — just (voxelSize, chunkTarget).
  const firstVoxelRef = useRef(true);
  useEffect(() => {
    if (firstVoxelRef.current) {
      firstVoxelRef.current = false;
      return;
    }
    const w = workerRef.current;
    if (!w || !geometryRef.current) return;

    const handle = window.setTimeout(() => {
      setCloudStats({
        phase:       "processing",
        rawPoints:   ingestedRef.current.rawPoints,
        voxelPoints: ingestedRef.current.voxelPoints,
        chunkCount:  ingestedRef.current.chunkCount,
        progress:    0,
        message:     "Re-voxeling…",
      });
      w.postMessage({
        type:              "voxel",
        voxelSize,
        targetChunkPoints: DEFAULT_TARGET_CHUNK_POINTS,
      } as InMsg);
    }, VOXEL_DEBOUNCE_MS);

    return () => window.clearTimeout(handle);
  }, [voxelSize]);

  // Reset the "first voxel" flag when a brand-new file is loaded so the
  // slider stays debounced afterwards (and the initial parseFile run
  // already carries voxelSize — no duplicate kickoff needed).
  useEffect(() => {
    if (state === "loading") firstVoxelRef.current = true;
  }, [state]);

  const loadingLabel =
    cloudStats?.message ??
    (state === "loading" ? "Starting…" : "");

  return (
    <div className="relative w-full h-full bg-[#050810]">
      {/* Loading spinner overlay */}
      {state === "loading" && !geometry && (
        <div className="absolute inset-0 flex flex-col items-center justify-center z-20 bg-[#050810]">
          <div className="relative flex h-16 w-16 items-center justify-center">
            <div className="absolute inset-0 rounded-full border-2 border-cyan-500/20" />
            <div className="absolute inset-0 rounded-full border-2 border-t-cyan-400 border-r-transparent border-b-transparent border-l-transparent animate-spin" />
            <div className="h-2 w-2 rounded-full bg-cyan-400 shadow-[0_0_12px_rgba(0,220,255,0.9)]" />
          </div>
          <p className="mt-5 text-sm font-mono text-cyan-400/80 tracking-wider">
            {loadingLabel}
          </p>
          {cloudStats?.progress != null && cloudStats.progress > 0 && (
            <p className="mt-1 text-[11px] font-mono text-white/50 tabular-nums">
              {Math.round(cloudStats.progress * 100)}%
            </p>
          )}
          <p className="mt-1.5 text-xs font-mono text-white/25 max-w-[260px] text-center truncate">
            {fileName}
            {fileSize > 0 ? ` · ${formatBytes(fileSize)}` : ""}
          </p>
          {fileSize > 500 * 1024 * 1024 && (
            <p className="mt-3 text-[10px] font-mono text-white/35 max-w-[260px] text-center leading-relaxed">
              Large cloud — parsing runs in a Web Worker so the UI
              stays responsive. This can take up to a minute for
              &gt;2 GB files.
            </p>
          )}
        </div>
      )}

      {/* Drop zone when idle or errored */}
      {(state === "idle" || state === "error") && (
        <FileDropZone
          onFile={handleFile}
          error={state === "error" ? error : undefined}
        />
      )}

      {/* 3D scene when loaded */}
      {state === "loaded" && geometry && (
        <SceneViewer
          geometry={geometry}
          chunks={chunks}
          fileName={fileName}
          voxelSize={voxelSize}
          onVoxelSizeChange={setVoxelSize}
          cloudStats={cloudStats}
          onReset={handleReset}
        />
      )}
    </div>
  );
}
