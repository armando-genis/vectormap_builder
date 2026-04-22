"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { RefObject } from "react";
import * as THREE from "three";
import type { ChunkOut, InMsg, OutMsg } from "./pointCloudWorker";

/**
 * Oriented, optionally tilted clip volume. When a `CropRegion` is passed
 * alongside `zCeiling`, the Y-discard is *scoped* to the rectangle's
 * XZ footprint: points outside always pass through, points inside pass
 * only if their Y ≤ the tilted clip plane's height at their local (X, Z).
 *
 * Why tilt as well as rotate? On a hillside street the ground isn't flat
 * in the world frame — it rises along one local axis. A horizontal
 * `zCeiling` then cuts diagonally through the roofs (peels off one side
 * only). Tilting the clip plane to match the slope makes the slider
 * peel off a consistent height above the road surface instead.
 *
 * Angles (`angle` / `pitch` / `roll`) are in radians. Applied in Y-X-Z
 * order (yaw first, then pitch around the rectangle's *local* X, then
 * roll around its *local* Z), matching Three.js's `Euler(x, y, z, "YXZ")`.
 *
 * Sign conventions, at the rectangle's center:
 *   - `angle` > 0 → rectangle's local X rotates toward world +Z.
 *   - `pitch` > 0 → clip plane rises as you move along local +Z
 *                   ("street uphill" along its length).
 *   - `roll`  > 0 → clip plane rises as you move along local +X
 *                   ("camber" toward one side).
 */
export interface CropRegion {
  /** World-space center on X. */
  cx: number;
  /** World-space center on Z. */
  cz: number;
  /** Half-extent along the rectangle's local X (= width / 2). */
  halfW: number;
  /** Half-extent along the rectangle's local Z (= length / 2). */
  halfL: number;
  /** Yaw of the rectangle around world Y (radians). */
  angle: number;
  /** Tilt around the rectangle's local X axis (radians). */
  pitch: number;
  /** Tilt around the rectangle's local Z axis (radians). */
  roll:  number;
}

/** Status surfaced to the parent (e.g. to show a status pill). */
export interface PointCloudStats {
  phase:       "idle" | "processing" | "ready" | "error";
  rawPoints:   number;   // points in the loaded PCD
  voxelPoints: number;   // points after voxel downsampling
  chunkCount:  number;   // how many spatial tiles we're rendering
  progress:    number | null; // 0..1 while `phase === "processing"`, else null
  message?:    string;
}

interface PointCloudProps {
  geometry: THREE.BufferGeometry;
  pointSize?: number;
  voxelSize?: number;   // 0 = disabled
  cameraMode?: "3d" | "2d";
  /**
   * Height ceiling in world Y.
   * Points whose Y is ABOVE this value are discarded in the fragment shader.
   * `null` / `Infinity` → no clipping.
   */
  zCeiling?: number | null;
  /**
   * Optional XZ rectangle restricting where `zCeiling` applies. When set,
   * the Y-discard only fires for points inside the rectangle; points
   * outside are always drawn. When omitted/null the Y-discard is global,
   * matching the original behaviour.
   */
  cropRegion?: CropRegion | null;
  /**
   * Called when the user clicks on a point in the cloud, with that point's
   * world-space position. Use to snap markers onto the real surface.
   */
  onPick?: (pos: [number, number, number]) => void;
  pickEnabled?: boolean;
  /**
   * Gives the parent access to the group containing all chunk <points>,
   * so surface-snap raycasts (`intersectObject(group, true)`) hit every
   * tile at once.
   */
  pointsRef?: RefObject<THREE.Group | null>;
  /** Approx points per spatial chunk. More chunks = better culling, more draw calls. */
  targetChunkPoints?: number;
  /** Notified on every status change (loading → ready, progress ticks, …). */
  onStatsChange?: (stats: PointCloudStats) => void;
}

// ---------------------------------------------------------------------------
// Main-thread <-> worker bridge.
//
// Chunking + voxel downsampling both used to run synchronously on the
// main thread, which for a ~100 M-point cloud froze the UI for 5-10 s.
// Everything heavy now runs in a module worker; the render tree only
// mounts chunk <points> once buffers land back here (zero-copy via
// transferables). The component survives voxel-slider drags by sending
// tiny `{ type: "voxel" }` deltas instead of re-transferring the cloud.
// ---------------------------------------------------------------------------

/** Debounce the voxel slider so a rapid drag doesn't thrash the worker. */
const VOXEL_DEBOUNCE_MS = 120;

/** Target points per spatial tile — tuned so mid-size clouds (~1 M pts)
 *  end up with a few dozen chunks. Frustum-cull and ray-picking both
 *  scale linearly with chunk count, so a handful-to-hundreds is ideal. */
const DEFAULT_TARGET_CHUNK_POINTS = 25_000;

/** Clone a Float32Array-compatible view into a fresh Float32Array so we
 *  can transfer it to the worker without detaching the original (the
 *  parent may still need `boundingBox` / `boundingSphere` on the source
 *  geometry). `BufferAttribute.array` is typed as `TypedArray` in three —
 *  the `ArrayLike<number>` constructor overload works for all variants. */
function cloneFloat32(src: ArrayLike<number>): Float32Array {
  return new Float32Array(src);
}

export function PointCloud({
  geometry,
  pointSize = 0.03,
  voxelSize = 0,
  cameraMode = "3d",
  zCeiling = null,
  cropRegion = null,
  onPick,
  pickEnabled = false,
  pointsRef,
  targetChunkPoints = DEFAULT_TARGET_CHUNK_POINTS,
  onStatsChange,
}: PointCloudProps) {
  // -----------------------------------------------------------------------
  // Uniforms (shared across every chunk's material). Stable object
  // identities so three.js compiles the patched shader once.
  // -----------------------------------------------------------------------
  const zCeilingUniform    = useMemo(() => ({ value: Number.POSITIVE_INFINITY }), []);
  const cropEnabledUniform = useMemo(() => ({ value: 0 }), []);
  const cropBoxUniform     = useMemo(() => ({ value: new THREE.Vector4(0, 0, 0, 0) }), []);
  const cropRotUniform     = useMemo(() => ({ value: new THREE.Vector2(1, 0) }), []);
  const cropTiltUniform    = useMemo(() => ({ value: new THREE.Vector2(0, 0) }), []);

  // In 2D mode sizeAttenuation=false → size is screen pixels.
  const is2D = cameraMode === "2d";

  // -----------------------------------------------------------------------
  // Single shared PointsMaterial re-used by every chunk. Reacting to
  // pointSize / cameraMode via mutable props so we don't thrash material
  // recompilation (same program, different uniforms/state).
  // -----------------------------------------------------------------------
  const material = useMemo(() => {
    const m = new THREE.PointsMaterial({
      vertexColors: true,
      transparent:  true,
      opacity:      0.9,
      depthWrite:   false,
    });

    m.onBeforeCompile = (shader) => {
      shader.uniforms.uZCeiling    = zCeilingUniform;
      shader.uniforms.uCropEnabled = cropEnabledUniform;
      shader.uniforms.uCropBox     = cropBoxUniform;
      shader.uniforms.uCropRot     = cropRotUniform;
      shader.uniforms.uCropTilt    = cropTiltUniform;

      shader.vertexShader =
        "varying vec3 vWorldPos;\n" +
        shader.vertexShader.replace(
          "#include <begin_vertex>",
          `#include <begin_vertex>
           vWorldPos = (modelMatrix * vec4(position, 1.0)).xyz;`
        );

      // Logic:
      //   - crop disabled       → Y-clip is global, horizontal
      //   - crop enabled + outside the rectangle → always keep
      //   - crop enabled + inside  the rectangle → tilted Y-clip applies
      //
      // Inside the rectangle, the clip height is a tilted plane:
      //   worldY > uZCeiling + tan(roll)·localX + tan(pitch)·localZ
      shader.fragmentShader =
        "varying vec3 vWorldPos;\n" +
        "uniform float uZCeiling;\n" +
        "uniform float uCropEnabled;\n" +
        "uniform vec4  uCropBox;\n" +
        "uniform vec2  uCropRot;\n" +
        "uniform vec2  uCropTilt;\n" +
        shader.fragmentShader.replace(
          "void main() {",
          `void main() {
             vec2 d = vec2(vWorldPos.x - uCropBox.x, vWorldPos.z - uCropBox.y);
             vec2 dl = vec2(
               d.x * uCropRot.x + d.y * uCropRot.y,
              -d.x * uCropRot.y + d.y * uCropRot.x
             );
             bool insideCrop =
               abs(dl.x) <= uCropBox.z &&
               abs(dl.y) <= uCropBox.w;
             bool applyClip = (uCropEnabled < 0.5) || insideCrop;
             float ceiling  = uZCeiling + uCropTilt.x * dl.x + uCropTilt.y * dl.y;
             if (applyClip && vWorldPos.y > ceiling) discard;`
        );
    };
    return m;
  }, [
    zCeilingUniform,
    cropEnabledUniform,
    cropBoxUniform,
    cropRotUniform,
    cropTiltUniform,
  ]);

  // Keep material state aligned with the live props without re-creating it.
  useEffect(() => { material.size = pointSize; }, [material, pointSize]);
  useEffect(() => { material.sizeAttenuation = !is2D; material.needsUpdate = true; }, [material, is2D]);

  // Release GPU resources when the component unmounts.
  useEffect(() => () => material.dispose(), [material]);

  // -----------------------------------------------------------------------
  // Uniform sync effects (unchanged behavior from the single-Points
  // version).
  // -----------------------------------------------------------------------
  useEffect(() => {
    zCeilingUniform.value =
      zCeiling == null || !Number.isFinite(zCeiling)
        ? Number.POSITIVE_INFINITY
        : zCeiling;
  }, [zCeiling, zCeilingUniform]);

  useEffect(() => {
    if (cropRegion) {
      cropEnabledUniform.value = 1;
      cropBoxUniform.value.set(
        cropRegion.cx,
        cropRegion.cz,
        Math.max(cropRegion.halfW, 1e-3),
        Math.max(cropRegion.halfL, 1e-3)
      );
      cropRotUniform.value.set(
        Math.cos(cropRegion.angle),
        Math.sin(cropRegion.angle)
      );
      const MAX_TILT = (85 * Math.PI) / 180;
      const pitch = Math.max(-MAX_TILT, Math.min(MAX_TILT, cropRegion.pitch));
      const roll  = Math.max(-MAX_TILT, Math.min(MAX_TILT, cropRegion.roll));
      cropTiltUniform.value.set(Math.tan(roll), Math.tan(pitch));
    } else {
      cropEnabledUniform.value = 0;
    }
  }, [
    cropRegion,
    cropEnabledUniform,
    cropBoxUniform,
    cropRotUniform,
    cropTiltUniform,
  ]);

  // -----------------------------------------------------------------------
  // Worker lifecycle. Lives for the lifetime of the component. Re-ingests
  // the cloud when `geometry` changes (= user loaded a new PCD);
  // otherwise just swaps voxel params.
  // -----------------------------------------------------------------------
  const workerRef = useRef<Worker | null>(null);

  // Chunk geometries currently mounted. Stored as plain objects so we can
  // dispose them deterministically in cleanup; never mutated in place.
  const [chunkGeos, setChunkGeos] = useState<
    Array<{ key: string; geo: THREE.BufferGeometry }>
  >([]);
  const chunkGeosRef = useRef(chunkGeos);
  useEffect(() => { chunkGeosRef.current = chunkGeos; }, [chunkGeos]);

  // Latest onStatsChange in a ref so the worker handler doesn't need to
  // be re-bound on every parent re-render (would cancel pending voxel
  // debounce timers for no reason).
  const onStatsRef = useRef(onStatsChange);
  useEffect(() => { onStatsRef.current = onStatsChange; }, [onStatsChange]);

  const emitStats = (stats: PointCloudStats) => {
    onStatsRef.current?.(stats);
  };

  // Tracks what's been ingested; survives voxel/chunk re-runs.
  const ingestedRef = useRef<{
    rawPoints:   number;
    voxelPoints: number;
    chunkCount:  number;
  }>({ rawPoints: 0, voxelPoints: 0, chunkCount: 0 });

  // -----------------------------------------------------------------------
  // Spawn / teardown the worker.
  // -----------------------------------------------------------------------
  useEffect(() => {
    const w = new Worker(
      new URL("./pointCloudWorker.ts", import.meta.url),
      { type: "module" }
    );
    workerRef.current = w;

    w.onmessage = (e: MessageEvent<OutMsg>) => {
      const msg = e.data;
      switch (msg.type) {
        case "progress": {
          const total = msg.total || 1;
          emitStats({
            phase:       "processing",
            rawPoints:   ingestedRef.current.rawPoints,
            voxelPoints: ingestedRef.current.voxelPoints,
            chunkCount:  ingestedRef.current.chunkCount,
            progress:    Math.min(1, msg.done / total),
            message:     msg.phase === "voxel" ? "Voxel downsampling…" : "Chunking cloud…",
          });
          break;
        }
        case "chunks": {
          // Tear down any previously-mounted chunk geometries.
          for (const g of chunkGeosRef.current) g.geo.dispose();

          // Build BufferGeometries for the new chunks. We own the Float32Arrays
          // transferred from the worker, so no copy needed.
          const next = msg.chunks.map<{ key: string; geo: THREE.BufferGeometry }>(
            (c: ChunkOut, i: number) => {
              const g = new THREE.BufferGeometry();
              g.setAttribute("position", new THREE.BufferAttribute(c.positions, 3));
              g.setAttribute("color",    new THREE.BufferAttribute(c.colors, 3));
              // Set bbox/sphere from the worker's per-chunk min/max so
              // frustum culling + raycast early-out work without a
              // full-buffer scan on the main thread.
              const min = new THREE.Vector3(...c.min);
              const max = new THREE.Vector3(...c.max);
              g.boundingBox    = new THREE.Box3(min, max);
              g.boundingSphere = new THREE.Sphere();
              g.boundingBox.getBoundingSphere(g.boundingSphere);
              return { key: `${c.cellKey}-${i}`, geo: g };
            }
          );
          setChunkGeos(next);

          ingestedRef.current = {
            rawPoints:   msg.rawPoints,
            voxelPoints: msg.voxelPoints,
            chunkCount:  next.length,
          };

          emitStats({
            phase:       "ready",
            rawPoints:   msg.rawPoints,
            voxelPoints: msg.voxelPoints,
            chunkCount:  next.length,
            progress:    null,
          });
          break;
        }
        case "error": {
          emitStats({
            phase:       "error",
            rawPoints:   ingestedRef.current.rawPoints,
            voxelPoints: ingestedRef.current.voxelPoints,
            chunkCount:  ingestedRef.current.chunkCount,
            progress:    null,
            message:     msg.message,
          });
          break;
        }
      }
    };

    return () => {
      w.postMessage({ type: "clear" } as InMsg);
      w.terminate();
      workerRef.current = null;

      // Dispose any geometries currently mounted. They own buffers we
      // transferred from the worker — disposing releases GPU memory and
      // lets the JS runtime free the typed arrays.
      for (const g of chunkGeosRef.current) g.geo.dispose();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // -----------------------------------------------------------------------
  // Ingest: whenever the source geometry changes, ship its buffers to
  // the worker. We CLONE them first so the source keeps a usable copy
  // (bbox/sphere calculations, future re-ingests, …).
  // -----------------------------------------------------------------------
  useEffect(() => {
    const w = workerRef.current;
    if (!w || !geometry) return;
    const posAttr = geometry.attributes.position;
    if (!posAttr) return;

    const positions = cloneFloat32(posAttr.array as ArrayLike<number>);
    const colAttr   = geometry.attributes.color;
    const colors    = colAttr
      ? cloneFloat32(colAttr.array as ArrayLike<number>)
      : undefined;

    // `.buffer` is typed as ArrayBufferLike (could be SharedArrayBuffer)
    // but we just allocated these Float32Arrays ourselves so they're
    // plain ArrayBuffers — safe to cast for the transferable list.
    const transfer: ArrayBuffer[] = [positions.buffer as ArrayBuffer];
    if (colors) transfer.push(colors.buffer as ArrayBuffer);

    emitStats({
      phase:       "processing",
      rawPoints:   positions.length / 3,
      voxelPoints: 0,
      chunkCount:  0,
      progress:    0,
      message:     "Handing off to worker…",
    });

    const msg: InMsg = {
      type:              "ingest",
      positions,
      colors,
      voxelSize,
      targetChunkPoints,
    };
    w.postMessage(msg, transfer);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [geometry]);

  // -----------------------------------------------------------------------
  // Voxel slider: debounced re-run. Uses the worker's cached raw buffers
  // so we don't re-transfer the cloud — just (voxelSize, chunkTarget).
  //
  // First render after `geometry` changes still goes through the `ingest`
  // effect above (which already carries `voxelSize`), so this guard skips
  // the duplicate kick-off on the initial mount.
  // -----------------------------------------------------------------------
  const skipFirstVoxel = useRef(true);
  useEffect(() => {
    if (skipFirstVoxel.current) {
      skipFirstVoxel.current = false;
      return;
    }
    const w = workerRef.current;
    if (!w) return;

    const handle = window.setTimeout(() => {
      emitStats({
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
        targetChunkPoints,
      } as InMsg);
    }, VOXEL_DEBOUNCE_MS);

    return () => window.clearTimeout(handle);
  }, [voxelSize, targetChunkPoints]);

  // Re-ingest should reset the "skip first voxel" flag so later voxel
  // changes still debounce properly.
  useEffect(() => {
    skipFirstVoxel.current = true;
  }, [geometry]);

  // -----------------------------------------------------------------------
  // Picking — forwarded from whichever chunk the raycaster hit. R3F's
  // onClick bubbles up to the group, so one handler covers every tile.
  // -----------------------------------------------------------------------
  const handleGroupClick = pickEnabled && onPick
    ? (e: { stopPropagation: () => void; point: THREE.Vector3 }) => {
        e.stopPropagation();
        onPick([e.point.x, e.point.y, e.point.z]);
      }
    : undefined;

  return (
    <group ref={pointsRef} onClick={handleGroupClick}>
      {chunkGeos.map(({ key, geo }) => (
        <points
          key={key}
          geometry={geo}
          material={material}
          // Explicit so three.js doesn't recompute a bbox on every mount;
          // worker already set a tight one on the geometry.
          frustumCulled
        />
      ))}
    </group>
  );
}
