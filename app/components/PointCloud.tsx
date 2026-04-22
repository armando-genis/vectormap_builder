"use client";

import { useEffect, useMemo } from "react";
import type { RefObject } from "react";
import * as THREE from "three";
import type { PointCloudChunk } from "./types";

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

// `PointCloudChunk` lives in `./types` so `PointCloudViewer` can construct
// chunks without also dragging in the renderer (and its whole Three.js
// material closure). This component is intentionally the "pure renderer"
// side of that split — no worker, no parsing, no downsampling.

interface PointCloudProps {
  /** Rendered chunks. Pass an empty array while the worker is still busy. */
  chunks: PointCloudChunk[];
  pointSize?: number;
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
}

export function PointCloud({
  chunks,
  pointSize = 0.03,
  cameraMode = "3d",
  zCeiling = null,
  cropRegion = null,
  onPick,
  pickEnabled = false,
  pointsRef,
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
      {chunks.map(({ key, geo }) => (
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
