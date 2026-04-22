"use client";

import { useEffect, useMemo } from "react";
import type { RefObject } from "react";
import * as THREE from "three";

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
   * Called when the user clicks on a point in the cloud, with that point's
   * world-space position. Use to snap markers onto the real surface.
   */
  onPick?: (pos: [number, number, number]) => void;
  pickEnabled?: boolean;
  /** Gives the parent access to the <points> object for extra raycasts. */
  pointsRef?: RefObject<THREE.Points | null>;
}

// ---------------------------------------------------------------------------
// Voxel downsampling with a fixed-size bit-array hash set.
//
// Why not Map/Set with string keys?
//   V8 caps Map/Set at ~16.7 M entries. Large point clouds (50 M+ points)
//   at small voxel sizes produce more unique voxels than that limit.
//
// Solution: a 2 MB Uint8Array used as a bit-set (16 M slots).
//   - O(n) time,  O(1) extra memory regardless of cloud size.
//   - Hash collisions are possible (≈ false positives) but the output is
//     always valid geometry — it just keeps slightly more points near
//     hash-collision boundaries.
// ---------------------------------------------------------------------------
const BITSET_BITS  = 24;                  // 2^24 = 16 777 216 slots
const BITSET_SLOTS = 1 << BITSET_BITS;   // 16 M
const BITSET_BYTES = BITSET_SLOTS >> 3;  // 2 MB

function hashVoxel(vx: number, vy: number, vz: number): number {
  // Mix three integers into one unsigned 32-bit hash.
  // Based on a fast integer hash that distributes well in practice.
  let h = Math.imul(vx, 0x9e3779b9) ^ Math.imul(vy, 0x85ebca6b) ^ Math.imul(vz, 0xc2b2ae35);
  h ^= h >>> 16;
  h  = Math.imul(h, 0x45d9f3b);
  h ^= h >>> 16;
  return h >>> 0; // unsigned
}

function voxelDownsample(
  geo: THREE.BufferGeometry,
  voxelSize: number
): THREE.BufferGeometry {
  const positions = geo.attributes.position.array as Float32Array;
  const srcColors = geo.attributes.color?.array as Float32Array | undefined;
  const count     = positions.length / 3;

  const bitset  = new Uint8Array(BITSET_BYTES);
  // Pre-allocate worst-case; trimmed at the end via subarray
  const indices = new Uint32Array(count);
  let   outCount = 0;

  for (let i = 0; i < count; i++) {
    const vx   = Math.floor(positions[i * 3]     / voxelSize);
    const vy   = Math.floor(positions[i * 3 + 1] / voxelSize);
    const vz   = Math.floor(positions[i * 3 + 2] / voxelSize);
    const slot = hashVoxel(vx, vy, vz) & (BITSET_SLOTS - 1);
    const byte = slot >> 3;
    const bit  = 1 << (slot & 7);

    if (!(bitset[byte] & bit)) {
      bitset[byte] |= bit;
      indices[outCount++] = i;
    }
  }

  const newPos = new Float32Array(outCount * 3);
  const newCol = srcColors ? new Float32Array(outCount * 3) : undefined;

  for (let j = 0; j < outCount; j++) {
    const i = indices[j];
    newPos[j * 3]     = positions[i * 3];
    newPos[j * 3 + 1] = positions[i * 3 + 1];
    newPos[j * 3 + 2] = positions[i * 3 + 2];
    if (newCol && srcColors) {
      newCol[j * 3]     = srcColors[i * 3];
      newCol[j * 3 + 1] = srcColors[i * 3 + 1];
      newCol[j * 3 + 2] = srcColors[i * 3 + 2];
    }
  }

  const result = new THREE.BufferGeometry();
  result.setAttribute("position", new THREE.BufferAttribute(newPos, 3));
  if (newCol) result.setAttribute("color", new THREE.BufferAttribute(newCol, 3));
  return result;
}

// Color points by Y (height in Y-up world). Skipped when geometry already has colors.
function addHeightColors(geo: THREE.BufferGeometry): THREE.BufferGeometry {
  if (geo.attributes.color) return geo;

  const positions = geo.attributes.position.array as Float32Array;
  const count     = positions.length / 3;
  const colors    = new Float32Array(count * 3);

  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < count; i++) {
    const y = positions[i * 3 + 1];
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const range = maxY - minY || 1;

  const color = new THREE.Color();
  for (let i = 0; i < count; i++) {
    const t = (positions[i * 3 + 1] - minY) / range;
    // hue: 0.667 (blue) at ground → 0.0 (red) at top
    color.setHSL(0.667 * (1 - t), 1.0, 0.55);
    colors[i * 3]     = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;
  }

  const result = geo.clone();
  result.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  return result;
}

export function PointCloud({
  geometry,
  pointSize = 0.03,
  voxelSize = 0,
  cameraMode = "3d",
  zCeiling = null,
  onPick,
  pickEnabled = false,
  pointsRef,
}: PointCloudProps) {
  const renderGeo = useMemo(() => {
    const downsampled =
      voxelSize > 0 ? voxelDownsample(geometry, voxelSize) : geometry;
    return addHeightColors(downsampled);
  }, [geometry, voxelSize]);

  // In 2D mode sizeAttenuation=false → size is screen pixels.
  // Caller already passes the correct value for the active mode.
  const is2D = cameraMode === "2d";

  // Live-updatable uniform driving the fragment-shader clip.
  // Stable object identity so three.js keeps the same program.
  const zCeilingUniform = useMemo<{ value: number }>(
    () => ({ value: Number.POSITIVE_INFINITY }),
    []
  );
  useEffect(() => {
    zCeilingUniform.value =
      zCeiling == null || !Number.isFinite(zCeiling)
        ? Number.POSITIVE_INFINITY
        : zCeiling;
  }, [zCeiling, zCeilingUniform]);

  // Inject a Y-threshold discard into the built-in PointsMaterial shader.
  // Memoized so three.js compiles the program only once.
  const onBeforeCompile = useMemo(() => {
    const patch = (shader: THREE.WebGLProgramParametersWithUniforms) => {
      shader.uniforms.uZCeiling = zCeilingUniform;

      shader.vertexShader =
        "varying float vWorldY;\n" +
        shader.vertexShader.replace(
          "#include <begin_vertex>",
          `#include <begin_vertex>
           vWorldY = (modelMatrix * vec4(position, 1.0)).y;`
        );

      shader.fragmentShader =
        "varying float vWorldY;\nuniform float uZCeiling;\n" +
        shader.fragmentShader.replace(
          "void main() {",
          `void main() {
             if (vWorldY > uZCeiling) discard;`
        );
    };
    return patch;
  }, [zCeilingUniform]);

  return (
    <points
      ref={pointsRef}
      geometry={renderGeo}
      onClick={
        pickEnabled && onPick
          ? (e) => {
              e.stopPropagation();
              // e.point is the world-space intersection; with a Points raycast
              // it lands on the picked point itself (not an interpolated plane),
              // so the marker sits exactly on the PCD surface.
              onPick([e.point.x, e.point.y, e.point.z]);
            }
          : undefined
      }
    >
      <pointsMaterial
        size={pointSize}
        vertexColors
        sizeAttenuation={!is2D}
        transparent
        opacity={0.9}
        depthWrite={false}
        onBeforeCompile={onBeforeCompile}
      />
    </points>
  );
}
