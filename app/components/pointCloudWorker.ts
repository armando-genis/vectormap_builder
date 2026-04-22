/// <reference lib="webworker" />

/**
 * Point-cloud processing worker.
 *
 * Does the three heavy passes that previously ran on the main thread and
 * froze the UI for multi-second stretches on maps with >10 M points:
 *
 *   1. Voxel downsample (hash-set on a 16 M-slot bit array).
 *   2. Height-based color assignment (HSL ramp by Y).
 *   3. Spatial XZ grid chunking — group the survivors into tiles of
 *      roughly `targetChunkPoints` each, with per-tile AABBs.
 *
 * The main thread hands over its raw positions/colors via zero-copy
 * transfer. They're cached here so the voxel slider can re-run the full
 * pass without shipping the buffers again.
 *
 * Protocol:
 *   in  ← { type: "ingest", positions, colors?, voxelSize, targetChunkPoints }
 *   in  ← { type: "voxel",  voxelSize, targetChunkPoints }
 *   in  ← { type: "clear" }                  // release cached buffers
 *   out → { type: "progress", phase, done, total }
 *   out → { type: "chunks",   chunks, bbox, voxelPoints, rawPoints }
 *   out → { type: "error",    message }
 *
 * Chunk payloads transfer their `positions` / `colors` Float32Array
 * buffers back to the main thread (detaches them here), so a 50 MB
 * cloud round-trip costs a single pointer swap per chunk.
 */

// ---------------------------------------------------------------------------
// Types (duplicated main-side in PointCloud.tsx — keep the two in sync).
// ---------------------------------------------------------------------------

export type Vec3Tuple = [number, number, number];

export interface ChunkOut {
  /** xyz·N, Float32Array transferable. */
  positions: Float32Array;
  /** rgb·N float in [0,1], same N as positions/3. */
  colors:    Float32Array;
  /** World-space AABB of the points in this chunk. */
  min: Vec3Tuple;
  max: Vec3Tuple;
  /** Grid cell index (cz * nx + cx), useful for debug. */
  cellKey: number;
}

export type InMsg =
  | {
      type:              "ingest";
      positions:         Float32Array;
      colors?:           Float32Array;
      voxelSize:         number;
      targetChunkPoints: number;
    }
  | {
      type:              "voxel";
      voxelSize:         number;
      targetChunkPoints: number;
    }
  | { type: "clear" };

export type OutMsg =
  | {
      type:  "progress";
      phase: "voxel" | "chunk";
      done:  number;
      total: number;
    }
  | {
      type:        "chunks";
      chunks:      ChunkOut[];
      min:         Vec3Tuple;
      max:         Vec3Tuple;
      voxelPoints: number;  // after downsampling
      rawPoints:   number;  // before downsampling
    }
  | { type: "error"; message: string };

// ---------------------------------------------------------------------------

const ctx = self as unknown as DedicatedWorkerGlobalScope;

// Cached raw inputs. Kept between re-voxel requests so we don't have to
// round-trip the whole cloud every time the user nudges the slider.
let rawPositions: Float32Array | null = null;
let rawColors:    Float32Array | null = null;

// ---------------------------------------------------------------------------
// Voxel downsampling. Same algorithm as the old PointCloud.voxelDownsample,
// lifted into the worker so the main thread never blocks on it.
// ---------------------------------------------------------------------------

const BITSET_BITS  = 24;                  // 16 M slots
const BITSET_SLOTS = 1 << BITSET_BITS;
const BITSET_BYTES = BITSET_SLOTS >> 3;   // 2 MB

function hashVoxel(vx: number, vy: number, vz: number): number {
  let h =
    Math.imul(vx, 0x9e3779b9) ^
    Math.imul(vy, 0x85ebca6b) ^
    Math.imul(vz, 0xc2b2ae35);
  h ^= h >>> 16;
  h  = Math.imul(h, 0x45d9f3b);
  h ^= h >>> 16;
  return h >>> 0;
}

interface Downsampled {
  positions: Float32Array;
  colors:    Float32Array | null;
  count:     number;
}

function voxelDownsample(
  positions: Float32Array,
  colors:    Float32Array | null,
  voxelSize: number
): Downsampled {
  const count = positions.length / 3;

  // No voxel → identity pass, but don't alias the caller's buffers
  // because downstream steps will transfer ownership away.
  if (voxelSize <= 0) {
    return {
      positions: new Float32Array(positions),
      colors:    colors ? new Float32Array(colors) : null,
      count,
    };
  }

  const bitset  = new Uint8Array(BITSET_BYTES);
  const indices = new Uint32Array(count);
  let outCount = 0;

  // Progress granularity — roughly 20 pings for even huge clouds.
  const progressEvery = Math.max(1, Math.floor(count / 20));

  for (let i = 0; i < count; i++) {
    const vx = Math.floor(positions[i * 3]     / voxelSize);
    const vy = Math.floor(positions[i * 3 + 1] / voxelSize);
    const vz = Math.floor(positions[i * 3 + 2] / voxelSize);
    const slot = hashVoxel(vx, vy, vz) & (BITSET_SLOTS - 1);
    const byte = slot >> 3;
    const bit  = 1 << (slot & 7);

    if (!(bitset[byte] & bit)) {
      bitset[byte] |= bit;
      indices[outCount++] = i;
    }

    if ((i & (progressEvery - 1)) === 0) {
      postProgress("voxel", i, count);
    }
  }

  const newPos = new Float32Array(outCount * 3);
  const newCol = colors ? new Float32Array(outCount * 3) : null;

  for (let j = 0; j < outCount; j++) {
    const i = indices[j];
    newPos[j * 3]     = positions[i * 3];
    newPos[j * 3 + 1] = positions[i * 3 + 1];
    newPos[j * 3 + 2] = positions[i * 3 + 2];
    if (newCol && colors) {
      newCol[j * 3]     = colors[i * 3];
      newCol[j * 3 + 1] = colors[i * 3 + 1];
      newCol[j * 3 + 2] = colors[i * 3 + 2];
    }
  }

  return { positions: newPos, colors: newCol, count: outCount };
}

// ---------------------------------------------------------------------------
// HSL height ramp — blue at floor, red at ceiling. Consumes global Y min/max
// so every chunk shares the same color scale (no seams between tiles).
// ---------------------------------------------------------------------------

function hsl(h: number, s: number, l: number): Vec3Tuple {
  // Inline HSL → RGB. MDN formula, clamped into [0, 1].
  const k = (n: number) => (n + h * 12) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) =>
    l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  return [f(0), f(8), f(4)];
}

function paintByHeight(
  positions: Float32Array,
  minY: number,
  maxY: number
): Float32Array {
  const count  = positions.length / 3;
  const colors = new Float32Array(count * 3);
  const range  = maxY - minY || 1;

  for (let i = 0; i < count; i++) {
    const t = (positions[i * 3 + 1] - minY) / range;
    // hue: 0.667 (blue) at ground → 0.0 (red) at top
    const [r, g, b] = hsl(0.667 * (1 - t), 1.0, 0.55);
    colors[i * 3]     = r;
    colors[i * 3 + 1] = g;
    colors[i * 3 + 2] = b;
  }
  return colors;
}

// ---------------------------------------------------------------------------
// Spatial chunking — bin post-voxel points onto an XZ grid with ~equal point
// counts per cell. Each cell becomes an independent THREE.Points on the
// main thread, so the renderer can frustum-cull and picking gets a free
// per-chunk bounding-sphere early-out.
// ---------------------------------------------------------------------------

function computeBBox(positions: Float32Array) {
  const n = positions.length / 3;
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = positions[i * 3];
    const y = positions[i * 3 + 1];
    const z = positions[i * 3 + 2];
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  return { minX, minY, minZ, maxX, maxY, maxZ };
}

function chunkOnGrid(
  positions:         Float32Array,
  colors:            Float32Array,
  targetChunkPoints: number
): { chunks: ChunkOut[]; min: Vec3Tuple; max: Vec3Tuple } {
  const N = positions.length / 3;
  if (N === 0) {
    return {
      chunks: [],
      min: [0, 0, 0],
      max: [0, 0, 0],
    };
  }

  const bb = computeBBox(positions);
  const rangeX = Math.max(bb.maxX - bb.minX, 1e-3);
  const rangeZ = Math.max(bb.maxZ - bb.minZ, 1e-3);

  // Choose the grid so each cell has ~targetChunkPoints on average.
  // Derivation: area per cell = (rangeX·rangeZ) · targetChunkPoints / N,
  // side of cell ≈ sqrt(area per cell). Clamp to keep the number of
  // cells sane regardless of cloud density (prevents degenerate cases
  // — e.g. tiny clouds becoming a single enormous cell or millions of
  // empty cells on ultra-dense ones).
  const area      = rangeX * rangeZ;
  const rawCell   = Math.sqrt(Math.max(1, area * targetChunkPoints / N));
  const cellSize  = Math.max(8, Math.min(rawCell, 200));

  const nx = Math.max(1, Math.ceil(rangeX / cellSize));
  const nz = Math.max(1, Math.ceil(rangeZ / cellSize));
  const invCell = 1 / cellSize;

  const cellOf = new Int32Array(N);
  const counts = new Int32Array(nx * nz);

  for (let i = 0; i < N; i++) {
    const cx = Math.min(
      nx - 1,
      Math.max(0, Math.floor((positions[i * 3]     - bb.minX) * invCell))
    );
    const cz = Math.min(
      nz - 1,
      Math.max(0, Math.floor((positions[i * 3 + 2] - bb.minZ) * invCell))
    );
    const key = cz * nx + cx;
    cellOf[i] = key;
    counts[key]++;
  }

  // Pre-allocate per-cell typed arrays and pack a lookup from key → slot.
  const keyToSlot = new Int32Array(nx * nz).fill(-1);
  const chunks: ChunkOut[] = [];
  for (let k = 0; k < nx * nz; k++) {
    const c = counts[k];
    if (c === 0) continue;
    keyToSlot[k] = chunks.length;
    chunks.push({
      positions: new Float32Array(c * 3),
      colors:    new Float32Array(c * 3),
      // Seed min/max inverted so the first point wins.
      min: [Infinity, Infinity, Infinity],
      max: [-Infinity, -Infinity, -Infinity],
      cellKey: k,
    });
  }

  // Running write offsets (in point units, not floats).
  const offs = new Int32Array(chunks.length);

  const progressEvery = Math.max(1, Math.floor(N / 20));

  for (let i = 0; i < N; i++) {
    const slot  = keyToSlot[cellOf[i]];
    if (slot < 0) continue; // defensive; shouldn't happen
    const ch    = chunks[slot];
    const w     = offs[slot]++;
    const w3    = w * 3;
    const i3    = i * 3;

    const x = positions[i3];
    const y = positions[i3 + 1];
    const z = positions[i3 + 2];

    ch.positions[w3]     = x;
    ch.positions[w3 + 1] = y;
    ch.positions[w3 + 2] = z;
    ch.colors[w3]        = colors[i3];
    ch.colors[w3 + 1]    = colors[i3 + 1];
    ch.colors[w3 + 2]    = colors[i3 + 2];

    if (x < ch.min[0]) ch.min[0] = x;
    if (y < ch.min[1]) ch.min[1] = y;
    if (z < ch.min[2]) ch.min[2] = z;
    if (x > ch.max[0]) ch.max[0] = x;
    if (y > ch.max[1]) ch.max[1] = y;
    if (z > ch.max[2]) ch.max[2] = z;

    if ((i & (progressEvery - 1)) === 0) {
      postProgress("chunk", i, N);
    }
  }

  return {
    chunks,
    min: [bb.minX, bb.minY, bb.minZ],
    max: [bb.maxX, bb.maxY, bb.maxZ],
  };
}

// ---------------------------------------------------------------------------
// Pipeline entry + messaging.
// ---------------------------------------------------------------------------

function postProgress(
  phase: "voxel" | "chunk",
  done:  number,
  total: number
) {
  const msg: OutMsg = { type: "progress", phase, done, total };
  ctx.postMessage(msg);
}

function runPipeline(voxelSize: number, targetChunkPoints: number) {
  if (!rawPositions) {
    const err: OutMsg = { type: "error", message: "no cached cloud — send ingest first" };
    ctx.postMessage(err);
    return;
  }

  const { positions: dsPos, colors: dsCol, count: nVox } = voxelDownsample(
    rawPositions,
    rawColors,
    voxelSize
  );

  // Height colors if the source had none.
  const bb = computeBBox(dsPos);
  const finalColors = dsCol ?? paintByHeight(dsPos, bb.minY, bb.maxY);

  const { chunks, min, max } = chunkOnGrid(dsPos, finalColors, targetChunkPoints);

  // Transfer every chunk's buffers — zero-copy handoff. `.buffer` is
  // typed as ArrayBufferLike (could be SharedArrayBuffer), but we
  // allocated these Float32Arrays locally so they're plain ArrayBuffers.
  const transfer: ArrayBuffer[] = [];
  for (const c of chunks) {
    transfer.push(c.positions.buffer as ArrayBuffer, c.colors.buffer as ArrayBuffer);
  }

  const out: OutMsg = {
    type:        "chunks",
    chunks,
    min,
    max,
    voxelPoints: nVox,
    rawPoints:   rawPositions.length / 3,
  };
  ctx.postMessage(out, transfer);
}

ctx.onmessage = (e: MessageEvent<InMsg>) => {
  const msg = e.data;
  try {
    switch (msg.type) {
      case "ingest": {
        rawPositions = msg.positions;
        rawColors    = msg.colors ?? null;
        runPipeline(msg.voxelSize, msg.targetChunkPoints);
        break;
      }
      case "voxel": {
        runPipeline(msg.voxelSize, msg.targetChunkPoints);
        break;
      }
      case "clear": {
        rawPositions = null;
        rawColors    = null;
        break;
      }
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    ctx.postMessage({ type: "error", message } as OutMsg);
  }
};

// Keep TypeScript happy — this file is compiled as a module.
export {};
