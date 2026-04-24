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
      /**
       * Read the raw PCD file inside the worker, parse it, convert from
       * the ROS Z-up frame to Three.js Y-up, cache the result, then run
       * voxel + chunk. Use this for anything large enough that
       * `file.arrayBuffer()` on the main thread fails with
       * `NotReadableError` (Chrome rejects main-thread ArrayBuffer
       * allocations above ~2 GB).
       */
      type:              "parseFile";
      file:              File;
      voxelSize:         number;
      targetChunkPoints: number;
    }
  | {
      /** Legacy path: raw positions/colors already in main-thread memory. */
      type:              "ingest";
      positions:         Float32Array;
      colors?:           Float32Array;
      voxelSize:         number;
      targetChunkPoints: number;
    }
  | {
      /** Re-run voxel + chunk against the cached raw cloud. */
      type:              "voxel";
      voxelSize:         number;
      targetChunkPoints: number;
    }
  | { type: "clear" };

export type OutMsg =
  | {
      type:  "progress";
      phase: "reading" | "parsing" | "transforming" | "voxel" | "chunk";
      done:  number;
      total: number;
      /** Optional human-readable detail ("Reading 1.2 / 2.1 GB…"). */
      message?: string;
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

  // No voxel → pass the caller's buffers straight through. The old
  // code cloned here "because downstream steps will transfer ownership
  // away", but downstream (chunkOnGrid) only *reads* from them and
  // copies into fresh per-chunk arrays — nothing is transferred. For
  // multi-GB clouds that unnecessary clone doubled peak memory and
  // blew the allocator ("Array buffer allocation failed").
  if (voxelSize <= 0) {
    return {
      positions,
      colors,
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

  // Two passes over the positions: count per cell, then write per cell.
  //
  // A naive implementation stores the per-point cell index in an
  // `Int32Array(N)` between the two passes, which is ~680 MB for a
  // 170 M-point cloud — a meaningful fraction of V8's budget on top
  // of the ~2 GB positions array already live. Recomputing the cell
  // in pass 2 is a couple of floor()s per point: a few hundred
  // milliseconds of CPU to save a multi-hundred-MB allocation is the
  // right trade on clouds this size.
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
    counts[cz * nx + cx]++;
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
    const i3    = i * 3;
    const cx = Math.min(
      nx - 1,
      Math.max(0, Math.floor((positions[i3]     - bb.minX) * invCell))
    );
    const cz = Math.min(
      nz - 1,
      Math.max(0, Math.floor((positions[i3 + 2] - bb.minZ) * invCell))
    );
    const slot  = keyToSlot[cz * nx + cx];
    if (slot < 0) continue; // defensive; shouldn't happen
    const ch    = chunks[slot];
    const w     = offs[slot]++;
    const w3    = w * 3;

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
  phase: "reading" | "parsing" | "transforming" | "voxel" | "chunk",
  done:  number,
  total: number,
  message?: string
) {
  const msg: OutMsg = { type: "progress", phase, done, total, message };
  ctx.postMessage(msg);
}

// ---------------------------------------------------------------------------
// PCD file reading + parsing, worker-side.
//
// On the main thread, `file.arrayBuffer()` fails with NotReadableError
// for ~2+ GB files in Chrome. Inside a worker context those limits are
// separate — empirically workers can allocate larger buffers. Plus, if
// even the worker fails, we've got a second fallback: stream the file
// and accumulate into a Uint8Array of the same size, which at least
// surfaces a deterministic error instead of the opaque "NotReadable".
// ---------------------------------------------------------------------------

async function readFileIntoBuffer(file: File): Promise<ArrayBuffer> {
  const total = file.size;

  // Primary path — single allocation, minimal overhead when it works.
  try {
    postProgress("reading", 0, total, "Reading file…");
    const buf = await file.arrayBuffer();
    postProgress("reading", total, total);
    return buf;
  } catch (err) {
    // Fall through to the streaming path. Log for diagnostics — the
    // main thread will see our error only if *both* paths fail.
    const m = err instanceof Error ? err.message : String(err);
    console.warn("[worker] file.arrayBuffer() failed, trying stream fallback:", m);
  }

  // Streaming fallback. Reads the file in OS-sized chunks (typically
  // 64 KB) and copies each into a pre-allocated Uint8Array. This is
  // slightly slower than arrayBuffer() because of the per-chunk copy,
  // but it's the only option for files that exceed the main-thread
  // ArrayBuffer allocation ceiling on Chrome.
  const buf = new Uint8Array(total);
  const reader = file.stream().getReader();
  let offset = 0;
  let lastProgress = -1;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!value) continue;

    if (offset + value.length > total) {
      throw new Error(
        `File stream returned more bytes (${offset + value.length}) than File.size reports (${total}).`
      );
    }
    buf.set(value, offset);
    offset += value.length;

    // Coalesce progress pings to ~1% granularity — postMessage is
    // cheap but not free, and we don't want to drown the main thread
    // in updates while reading a 2 GB file.
    const pct = Math.floor((offset / total) * 100);
    if (pct !== lastProgress) {
      postProgress("reading", offset, total);
      lastProgress = pct;
    }
  }

  if (offset !== total) {
    throw new Error(
      `File stream ended at ${offset} bytes but File.size = ${total}.`
    );
  }
  return buf.buffer as ArrayBuffer;
}

/**
 * In-place ROS Z-up → Three.js Y-up axis swap:
 *   X → X
 *   Z → Y  (height axis)
 *   Y → -Z
 * Done in place to avoid doubling peak memory on huge clouds.
 */
function zUpToYUpInPlace(positions: Float32Array): void {
  const n = positions.length / 3;
  const progressEvery = Math.max(1, Math.floor(n / 20));
  for (let i = 0; i < n; i++) {
    const i3 = i * 3;
    const origY = positions[i3 + 1];
    const origZ = positions[i3 + 2];
    // X stays put.
    positions[i3 + 1] =  origZ;
    positions[i3 + 2] = -origY;

    if ((i & (progressEvery - 1)) === 0) {
      postProgress("transforming", i, n);
    }
  }
}

// ---------------------------------------------------------------------------
// Streaming binary PCD parser.
//
// For files above ~2 GB, we can't materialize a single ArrayBuffer
// holding the whole file (V8 throws "Array buffer allocation failed").
// PCDLoader needs one, so we skip it entirely for binary PCDs:
//
//   1. Read the first 64 KB to parse the ASCII header.
//   2. Compute per-point stride + per-field byte offsets from FIELDS /
//      SIZE / TYPE / COUNT.
//   3. Stream the file via `file.stream()`; for each Uint8Array chunk
//      the browser hands us, copy point records straight into the
//      preallocated output Float32Arrays (positions + optional colors).
//      Z-up → Y-up happens right here, so no second pass.
//
// Peak memory in this path is `positions.byteLength + (optional)
// colors.byteLength` — e.g. for a 137 M-point XYZ-only cloud that's
// ~1.6 GB in one allocation, which V8 happily allocates. No 3 GB
// intermediate buffer is ever needed.
// ---------------------------------------------------------------------------

interface PcdHeader {
  fields:     string[];
  sizes:      number[];  // per-field byte size
  types:      string[];  // "F" float, "I" int, "U" unsigned
  counts:     number[];  // per-field repeat count (almost always 1)
  numPoints:  number;
  dataFormat: "ascii" | "binary" | "binary_compressed";
  /** Byte offset in the file where point data begins. */
  dataStart:  number;
  /** Bytes per point record. */
  pointStride: number;
}

function parsePcdHeader(bytes: Uint8Array): PcdHeader {
  // PCD headers are strictly ASCII, so char offset === byte offset.
  const text = new TextDecoder("ascii").decode(bytes);

  let fields: string[] = [];
  let sizes:  number[] = [];
  let types:  string[] = [];
  let counts: number[] = [];
  let numPoints  = 0;
  let dataFormat: PcdHeader["dataFormat"] = "ascii";

  // Walk lines until the DATA directive.
  const lines = text.split("\n");
  let charCursor = 0;
  for (const raw of lines) {
    const lineLen = raw.length + 1; // +1 for the consumed "\n"
    const line = raw.replace(/\r$/, "").trim();

    if (line.startsWith("#") || line.length === 0) {
      charCursor += lineLen;
      continue;
    }
    const parts = line.split(/\s+/);
    const key = parts[0];
    const rest = parts.slice(1);

    switch (key) {
      case "FIELDS": fields = rest;                    break;
      case "SIZE":   sizes  = rest.map(Number);        break;
      case "TYPE":   types  = rest;                    break;
      case "COUNT":  counts = rest.map(Number);        break;
      case "POINTS": numPoints = parseInt(rest[0], 10); break;
      case "DATA": {
        const f = (rest[0] ?? "").toLowerCase();
        if (f !== "ascii" && f !== "binary" && f !== "binary_compressed") {
          throw new Error(`PCD: unknown DATA format "${rest[0]}"`);
        }
        dataFormat = f;
        charCursor += lineLen;
        if (!counts.length) counts = fields.map(() => 1);
        const pointStride = fields.reduce(
          (acc, _, i) => acc + sizes[i] * counts[i],
          0
        );
        if (!Number.isFinite(pointStride) || pointStride <= 0) {
          throw new Error("PCD: could not compute point stride from header.");
        }
        return {
          fields, sizes, types, counts, numPoints,
          dataFormat,
          dataStart:   charCursor,
          pointStride,
        };
      }
    }
    charCursor += lineLen;
  }
  throw new Error("PCD: missing DATA line in header (file truncated?).");
}

interface FieldExtractors {
  /** Reads a float value from a point record at the right offset + type. */
  xRead: (v: DataView, base: number) => number;
  yRead: (v: DataView, base: number) => number;
  zRead: (v: DataView, base: number) => number;
  /** Null when the cloud has no per-point color. */
  rgbRead: ((v: DataView, base: number, out: Float32Array, i3: number) => void) | null;
}

function buildExtractors(h: PcdHeader): FieldExtractors {
  const offsets: number[] = [];
  {
    let acc = 0;
    for (let i = 0; i < h.fields.length; i++) {
      offsets.push(acc);
      acc += h.sizes[i] * h.counts[i];
    }
  }

  function floatReader(fieldIdx: number) {
    const off  = offsets[fieldIdx];
    const size = h.sizes[fieldIdx];
    const type = h.types[fieldIdx];

    if (type === "F" && size === 4) return (v: DataView, base: number) => v.getFloat32(base + off, true);
    if (type === "F" && size === 8) return (v: DataView, base: number) => v.getFloat64(base + off, true);
    if (type === "I" && size === 1) return (v: DataView, base: number) => v.getInt8 (base + off);
    if (type === "I" && size === 2) return (v: DataView, base: number) => v.getInt16(base + off, true);
    if (type === "I" && size === 4) return (v: DataView, base: number) => v.getInt32(base + off, true);
    if (type === "U" && size === 1) return (v: DataView, base: number) => v.getUint8 (base + off);
    if (type === "U" && size === 2) return (v: DataView, base: number) => v.getUint16(base + off, true);
    if (type === "U" && size === 4) return (v: DataView, base: number) => v.getUint32(base + off, true);
    throw new Error(`PCD: unsupported field type/size ${type}${size}`);
  }

  const xIdx = h.fields.indexOf("x");
  const yIdx = h.fields.indexOf("y");
  const zIdx = h.fields.indexOf("z");
  if (xIdx < 0 || yIdx < 0 || zIdx < 0) {
    throw new Error(`PCD: missing x/y/z fields (got: ${h.fields.join(",")})`);
  }

  const xRead = floatReader(xIdx);
  const yRead = floatReader(yIdx);
  const zRead = floatReader(zIdx);

  // RGB field is stored either as a packed float32 whose raw 4 bytes
  // are [B, G, R, A] (the PCD convention — the "float" interpretation
  // is a PCDLoader historical quirk, not how the bytes actually mean
  // anything), or as three explicit "r" "g" "b" U8 fields.
  let rgbRead: FieldExtractors["rgbRead"] = null;
  const rgbIdx = h.fields.findIndex(f => f === "rgb" || f === "rgba");
  if (rgbIdx >= 0) {
    const off = offsets[rgbIdx];
    rgbRead = (v, base, out, i3) => {
      // Little-endian: byte 0 = B, byte 1 = G, byte 2 = R. Alpha (byte
      // 3) is ignored.
      out[i3]     = v.getUint8(base + off + 2) / 255; // R
      out[i3 + 1] = v.getUint8(base + off + 1) / 255; // G
      out[i3 + 2] = v.getUint8(base + off    ) / 255; // B
    };
  } else {
    const rIdx = h.fields.indexOf("r");
    const gIdx = h.fields.indexOf("g");
    const bIdx = h.fields.indexOf("b");
    if (rIdx >= 0 && gIdx >= 0 && bIdx >= 0) {
      const rR = floatReader(rIdx);
      const rG = floatReader(gIdx);
      const rB = floatReader(bIdx);
      rgbRead = (v, base, out, i3) => {
        out[i3]     = rR(v, base) / 255;
        out[i3 + 1] = rG(v, base) / 255;
        out[i3 + 2] = rB(v, base) / 255;
      };
    }
  }

  return { xRead, yRead, zRead, rgbRead };
}

/** Read a small prefix of the file and parse its header. */
async function readHeader(file: File): Promise<PcdHeader> {
  // 64 KB is plenty — real PCD headers are usually <1 KB. If the file
  // is smaller, slice() just clamps to file.size.
  const SNIFF = Math.min(64 * 1024, file.size);
  const slice = await file.slice(0, SNIFF).arrayBuffer();
  return parsePcdHeader(new Uint8Array(slice));
}

/**
 * Streaming parse for binary PCD. Applies Z-up → Y-up inline so we
 * never need a second pass over `positions`.
 */
async function parseBinaryStreaming(
  file:              File,
  header:            PcdHeader,
  voxelSize:         number,
  targetChunkPoints: number,
) {
  const N       = header.numPoints;
  const stride  = header.pointStride;
  const extract = buildExtractors(header);

  postProgress("parsing", 0, 1, "Preparing output buffers…");

  // Preallocate output arrays. These are the two big ones — everything
  // else in this path is a small transient Uint8Array.
  const positions = new Float32Array(N * 3);
  const colors    = extract.rgbRead ? new Float32Array(N * 3) : null;

  const reader   = file.stream().getReader();
  const dataSize = file.size - header.dataStart;

  // State machine for the consumer side of the stream.
  //   - `fileOffset` tracks absolute position in the file.
  //   - `carry`      holds <stride bytes left over from the previous
  //                  chunk that couldn't be turned into a full point yet.
  let fileOffset   = 0;
  let carry:       Uint8Array | null = null;
  let pointsWritten = 0;
  let lastPct       = -1;

  // Reusable scratch DataView so we don't allocate one per iteration.
  const applyPoints = (bytes: Uint8Array, startByte: number): number => {
    if (pointsWritten >= N) return 0;
    const pointsInChunk = Math.floor((bytes.byteLength - startByte) / stride);
    if (pointsInChunk <= 0) return 0;
    const toWrite = Math.min(pointsInChunk, N - pointsWritten);

    const view = new DataView(bytes.buffer, bytes.byteOffset + startByte, toWrite * stride);

    const { xRead, yRead, zRead, rgbRead } = extract;
    let base = 0;
    for (let i = 0; i < toWrite; i++) {
      const rawX = xRead(view, base);
      const rawY = yRead(view, base);
      const rawZ = zRead(view, base);

      // Z-up → Y-up, inlined.
      const o3 = pointsWritten * 3;
      positions[o3]     =  rawX;
      positions[o3 + 1] =  rawZ;
      positions[o3 + 2] = -rawY;

      if (colors && rgbRead) rgbRead(view, base, colors, o3);

      base += stride;
      pointsWritten++;
    }
    return toWrite * stride;
  };

  while (pointsWritten < N) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!value) continue;

    let view = value;

    // Trim leading bytes that still belong to the header.
    if (fileOffset < header.dataStart) {
      const skip = Math.min(header.dataStart - fileOffset, view.byteLength);
      fileOffset += skip;
      if (skip === view.byteLength) continue;
      view = view.subarray(skip);
    }

    // Combine with any leftover <stride bytes from the previous chunk.
    // `carry` is always tiny (bytes, not megabytes) so this temporary
    // allocation is negligible.
    let working: Uint8Array;
    if (carry && carry.byteLength > 0) {
      working = new Uint8Array(carry.byteLength + view.byteLength);
      working.set(carry, 0);
      working.set(view, carry.byteLength);
      carry = null;
    } else {
      working = view;
    }

    const consumed = applyPoints(working, 0);
    const leftover = working.byteLength - consumed;
    if (leftover > 0 && pointsWritten < N) {
      // Tail of `working` holds a partial point record. Clone so we
      // don't hold a reference to the big browser-owned chunk (which
      // would prevent GC).
      carry = new Uint8Array(working.subarray(consumed));
    }

    fileOffset += view.byteLength;

    // Progress at 1 %: postMessage is cheap but not free.
    const pct = Math.floor(((fileOffset - header.dataStart) / dataSize) * 100);
    if (pct !== lastPct) {
      postProgress("reading", fileOffset - header.dataStart, dataSize);
      lastPct = pct;
    }
  }

  try { reader.releaseLock(); } catch { /* already released */ }

  if (pointsWritten !== N) {
    throw new Error(
      `PCD stream ended early: expected ${N} points, got ${pointsWritten}.`
    );
  }

  // Positions is already Y-up; no separate transform pass. Cache and
  // hand off to the existing voxel/chunk pipeline.
  rawPositions = positions;
  rawColors    = colors;
  runPipeline(voxelSize, targetChunkPoints);
}

/**
 * Parse + ingest a PCD file, then run the usual voxel/chunk pipeline.
 *
 * For binary PCDs (the common case for large clouds) this uses the
 * streaming parser above — no whole-file buffer needed. For ASCII or
 * LZF-compressed PCDs we fall back to PCDLoader, which requires the
 * full file buffer; those formats aren't commonly used for >2 GB data
 * anyway.
 */
async function parseAndIngest(
  file:              File,
  voxelSize:         number,
  targetChunkPoints: number
) {
  // Peek at the header first so we can route to the right parser.
  postProgress("parsing", 0, 1, "Reading header…");
  const header = await readHeader(file);

  if (header.dataFormat === "binary") {
    // Streaming fast path — works for files of any size.
    await parseBinaryStreaming(file, header, voxelSize, targetChunkPoints);
    return;
  }

  // Fallback path (ASCII / binary_compressed). Requires the whole file
  // in a single ArrayBuffer — will throw "Array buffer allocation
  // failed" for >2 GB files, which is the best we can do without a
  // custom ASCII/LZF streaming parser.
  const buffer = await readFileIntoBuffer(file);

  postProgress("parsing", 0, 1, "Parsing PCD…");
  const { PCDLoader } = await import(
    "three/examples/jsm/loaders/PCDLoader.js"
  );
  const loader = new PCDLoader();
  const points = loader.parse(buffer);

  const posAttr = points.geometry.attributes.position;
  if (!posAttr || posAttr.count === 0) {
    throw new Error("PCD parsed but contains no points.");
  }

  // PCDLoader always produces Float32BufferAttribute for xyz, so the
  // underlying typed array is a Float32Array. We take ownership of it
  // directly to avoid another 1–2 GB allocation.
  const positions = posAttr.array as Float32Array;

  const colAttr = points.geometry.attributes.color;
  const colors  = colAttr ? (colAttr.array as Float32Array) : null;

  postProgress("transforming", 0, positions.length / 3);
  zUpToYUpInPlace(positions);

  // Cache and run — identical path to the "ingest" message after this.
  rawPositions = positions;
  rawColors    = colors;
  runPipeline(voxelSize, targetChunkPoints);
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
  // Wrapped so both sync and async branches funnel errors back through
  // the same channel. Parse/read are async (`parseFile`); the rest stay
  // sync and still benefit from the try/catch.
  (async () => {
    try {
      switch (msg.type) {
        case "parseFile": {
          await parseAndIngest(msg.file, msg.voxelSize, msg.targetChunkPoints);
          break;
        }
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
  })();
};

// Keep TypeScript happy — this file is compiled as a module.
export {};
