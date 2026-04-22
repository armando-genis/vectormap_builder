import type * as THREE from "three";

/**
 * A spatial tile of the voxel-downsampled cloud, ready to render.
 *
 * Produced by `PointCloudViewer` (which owns the worker) and consumed
 * by `PointCloud` (which only renders). Kept in its own module so both
 * sides can import it without pulling in each other's implementations.
 */
export interface PointCloudChunk {
  /** Stable identifier so React can diff chunk arrays across re-voxel runs. */
  key: string;
  /** Geometry with `position` / `color` attributes and a tight bbox/sphere. */
  geo: THREE.BufferGeometry;
}

/**
 * Status surfaced by the point-cloud processing pipeline. Used to
 * drive the loading overlay and the in-scene stats pill.
 *
 * `phase` semantics:
 *   - "idle"       → nothing loaded
 *   - "processing" → worker is reading/parsing/voxeling/chunking
 *   - "ready"      → chunks are mounted and visible
 *   - "error"      → worker reported a failure; see `message`
 */
export interface PointCloudStats {
  phase:       "idle" | "processing" | "ready" | "error";
  rawPoints:   number;   // points in the loaded PCD
  voxelPoints: number;   // points after voxel downsampling
  chunkCount:  number;   // how many spatial tiles we're rendering
  /** 0..1 while `phase === "processing"`, null otherwise. */
  progress:    number | null;
  /** Human-readable status line ("Reading file…", "Chunking cloud…"). */
  message?:    string;
}
