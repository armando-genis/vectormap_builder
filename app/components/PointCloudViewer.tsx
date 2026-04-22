"use client";

import { useState, useCallback } from "react";
import * as THREE from "three";
import { FileDropZone } from "./FileDropZone";
import { SceneViewer } from "./SceneViewer";

type ViewerState = "idle" | "loading" | "loaded" | "error";

// Convert ROS/robot Z-up frame → Three.js Y-up frame
// ROS: X=forward, Y=left, Z=up  →  Three.js: X=right, Y=up, Z=toward
function transformZUpToYUp(geo: THREE.BufferGeometry): THREE.BufferGeometry {
  const src = geo.attributes.position.array as Float32Array;
  const count = src.length / 3;
  const dst = new Float32Array(src.length);

  for (let i = 0; i < count; i++) {
    dst[i * 3]     =  src[i * 3];      // X → X
    dst[i * 3 + 1] =  src[i * 3 + 2]; // Z → Y  (height axis)
    dst[i * 3 + 2] = -src[i * 3 + 1]; // Y → -Z
  }

  const result = geo.clone();
  result.setAttribute("position", new THREE.BufferAttribute(dst, 3));
  result.computeBoundingBox();
  result.computeBoundingSphere();
  return result;
}

export default function PointCloudViewer() {
  const [state, setState] = useState<ViewerState>("idle");
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null);
  const [fileName, setFileName] = useState("");
  const [pointCount, setPointCount] = useState(0);
  const [error, setError] = useState("");

  const handleFile = useCallback(async (file: File) => {
    setState("loading");
    setError("");
    setFileName(file.name);

    try {
      const { PCDLoader } = await import(
        "three/examples/jsm/loaders/PCDLoader.js"
      );
      const loader = new PCDLoader();
      const url = URL.createObjectURL(file);

      loader.load(
        url,
        (points: THREE.Points) => {
          URL.revokeObjectURL(url);
          const geo = transformZUpToYUp(points.geometry);
          const count = geo.attributes.position.count;
          setGeometry(geo);
          setPointCount(count);
          setState("loaded");
        },
        undefined,
        () => {
          URL.revokeObjectURL(url);
          setError("Failed to parse PCD file. Make sure it is a valid .pcd format.");
          setState("error");
        }
      );
    } catch {
      setError("Could not load the file reader module.");
      setState("error");
    }
  }, []);

  const handleReset = useCallback(() => {
    setGeometry(null);
    setFileName("");
    setPointCount(0);
    setError("");
    setState("idle");
  }, []);

  return (
    <div className="relative w-full h-full bg-[#050810]">
      {/* Loading spinner overlay */}
      {state === "loading" && (
        <div className="absolute inset-0 flex flex-col items-center justify-center z-20 bg-[#050810]">
          <div className="relative flex h-16 w-16 items-center justify-center">
            <div className="absolute inset-0 rounded-full border-2 border-cyan-500/20" />
            <div className="absolute inset-0 rounded-full border-2 border-t-cyan-400 border-r-transparent border-b-transparent border-l-transparent animate-spin" />
            <div className="h-2 w-2 rounded-full bg-cyan-400 shadow-[0_0_12px_rgba(0,220,255,0.9)]" />
          </div>
          <p className="mt-5 text-sm font-mono text-cyan-400/80 tracking-wider">
            Parsing point cloud…
          </p>
          <p className="mt-1.5 text-xs font-mono text-white/25 max-w-[220px] text-center truncate">
            {fileName}
          </p>
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
          fileName={fileName}
          pointCount={pointCount}
          onReset={handleReset}
        />
      )}
    </div>
  );
}
