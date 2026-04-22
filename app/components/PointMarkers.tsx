"use client";

import { useMemo } from "react";
import * as THREE from "three";

export type Marker = [number, number, number];

interface PointMarkersProps {
  markers: Marker[];
}

/**
 * Renders placed markers as pixel-sized dots that always appear on top of the
 * point cloud (depthTest disabled + elevated renderOrder).
 */
export function PointMarkers({ markers }: PointMarkersProps) {
  const geometry = useMemo(() => {
    const arr = new Float32Array(markers.length * 3);
    for (let i = 0; i < markers.length; i++) {
      arr[i * 3]     = markers[i][0];
      arr[i * 3 + 1] = markers[i][1];
      arr[i * 3 + 2] = markers[i][2];
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(arr, 3));
    return geo;
  }, [markers]);

  if (markers.length === 0) return null;

  return (
    <>
      {/* Outer halo (dimmer, larger) */}
      <points geometry={geometry} renderOrder={20}>
        <pointsMaterial
          size={14}
          sizeAttenuation={false}
          color="#22d3ee"
          transparent
          opacity={0.35}
          depthTest={false}
        />
      </points>

      {/* Inner core */}
      <points geometry={geometry} renderOrder={21}>
        <pointsMaterial
          size={7}
          sizeAttenuation={false}
          color="#ffffff"
          transparent
          depthTest={false}
        />
      </points>
    </>
  );
}
