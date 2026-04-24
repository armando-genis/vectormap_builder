"use client";

import dynamic from "next/dynamic";

const PointCloudViewer = dynamic(() => import("./PointCloudViewer"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full w-full items-center justify-center bg-[#050810]">
      <div className="h-2 w-2 rounded-full bg-cyan-400 animate-pulse shadow-[0_0_12px_rgba(0,220,255,0.9)]" />
    </div>
  ),
});

export function DynamicViewer() {
  return <PointCloudViewer />;
}
