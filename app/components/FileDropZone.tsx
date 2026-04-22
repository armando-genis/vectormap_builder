"use client";

import { useCallback, useRef, useState } from "react";

interface FileDropZoneProps {
  onFile: (file: File) => void;
  error?: string;
}

export function FileDropZone({ onFile, error }: FileDropZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) onFile(file);
    },
    [onFile]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    if (!e.currentTarget.contains(e.relatedTarget as Node)) {
      setIsDragging(false);
    }
  }, []);

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onFile(file);
    },
    [onFile]
  );

  return (
    <div
      className="absolute inset-0 flex items-center justify-center bg-[#050810] z-10"
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
    >
      {/* Background grid pattern */}
      <div
        className="absolute inset-0 opacity-10"
        style={{
          backgroundImage:
            "linear-gradient(rgba(0,180,255,0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(0,180,255,0.3) 1px, transparent 1px)",
          backgroundSize: "40px 40px",
        }}
      />

      {/* Glow behind card */}
      <div className="absolute w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />

      {/* Drop card */}
      <div
        className={`relative flex flex-col items-center gap-6 rounded-2xl border-2 p-12 w-[420px] transition-all duration-200 cursor-pointer select-none
          ${
            isDragging
              ? "border-cyan-400 bg-cyan-500/10 shadow-[0_0_40px_rgba(0,220,255,0.3)]"
              : "border-white/10 bg-white/[0.04] hover:border-cyan-500/50 hover:bg-white/[0.06]"
          }`}
        onClick={() => inputRef.current?.click()}
      >
        {/* Icon */}
        <div
          className={`flex h-20 w-20 items-center justify-center rounded-full border transition-colors duration-200
            ${isDragging ? "border-cyan-400 bg-cyan-500/20" : "border-white/15 bg-white/5"}`}
        >
          <svg
            className={`w-9 h-9 transition-colors duration-200 ${isDragging ? "text-cyan-300" : "text-white/50"}`}
            fill="none"
            stroke="currentColor"
            strokeWidth={1.5}
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M12 16.5V9.75m0 0 3 3m-3-3-3 3M6.75 19.5a4.5 4.5 0 0 1-1.41-8.775 5.25 5.25 0 0 1 10.233-2.33 3 3 0 0 1 3.758 3.848A3.752 3.752 0 0 1 18 19.5H6.75Z"
            />
          </svg>
        </div>

        {/* Text */}
        <div className="text-center">
          <p
            className={`text-lg font-semibold transition-colors duration-200 ${isDragging ? "text-cyan-300" : "text-white/80"}`}
          >
            {isDragging ? "Release to load" : "Drop PCD file here"}
          </p>
          <p className="mt-1.5 text-sm text-white/35">
            or{" "}
            <span className="text-cyan-400 underline underline-offset-2">
              browse from your computer
            </span>
          </p>
          <p className="mt-3 text-xs text-white/20 font-mono">
            Supports .pcd (ASCII &amp; binary)
          </p>
        </div>

        {/* Error */}
        {error && (
          <div className="w-full rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-2.5">
            <p className="text-xs text-red-400 text-center">{error}</p>
          </div>
        )}

        <input
          ref={inputRef}
          type="file"
          accept=".pcd"
          className="hidden"
          onChange={handleFileInput}
          onClick={(e) => (e.currentTarget.value = "")}
        />
      </div>

      {/* Title */}
      <div className="absolute top-8 left-8 flex items-center gap-2.5">
        <div className="h-2 w-2 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(0,220,255,0.8)]" />
        <span className="text-sm font-semibold tracking-widest text-white/50 uppercase font-mono">
          VectorMap Builder
        </span>
      </div>
    </div>
  );
}
