/**
 * Session save / load for VectorMap Builder.
 *
 * Format: plain JSON with a `.vmb` extension (VectorMap Builder session).
 * The `version` field allows future migrations without breaking old files.
 */

import type { Lanelet } from "./lanelet/types";
import type { NodeRegistry } from "./lanelet/types";

export const SESSION_VERSION = 1 as const;

export interface VmbCropState {
  enabled: boolean;
  cx:      number;
  cz:      number;
  width:   number;
  length:  number;
  angle:   number;
  pitch:   number;
  roll:    number;
}

export interface VmbSession {
  version:   typeof SESSION_VERSION;
  savedAt:   string;
  voxelSize: number;
  zCeiling:  number;
  crop:      VmbCropState;
  registry:  NodeRegistry;
  lanelets:  Lanelet[];
}

/** Serialise current state and trigger a browser download. */
export function saveSession(
  session: Omit<VmbSession, "version" | "savedAt">,
  baseName = "session"
): void {
  const payload: VmbSession = {
    version:  SESSION_VERSION,
    savedAt:  new Date().toISOString(),
    ...session,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a   = document.createElement("a");
  a.href     = url;
  a.download = `${baseName}.vmb`;
  a.click();
  URL.revokeObjectURL(url);
}

/** Parse a `.vmb` file and return the session, or throw on bad data. */
export async function loadSessionFile(file: File): Promise<VmbSession> {
  const text = await file.text();
  const data = JSON.parse(text) as Partial<VmbSession>;
  if (data.version !== SESSION_VERSION) {
    throw new Error(
      `Unsupported session version ${data.version ?? "unknown"}. Expected ${SESSION_VERSION}.`
    );
  }
  if (!Array.isArray(data.lanelets) || !data.registry || !data.crop) {
    throw new Error("Session file is missing required fields.");
  }
  return data as VmbSession;
}
