"use client";

import { useEffect, useState } from "react";
import type { Lanelet, LaneletTurn } from "./types";
import type { JointType } from "./geometry";

interface LaneletPropertiesProps {
  lanelets: Lanelet[];        // full list
  selectedIds: Set<number>;
  onUpdate: (ids: number[], patch: Partial<Lanelet>) => void;
  onResizeWidth: (ids: number[], newWidth: number) => void;
  onReverse: (ids: number[]) => void;
  onDelete: (ids: number[]) => void;
  onDeselectAll: () => void;
  /** Spawn a neighbor lanelet sharing the selected lanelet's left/right edge. */
  onDuplicateNeighbor: (sourceId: number, side: "left" | "right") => void;
  /** Create a connector lanelet from `fromId`'s end to `toId`'s start. */
  onCreateJoint: (fromId: number, toId: number, type: JointType) => void;
}

export function LaneletProperties({
  lanelets,
  selectedIds,
  onUpdate,
  onResizeWidth,
  onReverse,
  onDelete,
  onDeselectAll,
  onDuplicateNeighbor,
  onCreateJoint,
}: LaneletPropertiesProps) {
  // Joint panel local state — target lanelet id and turn type.
  // Hooks must run unconditionally, so declare them before the early
  // return on empty selection.
  const [jointTargetId, setJointTargetId] = useState<number | null>(null);
  const [jointType, setJointType] = useState<JointType>("straight");

  // Reset target when the selection changes to a different single lanelet
  // (stale id from a previous selection must not leak through).
  const singleId =
    selectedIds.size === 1 ? Array.from(selectedIds)[0] : null;
  useEffect(() => {
    setJointTargetId(null);
  }, [singleId]);

  if (selectedIds.size === 0) return null;

  const selected = lanelets.filter((l) => selectedIds.has(l.id));
  const single   = selected.length === 1 ? selected[0] : null;
  const ids      = selected.map((l) => l.id);

  // For multi-selection, show the value only when it's the same across all.
  const common = <K extends keyof Lanelet>(key: K): Lanelet[K] | "__mixed__" => {
    const first = selected[0][key];
    for (let i = 1; i < selected.length; i++) {
      if (selected[i][key] !== first) return "__mixed__";
    }
    return first;
  };

  const width = common("width");
  const sub   = common("subType");
  const turn  = common("turnDirection");
  const speed = common("speedLimit");

  // Turn direction and joints don't apply to crosswalks — they're a
  // crossing area, not a directional travel lane. Hide those UI bits when
  // every selected lanelet is a crosswalk.
  const allCrosswalk = sub === "crosswalk";

  const kindLabel = sub === "__mixed__"
    ? "mixed"
    : sub === "crosswalk"
      ? "crosswalk"
      : "lanelet";

  return (
    <div className="absolute top-20 right-5 flex flex-col gap-3 rounded-xl border border-white/10 bg-black/70 p-4 backdrop-blur-md w-64 pointer-events-auto">

      <div className="flex items-center justify-between">
        <div className="text-[10px] font-mono text-white/40 uppercase tracking-wider">
          {single
            ? `${kindLabel === "crosswalk" ? "Crosswalk" : "Lanelet"} #${single.id}`
            : `${selected.length} ${kindLabel}s`}
        </div>
        <button
          onClick={onDeselectAll}
          className="text-[10px] font-mono text-white/40 hover:text-white/80 cursor-pointer"
          title="Deselect (Esc)"
        >
          ✕
        </button>
      </div>

      {/* ── Width ─────────────────────────────────────────── */}
      <FieldRow label="Width">
        <input
          type="range"
          min={0.5}
          max={10}
          step={0.1}
          value={width === "__mixed__" ? 3 : (width as number)}
          onChange={(e) => onResizeWidth(ids, parseFloat(e.target.value))}
          className="w-full accent-cyan-400 cursor-pointer"
        />
        <span className="text-xs font-mono text-white/70 w-12 text-right">
          {width === "__mixed__" ? "—" : `${(width as number).toFixed(1)} m`}
        </span>
      </FieldRow>

      {/* Type is chosen by the drawing tool (Lanelet / Crosswalk buttons),
          not a dropdown here — mirrors MapToolbox's "pick your tool, draw"
          flow and avoids ambiguity like setting "crosswalk" on a lanelet
          that's already been given turn/speed metadata. */}

      {/* ── Turn direction (road lanelets only) ──────────── */}
      {!allCrosswalk && (
        <FieldRow label="Turn">
          <select
            value={
              turn === "__mixed__"
                ? ""
                : (turn as LaneletTurn) === null
                  ? "none"
                  : (turn as string)
            }
            onChange={(e) => {
              const v = e.target.value;
              const t: LaneletTurn = v === "none" ? null : (v as LaneletTurn);
              onUpdate(ids, { turnDirection: t });
            }}
            className="flex-1 bg-white/5 border border-white/10 rounded-md px-2 py-1 text-xs font-mono text-white/80 focus:outline-none focus:border-cyan-400/50 cursor-pointer"
          >
            {turn === "__mixed__" && <option value="">(mixed)</option>}
            <option value="none">none</option>
            <option value="straight">straight</option>
            <option value="left">left</option>
            <option value="right">right</option>
          </select>
        </FieldRow>
      )}

      {/* ── Speed limit ──────────────────────────────────── */}
      <FieldRow label="Speed">
        <input
          type="number"
          min={0}
          max={200}
          step={1}
          placeholder={speed === "__mixed__" ? "mixed" : "—"}
          value={
            speed === "__mixed__" || speed === undefined
              ? ""
              : (speed as number)
          }
          onChange={(e) => {
            const v = e.target.value;
            onUpdate(ids, {
              speedLimit: v === "" ? undefined : parseFloat(v),
            });
          }}
          className="w-full bg-white/5 border border-white/10 rounded-md px-2 py-1 text-xs font-mono text-white/80 focus:outline-none focus:border-cyan-400/50"
        />
        <span className="text-[10px] font-mono text-white/40 w-8">km/h</span>
      </FieldRow>

      {/* ── Add neighbor (single road lanelet only) ─────── */}
      {single && !allCrosswalk && (
        <div className="flex flex-col gap-1.5 pt-1 border-t border-white/10">
          <div className="text-[10px] font-mono text-white/40 uppercase tracking-wider pt-2">
            Add neighbor
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => onDuplicateNeighbor(single.id, "left")}
              className="flex-1 flex items-center justify-center gap-1.5 px-2 py-1.5 rounded-md text-[11px] font-mono text-white/70 border border-cyan-400/30 bg-cyan-500/10 hover:bg-cyan-500/20 hover:text-cyan-200 transition-colors cursor-pointer"
              title="Spawn a new lanelet to the LEFT, sharing the inner edge (lane change allowed)"
            >
              <svg className="w-3 h-3" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M10.5 19.5 3 12m0 0 7.5-7.5M3 12h18" />
              </svg>
              Left
            </button>
            <button
              onClick={() => onDuplicateNeighbor(single.id, "right")}
              className="flex-1 flex items-center justify-center gap-1.5 px-2 py-1.5 rounded-md text-[11px] font-mono text-white/70 border border-cyan-400/30 bg-cyan-500/10 hover:bg-cyan-500/20 hover:text-cyan-200 transition-colors cursor-pointer"
              title="Spawn a new lanelet to the RIGHT, sharing the inner edge (lane change allowed)"
            >
              <svg className="w-3 h-3" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M13.5 4.5 21 12m0 0-7.5 7.5M21 12H3" />
              </svg>
              Right
            </button>
          </div>
          <div className="text-[9px] font-mono text-white/30 leading-3 pt-0.5">
            Shared edge becomes dashed (lane change allowed)
          </div>
        </div>
      )}

      {/* ── Joint (single road lanelet only) ────────────── */}
      {single && !allCrosswalk && (() => {
        const others = lanelets.filter((l) => l.id !== single.id);
        const target = jointTargetId !== null
          ? others.find((l) => l.id === jointTargetId) ?? null
          : null;
        const canCreate = target !== null;
        return (
          <div className="flex flex-col gap-1.5 pt-1 border-t border-white/10">
            <div className="text-[10px] font-mono text-white/40 uppercase tracking-wider pt-2">
              Joint
            </div>

            {/* From / To — reads like "id #{from} → id #{to}" */}
            <div className="flex items-center gap-2 text-[11px] font-mono text-white/70">
              <span className="text-white/40">From</span>
              <span className="px-1.5 py-0.5 rounded bg-cyan-500/15 border border-cyan-400/30 text-cyan-200">
                #{single.id}
              </span>
              <svg className="w-3 h-3 text-white/30" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M13.5 4.5 21 12m0 0-7.5 7.5M21 12H3" />
              </svg>
              <span className="text-white/40">To</span>
              <select
                value={jointTargetId ?? ""}
                onChange={(e) => {
                  const v = e.target.value;
                  setJointTargetId(v === "" ? null : parseInt(v, 10));
                }}
                className="flex-1 bg-white/5 border border-white/10 rounded-md px-2 py-1 text-xs font-mono text-white/80 focus:outline-none focus:border-cyan-400/50 cursor-pointer"
                disabled={others.length === 0}
              >
                <option value="">
                  {others.length === 0 ? "— no others —" : "(select…)"}
                </option>
                {others.map((l) => (
                  <option key={l.id} value={l.id}>
                    #{l.id}
                  </option>
                ))}
              </select>
            </div>

            {/* Type — straight / left / right */}
            <FieldRow label="Shape">
              <div className="flex w-full rounded-md overflow-hidden border border-white/10">
                {(["straight", "left", "right"] as JointType[]).map((t) => (
                  <button
                    key={t}
                    onClick={() => setJointType(t)}
                    className={
                      "flex-1 px-2 py-1 text-[11px] font-mono transition-colors cursor-pointer " +
                      (jointType === t
                        ? "bg-cyan-500/30 text-cyan-100"
                        : "bg-white/5 text-white/60 hover:bg-white/10")
                    }
                  >
                    {t}
                  </button>
                ))}
              </div>
            </FieldRow>

            <button
              onClick={() => {
                if (target) onCreateJoint(single.id, target.id, jointType);
              }}
              disabled={!canCreate}
              className="flex items-center justify-center gap-1.5 px-2 py-1.5 rounded-md text-[11px] font-mono border transition-colors cursor-pointer disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-transparent border-emerald-400/30 bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-200 hover:text-emerald-100"
              title="Create a connector lanelet from this end to the target start"
            >
              <svg className="w-3 h-3" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M12 4v16m8-8H4" />
              </svg>
              Create joint
            </button>
            <div className="text-[9px] font-mono text-white/30 leading-3 pt-0.5">
              Connector joins #{single.id} end → #{target?.id ?? "?"} start
            </div>
          </div>
        );
      })()}

      {/* ── Actions ───────────────────────────────────────── */}
      <div className="flex gap-2 pt-1">
        {/* Reverse is only meaningful for directional road lanelets. */}
        {!allCrosswalk && (
          <button
            onClick={() => onReverse(ids)}
            className="flex-1 flex items-center justify-center gap-1.5 px-2 py-1.5 rounded-md text-[11px] font-mono text-white/60 border border-white/10 hover:text-white/90 hover:bg-white/5 transition-colors cursor-pointer"
            title="Flip direction of travel"
          >
            <svg className="w-3 h-3" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M7 16V4m0 0L3 8m4-4 4 4m6 0v12m0 0 4-4m-4 4-4-4" />
            </svg>
            Reverse
          </button>
        )}
        <button
          onClick={() => onDelete(ids)}
          className="flex-1 flex items-center justify-center gap-1.5 px-2 py-1.5 rounded-md text-[11px] font-mono text-red-300/80 border border-red-400/20 bg-red-500/5 hover:bg-red-500/15 hover:text-red-200 transition-colors cursor-pointer"
          title="Delete (Del)"
        >
          <svg className="w-3 h-3" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round"
              d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79" />
          </svg>
          Delete
        </button>
      </div>
    </div>
  );
}

function FieldRow({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] font-mono text-white/40 uppercase tracking-wider w-12 shrink-0">
        {label}
      </span>
      <div className="flex-1 flex items-center gap-2">{children}</div>
    </div>
  );
}
