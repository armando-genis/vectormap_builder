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
  /**
   * Toggle "rect lock" on the given lanelets. On true, interior control
   * points are snapped onto the straight start→end axis and stay there.
   */
  onSetStraight: (ids: number[], straight: boolean) => void;
  onFitPlane: (ids: number[]) => void;
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
  onSetStraight,
  onFitPlane,
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
  const straight = common("straight");

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

      {/* ── Width ───────────────────────────────────────────
          Slider for dragging + numeric input for typing an exact
          value. Both hit the same onResizeWidth handler so the two
          controls stay in sync. */}
      <FieldRow label="Width">
        <input
          type="range"
          min={0.5}
          max={10}
          step={0.1}
          value={width === "__mixed__" ? 3 : (width as number)}
          onChange={(e) => onResizeWidth(ids, parseFloat(e.target.value))}
          className="flex-1 accent-cyan-400 cursor-pointer"
        />
        <WidthNumberInput
          value={width === "__mixed__" ? null : (width as number)}
          onChange={(v) => onResizeWidth(ids, v)}
        />
        <span className="text-[10px] font-mono text-white/40">m</span>
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

      {/* ── Rect lock ────────────────────────────────────── */}
      {/* Available for both roads and crosswalks — a rectangular
          crosswalk with only length-draggable ends is useful too. */}
      <FieldRow label="Shape">
        <button
          onClick={() => {
            const next = straight === "__mixed__" ? true : !(straight as boolean | undefined);
            onSetStraight(ids, next);
          }}
          className={
            "flex-1 flex items-center justify-between gap-2 px-2 py-1.5 rounded-md text-[11px] font-mono border transition-colors cursor-pointer " +
            (straight === true
              ? "border-amber-400/40 bg-amber-500/15 text-amber-200 hover:bg-amber-500/25"
              : "border-white/10 bg-white/5 text-white/60 hover:bg-white/10 hover:text-white/90")
          }
          title={
            straight === true
              ? "Rect lock ON — interior handles disabled, only endpoints move (click to unlock)"
              : "Rect lock OFF — drag interior handles to curve (click to lock as straight)"
          }
        >
          <span className="flex items-center gap-1.5">
            <svg className="w-3 h-3" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              {straight === true ? (
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M16.5 10.5V6.75a4.5 4.5 0 1 0-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 0 0 2.25-2.25v-6.75a2.25 2.25 0 0 0-2.25-2.25H6.75a2.25 2.25 0 0 0-2.25 2.25v6.75a2.25 2.25 0 0 0 2.25 2.25Z" />
              ) : (
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M13.5 10.5V6.75a4.5 4.5 0 1 1 9 0v3.75M3.75 21.75h10.5a2.25 2.25 0 0 0 2.25-2.25v-6.75a2.25 2.25 0 0 0-2.25-2.25H3.75a2.25 2.25 0 0 0-2.25 2.25v6.75a2.25 2.25 0 0 0 2.25 2.25Z" />
              )}
            </svg>
            {straight === "__mixed__" ? "Mixed" : straight === true ? "Rect lock" : "Curve"}
          </span>
          <span className="text-[9px] text-white/40">
            {straight === true ? "ends only" : "all handles"}
          </span>
        </button>
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
      <div className="flex flex-col gap-2 pt-1">
        <button
          onClick={() => onFitPlane(ids)}
          className="flex items-center justify-center gap-1.5 px-2 py-1.5 rounded-md text-[11px] font-mono text-violet-300/90 border border-violet-400/30 bg-violet-500/10 hover:bg-violet-500/20 hover:text-violet-200 transition-colors cursor-pointer"
          title="Fit the lanelet's boundary nodes onto the dominant road plane found in the point cloud below it (RANSAC — rejects car roofs and noise)"
        >
          <svg className="w-3 h-3" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round"
              d="M3 7h18M3 12h18M3 17h18" />
          </svg>
          Fit on plane
        </button>
        <div className="flex gap-2">
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

/**
 * Editable numeric input for the lanelet width. Mirrors the slider
 * value but keeps its own string buffer so typing intermediate
 * characters (empty, `.`, `0.`) doesn't get stomped on by a
 * round-trip through the `value` prop.
 *
 * Commits valid numbers on every keystroke, clamps to [0.5, 10] on
 * blur, and shows an em-dash placeholder when the selection is mixed
 * (null `value`) so users know they're editing a collective width.
 */
function WidthNumberInput({
  value,
  onChange,
}: {
  value: number | null;
  onChange: (v: number) => void;
}) {
  const MIN = 0.5;
  const MAX = 10;
  const fmt = (v: number) => v.toFixed(1);
  const [text, setText] = useState(() => (value === null ? "" : fmt(value)));
  useEffect(() => {
    setText(value === null ? "" : fmt(value));
  }, [value]);

  return (
    <input
      type="number"
      placeholder="—"
      step={0.1}
      min={MIN}
      max={MAX}
      value={text}
      onChange={(e) => {
        setText(e.target.value);
        const n = parseFloat(e.target.value);
        if (!Number.isNaN(n)) onChange(n);
      }}
      onBlur={() => {
        const n = parseFloat(text);
        if (Number.isNaN(n)) {
          setText(value === null ? "" : fmt(value));
          return;
        }
        const clamped = Math.max(MIN, Math.min(MAX, n));
        if (clamped !== n) {
          onChange(clamped);
          setText(fmt(clamped));
        }
      }}
      className="w-14 bg-white/5 border border-white/10 rounded px-1.5 py-0.5 text-[11px] font-mono text-white/80 text-right focus:outline-none focus:border-cyan-400/50"
    />
  );
}
