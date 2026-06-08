"use client";

import { useState, useMemo } from "react";
import { motion } from "framer-motion";
import { Bookmark, BookmarkCheck, ChevronDown, ChevronUp, Search } from "lucide-react";
import type { PredictionResult, ProbabilityTier } from "@/lib/types";
import { getProbabilityTier } from "@/lib/types";
import ProbabilityBadge from "./ProbabilityBadge";

interface ResultsTableProps {
  results: PredictionResult[];
  onRowClick: (result: PredictionResult) => void;
  savedIds: Set<string>;
  onToggleSaved: (result: PredictionResult) => void;
}

type SortDir = "desc" | "asc";
type FilterType = "ALL" | "IIT" | "NIT" | "IIIT" | "GFTI";
type TierFilter = "ALL" | ProbabilityTier;

const TYPE_BADGE: Record<string, string> = {
  IIT: "bg-neutral-900 text-neutral-200 border-neutral-700",
  NIT: "bg-neutral-900 text-cyan-300 border-cyan-500/25",
  IIIT: "bg-neutral-900 text-violet-300 border-violet-500/25",
  GFTI: "bg-neutral-900 text-neutral-400 border-neutral-700",
};

// Skeleton shown while simulation is running
export function TableSkeleton() {
  return (
    <div className="space-y-2 mt-4">
      {Array.from({ length: 6 }).map((_, i) => (
        <div
          key={i}
          className="h-16 rounded-md bg-neutral-800/40 animate-pulse"
          style={{ opacity: 1 - i * 0.12 }}
        />
      ))}
    </div>
  );
}

export default function ResultsTable({
  results,
  onRowClick,
  savedIds,
  onToggleSaved,
}: ResultsTableProps) {
  const [search, setSearch] = useState("");
  const [filterType, setFilterType] = useState<FilterType>("ALL");
  const [tierFilter, setTierFilter] = useState<TierFilter>("ALL");
  // Default: highest probability first (colleges you can get = top)
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const filtered = useMemo(() => {
    const q = search.toLowerCase().trim();
    const subset = results.filter((r) => {
      if (filterType !== "ALL" && r.institute_type !== filterType) return false;
      if (tierFilter !== "ALL" && getProbabilityTier(r.probability_percent) !== tierFilter) return false;
      if (q && !r.institute_name.toLowerCase().includes(q) && !r.program_name.toLowerCase().includes(q)) return false;
      return true;
    });

    // Always produce a fresh sorted array — never mutate in place
    return [...subset].sort((a, b) =>
      sortDir === "desc"
        ? b.probability_percent - a.probability_percent
        : a.probability_percent - b.probability_percent
    );
  }, [results, search, filterType, tierFilter, sortDir]);

  const FILTERS: FilterType[] = ["ALL", "IIT", "NIT", "IIIT", "GFTI"];
  const TIER_FILTERS: { value: TierFilter; label: string }[] = [
    { value: "ALL", label: "All odds" },
    { value: "safe", label: "Safe" },
    { value: "target", label: "Target" },
    { value: "reach", label: "Reach" },
  ];

  return (
    <div className="space-y-4">
      {/* ── Filter Bar ───────────────────────────────────────────────── */}
      <div className="space-y-3 rounded-md border border-neutral-800 bg-neutral-900/25 p-3">
        <div className="flex flex-col xl:flex-row gap-3 items-start xl:items-center">
        {/* Search */}
        <div className="relative flex-1 w-full">
          <Search size={15} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-neutral-500 pointer-events-none" />
          <input
            type="text"
            placeholder="Search institute or branch…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full bg-neutral-950 border border-neutral-800 rounded-md pl-9 pr-4 py-2.5
              text-sm text-neutral-200 placeholder:text-neutral-600 outline-none
              focus:ring-2 focus:ring-cyan-500/25 focus:border-cyan-500 transition-colors"
          />
        </div>

        {/* Type filter pills */}
        <div className="flex gap-1.5 flex-wrap shrink-0">
          {FILTERS.map((f) => (
            <button
              key={f}
              onClick={() => setFilterType(f)}
              className={`px-3 py-1.5 rounded-md text-xs font-semibold border transition-colors duration-150
                ${filterType === f
                  ? "bg-cyan-600 border-cyan-500 text-white"
                  : "bg-neutral-950 border-neutral-800 text-neutral-400 hover:border-neutral-600 hover:text-neutral-200"
                }`}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

        <div className="flex flex-wrap gap-1.5">
          {TIER_FILTERS.map((filter) => (
            <button
              key={filter.value}
              onClick={() => setTierFilter(filter.value)}
              className={`px-3 py-1.5 rounded-md text-xs font-medium border transition-colors duration-150
                ${tierFilter === filter.value
                  ? "bg-neutral-200 border-neutral-200 text-neutral-950"
                  : "bg-neutral-950 border-neutral-800 text-neutral-400 hover:border-neutral-600 hover:text-neutral-200"
                }`}
            >
              {filter.label}
            </button>
          ))}
        </div>
      </div>

      {/* ── Desktop Table ─────────────────────────────────────────────── */}
      <div className="hidden sm:block rounded-md border border-neutral-800 overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-neutral-800 bg-neutral-900/80">
              <th className="w-12 px-3 py-3.5" />
              <th className="text-left px-5 py-3.5 text-xs text-neutral-500 font-medium w-[42%]">
                Institute / Program
              </th>
              <th className="text-left px-5 py-3.5 text-xs text-neutral-500 font-medium">
                Quota · Category
              </th>
              <th className="text-right px-5 py-3.5 text-xs text-neutral-500 font-medium">
                Proj. Cutoff
              </th>
              {/* Clickable sort header */}
              <th
                onClick={() => setSortDir((d) => (d === "desc" ? "asc" : "desc"))}
                className="text-right px-5 py-3.5 text-xs text-neutral-400 font-medium
                  cursor-pointer hover:text-white select-none transition-colors group"
              >
                <span className="inline-flex items-center gap-1 justify-end">
                  Probability
                  <span className="text-cyan-400 group-hover:text-cyan-300 transition-colors">
                    {sortDir === "desc" ? <ChevronDown size={13} /> : <ChevronUp size={13} />}
                  </span>
                </span>
              </th>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 ? (
              <tr>
                <td colSpan={5} className="px-5 py-12 text-center text-neutral-600 text-sm">
                  No results match your filters.
                </td>
              </tr>
            ) : (
              filtered.map((r, i) => (
                <motion.tr
                  // key includes sort direction so Framer re-renders on sort change
                  key={`${sortDir}-${r.id}`}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: Math.min(i * 0.025, 0.4), duration: 0.2 }}
                  onClick={() => onRowClick(r)}
                  className="border-b border-neutral-800/70 last:border-0 hover:bg-neutral-900/70
                    cursor-pointer transition-colors group"
                >
                  <td className="px-3 py-4 align-top">
                    <button
                      type="button"
                      onClick={(event) => {
                        event.stopPropagation();
                        onToggleSaved(r);
                      }}
                      className={`rounded-md border p-1.5 transition-colors ${
                        savedIds.has(r.id)
                          ? "border-cyan-500/40 bg-cyan-500/10 text-cyan-300"
                          : "border-neutral-800 text-neutral-600 hover:border-neutral-600 hover:text-neutral-200"
                      }`}
                      aria-label={savedIds.has(r.id) ? "Remove from choice list" : "Save to choice list"}
                    >
                      {savedIds.has(r.id) ? <BookmarkCheck size={15} /> : <Bookmark size={15} />}
                    </button>
                  </td>
                  {/* Institute + Program */}
                  <td className="px-5 py-4">
                    <div className="flex items-start gap-3">
                      <span className={`shrink-0 text-[10px] font-bold px-1.5 py-0.5 rounded-sm border mt-0.5 ${TYPE_BADGE[r.institute_type]}`}>
                        {r.institute_type}
                      </span>
                      <div>
                        <p className="text-neutral-100 font-medium group-hover:text-white transition-colors leading-tight">
                          {r.institute_name}
                        </p>
                        <p className="text-neutral-500 text-xs mt-0.5">{r.program_name}</p>
                      </div>
                    </div>
                  </td>

                  {/* Quota + Category */}
                  <td className="px-5 py-4 text-xs">
                    <div className="space-y-1">
                      <span className={`inline-flex items-center px-2 py-0.5 rounded-md text-[10px] font-bold border
                        ${r.quota_applied === "HS"
                          ? "bg-emerald-500/10 text-emerald-300 border-emerald-500/25"
                          : "bg-cyan-500/10 text-cyan-300 border-cyan-500/25"
                        }`}>
                        {r.quota_applied}
                      </span>
                      <div className="text-neutral-600">{r.category}</div>
                    </div>
                  </td>

                  {/* Projected cutoff */}
                  <td className="px-5 py-4 text-right">
                    <span className="text-neutral-300 tabular-nums font-medium">
                      {r.projected_closing_rank.toLocaleString()}
                    </span>
                  </td>

                  {/* Probability badge */}
                  <td className="px-5 py-4">
                    <div className="flex justify-end">
                      <ProbabilityBadge probability={r.probability_percent} />
                    </div>
                  </td>
                </motion.tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* ── Mobile Cards ──────────────────────────────────────────────── */}
      <div className="sm:hidden space-y-3">
        {filtered.length === 0 ? (
          <p className="text-center text-neutral-600 py-10 text-sm">
            No results match your filters.
          </p>
        ) : (
          filtered.map((r, i) => (
            <motion.div
              key={`${sortDir}-${r.id}`}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: Math.min(i * 0.025, 0.4) }}
              role="button"
              tabIndex={0}
              onClick={() => onRowClick(r)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  onRowClick(r);
                }
              }}
              className="w-full text-left p-4 rounded-md border border-neutral-800
                bg-neutral-900/45 hover:bg-neutral-900 transition-colors hover:border-neutral-700"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-start gap-2.5 min-w-0">
                  <span className={`shrink-0 text-[10px] font-bold px-1.5 py-0.5 rounded-sm border mt-0.5 ${TYPE_BADGE[r.institute_type]}`}>
                    {r.institute_type}
                  </span>
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-neutral-100 truncate">{r.institute_name}</p>
                    <p className="text-xs text-neutral-500 mt-0.5 truncate">{r.program_name}</p>
                  </div>
                </div>
                <div className="flex shrink-0 items-start gap-2">
                  <ProbabilityBadge probability={r.probability_percent} showMeter={false} />
                  <span
                    role="button"
                    tabIndex={0}
                    onClick={(event) => {
                      event.stopPropagation();
                      onToggleSaved(r);
                    }}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        event.stopPropagation();
                        onToggleSaved(r);
                      }
                    }}
                    className={`rounded-md border p-1.5 transition-colors ${
                      savedIds.has(r.id)
                        ? "border-cyan-500/40 bg-cyan-500/10 text-cyan-300"
                        : "border-neutral-800 text-neutral-600"
                    }`}
                    aria-label={savedIds.has(r.id) ? "Remove from choice list" : "Save to choice list"}
                  >
                    {savedIds.has(r.id) ? <BookmarkCheck size={14} /> : <Bookmark size={14} />}
                  </span>
                </div>
              </div>
              <div className="mt-3 flex items-center gap-2 text-xs text-neutral-500">
                <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-bold border
                  ${r.quota_applied === "HS"
                    ? "bg-emerald-500/10 text-emerald-300 border-emerald-500/25"
                    : "bg-cyan-500/10 text-cyan-300 border-cyan-500/25"
                  }`}>
                  {r.quota_applied}
                </span>
                <span>·</span>
                <span>{r.category}</span>
                <span>·</span>
                <span>Proj. {r.projected_closing_rank.toLocaleString()}</span>
              </div>
            </motion.div>
          ))
        )}
      </div>

      {filtered.length > 0 && (
        <p className="text-center text-xs text-neutral-600 pt-1">
          Showing {filtered.length} of {results.length} options ·{" "}
          {sortDir === "desc" ? "Highest probability first" : "Lowest probability first"}
        </p>
      )}
    </div>
  );
}
