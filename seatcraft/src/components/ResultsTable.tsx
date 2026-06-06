"use client";

import { useState, useMemo } from "react";
import { motion } from "framer-motion";
import { Search, ChevronDown, ChevronUp } from "lucide-react";
import type { PredictionResult } from "@/lib/types";
import ProbabilityBadge from "./ProbabilityBadge";

interface ResultsTableProps {
  results: PredictionResult[];
  onRowClick: (result: PredictionResult) => void;
}

type SortDir = "desc" | "asc";
type FilterType = "ALL" | "IIT" | "NIT" | "IIIT" | "GFTI";

const TYPE_BADGE: Record<string, string> = {
  IIT: "bg-indigo-500/15 text-indigo-300 border-indigo-500/30",
  NIT: "bg-sky-500/15 text-sky-300 border-sky-500/30",
  IIIT: "bg-violet-500/15 text-violet-300 border-violet-500/30",
  GFTI: "bg-zinc-500/15 text-zinc-300 border-zinc-500/30",
};

// Skeleton shown while simulation is running
export function TableSkeleton() {
  return (
    <div className="space-y-2 mt-4">
      {Array.from({ length: 6 }).map((_, i) => (
        <div
          key={i}
          className="h-16 rounded-xl bg-zinc-800/40 animate-pulse"
          style={{ opacity: 1 - i * 0.12 }}
        />
      ))}
    </div>
  );
}

export default function ResultsTable({ results, onRowClick }: ResultsTableProps) {
  const [search, setSearch] = useState("");
  const [filterType, setFilterType] = useState<FilterType>("ALL");
  // Default: highest probability first (colleges you can get = top)
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const filtered = useMemo(() => {
    const q = search.toLowerCase().trim();
    const subset = results.filter((r) => {
      if (filterType !== "ALL" && r.institute_type !== filterType) return false;
      if (q && !r.institute_name.toLowerCase().includes(q) && !r.program_name.toLowerCase().includes(q)) return false;
      return true;
    });

    // Always produce a fresh sorted array — never mutate in place
    return [...subset].sort((a, b) =>
      sortDir === "desc"
        ? b.probability_percent - a.probability_percent
        : a.probability_percent - b.probability_percent
    );
  }, [results, search, filterType, sortDir]);

  const FILTERS: FilterType[] = ["ALL", "IIT", "NIT", "IIIT", "GFTI"];

  return (
    <div className="space-y-4">
      {/* ── Filter Bar ───────────────────────────────────────────────── */}
      <div className="flex flex-col sm:flex-row gap-3 items-start sm:items-center">
        {/* Search */}
        <div className="relative flex-1 w-full">
          <Search size={15} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-zinc-500 pointer-events-none" />
          <input
            type="text"
            placeholder="Search institute or branch…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full bg-zinc-900 border border-zinc-800 rounded-xl pl-9 pr-4 py-2.5
              text-sm text-zinc-200 placeholder:text-zinc-600 outline-none
              focus:ring-2 focus:ring-indigo-500/40 focus:border-indigo-500 transition-all"
          />
        </div>

        {/* Type filter pills */}
        <div className="flex gap-1.5 flex-wrap shrink-0">
          {FILTERS.map((f) => (
            <button
              key={f}
              onClick={() => setFilterType(f)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all duration-150
                ${filterType === f
                  ? "bg-indigo-600 border-indigo-500 text-white shadow-sm shadow-indigo-500/30"
                  : "bg-zinc-900 border-zinc-800 text-zinc-400 hover:border-zinc-600 hover:text-zinc-200"
                }`}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      {/* ── Desktop Table ─────────────────────────────────────────────── */}
      <div className="hidden sm:block rounded-2xl border border-zinc-800 overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-zinc-800 bg-zinc-900/80">
              <th className="text-left px-5 py-3.5 text-xs text-zinc-500 font-medium w-[42%]">
                Institute / Program
              </th>
              <th className="text-left px-5 py-3.5 text-xs text-zinc-500 font-medium">
                Quota · Category
              </th>
              <th className="text-right px-5 py-3.5 text-xs text-zinc-500 font-medium">
                Proj. Cutoff
              </th>
              {/* Clickable sort header */}
              <th
                onClick={() => setSortDir((d) => (d === "desc" ? "asc" : "desc"))}
                className="text-right px-5 py-3.5 text-xs text-zinc-400 font-medium
                  cursor-pointer hover:text-white select-none transition-colors group"
              >
                <span className="inline-flex items-center gap-1 justify-end">
                  Probability
                  <span className="text-indigo-400 group-hover:text-indigo-300 transition-colors">
                    {sortDir === "desc" ? <ChevronDown size={13} /> : <ChevronUp size={13} />}
                  </span>
                </span>
              </th>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 ? (
              <tr>
                <td colSpan={4} className="px-5 py-12 text-center text-zinc-600 text-sm">
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
                  className="border-b border-zinc-800/50 last:border-0 hover:bg-zinc-900/60
                    cursor-pointer transition-colors group"
                >
                  {/* Institute + Program */}
                  <td className="px-5 py-4">
                    <div className="flex items-start gap-3">
                      <span className={`shrink-0 text-[10px] font-bold px-1.5 py-0.5 rounded border mt-0.5 ${TYPE_BADGE[r.institute_type]}`}>
                        {r.institute_type}
                      </span>
                      <div>
                        <p className="text-zinc-100 font-medium group-hover:text-white transition-colors leading-tight">
                          {r.institute_name}
                        </p>
                        <p className="text-zinc-500 text-xs mt-0.5">{r.program_name}</p>
                      </div>
                    </div>
                  </td>

                  {/* Quota + Category */}
                  <td className="px-5 py-4 text-xs">
                    <div className="space-y-1">
                      <span className={`inline-flex items-center px-2 py-0.5 rounded-md text-[10px] font-bold border
                        ${r.quota_applied === "HS"
                          ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/30"
                          : "bg-sky-500/15 text-sky-400 border-sky-500/30"
                        }`}>
                        {r.quota_applied}
                      </span>
                      <div className="text-zinc-600">{r.category}</div>
                    </div>
                  </td>

                  {/* Projected cutoff */}
                  <td className="px-5 py-4 text-right">
                    <span className="text-zinc-300 tabular-nums font-medium">
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
          <p className="text-center text-zinc-600 py-10 text-sm">
            No results match your filters.
          </p>
        ) : (
          filtered.map((r, i) => (
            <motion.button
              key={`${sortDir}-${r.id}`}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: Math.min(i * 0.025, 0.4) }}
              onClick={() => onRowClick(r)}
              className="w-full text-left p-4 rounded-xl border border-zinc-800
                bg-zinc-900/50 hover:bg-zinc-900 transition-all hover:border-zinc-700"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-start gap-2.5 min-w-0">
                  <span className={`shrink-0 text-[10px] font-bold px-1.5 py-0.5 rounded border mt-0.5 ${TYPE_BADGE[r.institute_type]}`}>
                    {r.institute_type}
                  </span>
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-zinc-100 truncate">{r.institute_name}</p>
                    <p className="text-xs text-zinc-500 mt-0.5 truncate">{r.program_name}</p>
                  </div>
                </div>
                <ProbabilityBadge probability={r.probability_percent} showMeter={false} />
              </div>
              <div className="mt-3 flex items-center gap-2 text-xs text-zinc-500">
                <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-bold border
                  ${r.quota_applied === "HS"
                    ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/30"
                    : "bg-sky-500/15 text-sky-400 border-sky-500/30"
                  }`}>
                  {r.quota_applied}
                </span>
                <span>·</span>
                <span>{r.category}</span>
                <span>·</span>
                <span>Proj. {r.projected_closing_rank.toLocaleString()}</span>
              </div>
            </motion.button>
          ))
        )}
      </div>

      {filtered.length > 0 && (
        <p className="text-center text-xs text-zinc-600 pt-1">
          Showing {filtered.length} of {results.length} options ·{" "}
          {sortDir === "desc" ? "Highest probability first" : "Lowest probability first"}
        </p>
      )}
    </div>
  );
}
