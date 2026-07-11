"use client";

import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Bookmark, BookmarkCheck, Search } from "lucide-react";
import type { BucketFilter, PredictionResult } from "@/lib/types";
import {
  filterByBucket,
  getRankTypeLabel,
  getTierForResult,
} from "@/lib/types";
import {
  RESULT_SORT_LABELS,
  sortResults,
  type ResultSortMode,
} from "@/lib/result-sorting";
import ProbabilityBadge from "./ProbabilityBadge";

interface ResultsTableProps {
  results: PredictionResult[];
  onRowClick: (result: PredictionResult) => void;
  savedIds: Set<string>;
  onToggleSaved: (result: PredictionResult) => void;
}

type FilterType = "ALL" | "IIT" | "NIT" | "IIIT" | "GFTI";

const FILTERS: FilterType[] = ["ALL", "IIT", "NIT", "IIIT", "GFTI"];
const TIER_FILTERS: { value: BucketFilter; label: string }[] = [
  { value: "ALL", label: "All buckets" },
  { value: "safe", label: "Safe" },
  { value: "target", label: "Target" },
  { value: "reach", label: "Reach" },
];

const TYPE_BADGE: Record<string, string> = {
  IIT: "border-neutral-700 bg-neutral-900 text-neutral-200",
  NIT: "border-cyan-500/25 bg-neutral-900 text-cyan-300",
  IIIT: "border-violet-500/25 bg-neutral-900 text-violet-300",
  GFTI: "border-neutral-700 bg-neutral-900 text-neutral-400",
};

function formatInterval(result: PredictionResult): string {
  if (
    typeof result.uncertainty_lower !== "number" ||
    typeof result.uncertainty_upper !== "number"
  ) {
    return "Not reported";
  }
  return `${result.uncertainty_lower.toLocaleString()}–${result.uncertainty_upper.toLocaleString()}`;
}

function formatScore(result: PredictionResult): string {
  return typeof result.recommendation_score === "number"
    ? `${result.recommendation_score.toFixed(1)} / 100`
    : "Not reported";
}

export function TableSkeleton() {
  return (
    <div className="mt-6 space-y-2" role="status" aria-label="Loading recommendations">
      {Array.from({ length: 6 }).map((_, index) => (
        <div
          key={index}
          className="h-16 animate-pulse rounded-md bg-neutral-800/40"
          style={{ opacity: 1 - index * 0.12 }}
        />
      ))}
      <span className="sr-only">Loading recommendations</span>
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
  const [tierFilter, setTierFilter] = useState<BucketFilter>("ALL");
  const [sortMode, setSortMode] = useState<ResultSortMode>("best_fit");

  const filtered = useMemo(() => {
    const query = search.toLowerCase().trim();
    const subset = filterByBucket(results, tierFilter).filter((result) => {
      if (filterType !== "ALL" && result.institute_type !== filterType) return false;
      return (
        !query ||
        result.institute_name.toLowerCase().includes(query) ||
        result.program_name.toLowerCase().includes(query)
      );
    });
    return sortResults(subset, sortMode);
  }, [filterType, results, search, sortMode, tierFilter]);

  return (
    <div className="space-y-4" aria-label="Recommendation results">
      <div className="space-y-3 rounded-md border border-neutral-800 bg-neutral-900/25 p-3">
        <div className="flex flex-col items-start gap-3 xl:flex-row xl:items-center">
          <div className="relative w-full flex-1">
            <label htmlFor="result-search" className="sr-only">
              Search institute or programme
            </label>
            <Search
              size={15}
              className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-neutral-500"
              aria-hidden="true"
            />
            <input
              id="result-search"
              type="search"
              placeholder="Search institute or programme…"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              className="w-full rounded-md border border-neutral-800 bg-neutral-950 py-2.5 pl-9 pr-4 text-sm text-neutral-200 outline-none transition-colors placeholder:text-neutral-600 focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/25"
            />
          </div>

          <div className="flex flex-wrap gap-1.5" role="group" aria-label="Institute type filter">
            {FILTERS.map((value) => (
              <button
                key={value}
                type="button"
                aria-pressed={filterType === value}
                onClick={() => setFilterType(value)}
                className={`rounded-md border px-3 py-1.5 text-xs font-semibold transition-colors ${
                  filterType === value
                    ? "border-cyan-500 bg-cyan-600 text-white"
                    : "border-neutral-800 bg-neutral-950 text-neutral-400 hover:border-neutral-600 hover:text-neutral-200"
                }`}
              >
                {value === "ALL" ? "All institutes" : value}
              </button>
            ))}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <div className="flex flex-wrap gap-2" role="group" aria-label="Recommendation bucket filter">
            {TIER_FILTERS.map((filter) => (
              <button
                key={filter.value}
                type="button"
                data-testid={`bucket-filter-${filter.value.toLowerCase()}`}
                aria-pressed={tierFilter === filter.value}
                onClick={() => setTierFilter(filter.value)}
                className={`rounded-md border px-3 py-1.5 text-xs font-medium transition-colors ${
                  tierFilter === filter.value
                    ? "border-neutral-200 bg-neutral-200 text-neutral-950"
                    : "border-neutral-800 bg-neutral-950 text-neutral-400 hover:border-neutral-600 hover:text-neutral-200"
                }`}
              >
                {filter.label}
              </button>
            ))}
          </div>
          <label htmlFor="result-sort" className="ml-auto flex items-center gap-2 text-xs text-neutral-500">
            Sort results
            <select
              id="result-sort"
              value={sortMode}
              onChange={(event) => setSortMode(event.target.value as ResultSortMode)}
              className="rounded-md border border-neutral-800 bg-neutral-950 px-2.5 py-1.5 text-xs text-neutral-200 outline-none transition-colors focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/25"
            >
              {(Object.keys(RESULT_SORT_LABELS) as ResultSortMode[]).map((mode) => (
                <option key={mode} value={mode}>
                  {RESULT_SORT_LABELS[mode]}
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>

      <p className="text-xs leading-relaxed text-neutral-500">
        The uncalibrated admission-likelihood estimate is a normal-approximation output
        for simulated cutoff exceedance, not an individual outcome probability.
        Recommendation score is a separate 0–100 heuristic used only to order choices.
        Safe, target, and reach are recommendation buckets—not guarantees.
      </p>

      <div className="hidden overflow-x-auto rounded-md border border-neutral-800 sm:block">
        <table className="min-w-[980px] w-full text-sm">
          <caption className="sr-only">
            JoSAA recommendations with rank routing, admission-likelihood estimate, score,
            and prediction interval
          </caption>
          <thead>
            <tr className="border-b border-neutral-800 bg-neutral-900/80">
              <th scope="col" className="w-12 px-3 py-3.5">
                <span className="sr-only">Save</span>
              </th>
              <th scope="col" className="w-[30%] px-4 py-3.5 text-left text-xs font-medium text-neutral-500">
                Institute / programme
              </th>
              <th scope="col" className="px-4 py-3.5 text-left text-xs font-medium text-neutral-500">
                Rank used
              </th>
              <th scope="col" className="px-4 py-3.5 text-right text-xs font-medium text-neutral-500">
                Predicted cutoff
              </th>
              <th scope="col" className="px-4 py-3.5 text-right text-xs font-medium text-neutral-500">
                Likelihood estimate
              </th>
              <th scope="col" className="px-4 py-3.5 text-right text-xs font-medium text-neutral-500">
                Recommendation score
              </th>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-5 py-12 text-center text-sm text-neutral-500">
                  No recommendations match these filters.
                </td>
              </tr>
            ) : (
              filtered.map((result, index) => (
                <motion.tr
                  key={`${sortMode}-${result.id}`}
                  data-testid={`result-row-${result.id}`}
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: Math.min(index * 0.015, 0.2), duration: 0.15 }}
                  className="border-b border-neutral-800/70 last:border-0 hover:bg-neutral-900/70"
                >
                  <td className="px-3 py-4 align-top">
                    <button
                      type="button"
                      onClick={() => onToggleSaved(result)}
                      className={`rounded-md border p-1.5 transition-colors ${
                        savedIds.has(result.id)
                          ? "border-cyan-500/40 bg-cyan-500/10 text-cyan-300"
                          : "border-neutral-800 text-neutral-600 hover:border-neutral-600 hover:text-neutral-200"
                      }`}
                      aria-label={`${savedIds.has(result.id) ? "Remove" : "Save"} ${
                        result.institute_name
                      } ${result.program_name} ${savedIds.has(result.id) ? "from" : "to"} choice list`}
                    >
                      {savedIds.has(result.id) ? (
                        <BookmarkCheck size={15} aria-hidden="true" />
                      ) : (
                        <Bookmark size={15} aria-hidden="true" />
                      )}
                    </button>
                  </td>
                  <td className="px-4 py-4">
                    <button
                      type="button"
                      onClick={() => onRowClick(result)}
                      className="flex items-start gap-3 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500"
                      aria-label={`Open details for ${result.institute_name}, ${result.program_name}`}
                    >
                      <span className={`mt-0.5 shrink-0 rounded-sm border px-1.5 py-0.5 text-[10px] font-bold ${TYPE_BADGE[result.institute_type]}`}>
                        {result.institute_type}
                      </span>
                      <span>
                        <span className="block font-medium leading-tight text-neutral-100">
                          {result.institute_name}
                        </span>
                        <span className="mt-0.5 block text-xs text-neutral-500">
                          {result.program_name}
                        </span>
                        <span className="mt-1 block text-[11px] text-neutral-600">
                          {result.quota_applied} quota · {result.category}
                        </span>
                      </span>
                    </button>
                  </td>
                  <td className="px-4 py-4 text-xs">
                    <p className="font-medium text-neutral-300">{getRankTypeLabel(result)}</p>
                    <p className="mt-1 tabular-nums text-neutral-500">
                      {typeof result.rank_used === "number"
                        ? result.rank_used.toLocaleString()
                        : "Rank not reported"}
                    </p>
                  </td>
                  <td className="px-4 py-4 text-right">
                    <p className="font-medium tabular-nums text-neutral-300">
                      {result.projected_closing_rank.toLocaleString()}
                    </p>
                    <p className="mt-1 text-[11px] tabular-nums text-neutral-500">
                      90% prediction interval: {formatInterval(result)}
                    </p>
                  </td>
                  <td className="px-4 py-4">
                    <div className="flex justify-end">
                      <ProbabilityBadge
                        probability={result.probability_percent}
                        tier={getTierForResult(result)}
                      />
                    </div>
                  </td>
                  <td className="px-4 py-4 text-right">
                    <p className="font-medium tabular-nums text-neutral-300">{formatScore(result)}</p>
                    <p className="mt-1 text-[11px] text-neutral-600">Heuristic, not probability</p>
                  </td>
                </motion.tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      <div className="space-y-3 sm:hidden">
        {filtered.length === 0 ? (
          <p className="py-10 text-center text-sm text-neutral-500">
            No recommendations match these filters.
          </p>
        ) : (
          filtered.map((result) => (
            <article
              key={result.id}
              data-testid={`mobile-result-${result.id}`}
              className="rounded-md border border-neutral-800 bg-neutral-900/45 p-4"
            >
              <div className="flex items-start justify-between gap-3">
                <button
                  type="button"
                  onClick={() => onRowClick(result)}
                  className="min-w-0 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500"
                >
                  <span className={`rounded-sm border px-1.5 py-0.5 text-[10px] font-bold ${TYPE_BADGE[result.institute_type]}`}>
                    {result.institute_type}
                  </span>
                  <span className="mt-2 block truncate text-sm font-medium text-neutral-100">
                    {result.institute_name}
                  </span>
                  <span className="mt-0.5 block truncate text-xs text-neutral-500">
                    {result.program_name}
                  </span>
                </button>
                <button
                  type="button"
                  onClick={() => onToggleSaved(result)}
                  aria-label={`${savedIds.has(result.id) ? "Remove" : "Save"} ${result.institute_name} ${result.program_name}`}
                  className="rounded-md border border-neutral-800 p-2 text-neutral-400"
                >
                  {savedIds.has(result.id) ? <BookmarkCheck size={15} /> : <Bookmark size={15} />}
                </button>
              </div>
              <div className="mt-3 flex flex-wrap items-center justify-between gap-3 border-t border-neutral-800 pt-3">
                <div className="text-xs text-neutral-500">
                  <p>{getRankTypeLabel(result)} · {result.rank_used?.toLocaleString() ?? "rank not reported"}</p>
                  <p className="mt-1">90% prediction interval: {formatInterval(result)}</p>
                  <p className="mt-1">Recommendation score: {formatScore(result)}</p>
                </div>
                <ProbabilityBadge
                  probability={result.probability_percent}
                  tier={getTierForResult(result)}
                  showMeter={false}
                />
              </div>
            </article>
          ))
        )}
      </div>

      <p className="text-center text-xs text-neutral-600" aria-live="polite">
        Showing {filtered.length} of {results.length} returned recommendations · {RESULT_SORT_LABELS[sortMode]}
      </p>
    </div>
  );
}
