"use client";

import { useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, TrendingUp, TrendingDown } from "lucide-react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";
import type { PredictionResult } from "@/lib/types";
import { getRankTypeLabel } from "@/lib/types";

interface VolatilityDrawerProps {
  result: PredictionResult | null;
  onClose: () => void;
}

// Custom tooltip for the recharts chart
const CustomTooltip = ({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: { value: number; name: string; color: string }[];
  label?: string;
}) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-neutral-900 border border-neutral-800 rounded-md px-4 py-3 shadow-xl text-sm">
      <p className="text-neutral-400 text-xs mb-1.5">{label}</p>
      {payload.map((p) => (
        <div key={p.name} className="flex items-center gap-2">
          <span
            className="w-2 h-2 rounded-full"
            style={{ background: p.color }}
          />
          <span className="text-neutral-300">{p.name}:</span>
          <span className="text-white font-semibold">
            {p.value.toLocaleString()}
          </span>
        </div>
      ))}
    </div>
  );
};

export default function VolatilityDrawer({
  result,
  onClose,
}: VolatilityDrawerProps) {
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const history = result?.historical_data ?? [];
  const trend =
    history.length >= 2
      ? history[history.length - 1].closing_rank - history[0].closing_rank
      : 0;

  useEffect(() => {
    if (!result) return;
    const previouslyFocused = document.activeElement as HTMLElement | null;
    const focusTimer = window.setTimeout(() => closeButtonRef.current?.focus(), 0);
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleEscape);
    return () => {
      window.clearTimeout(focusTimer);
      document.removeEventListener("keydown", handleEscape);
      previouslyFocused?.focus();
    };
  }, [onClose, result]);

  return (
    <AnimatePresence>
      {result && (
        <>
          {/* Backdrop */}
          <motion.div
            key="backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            aria-hidden="true"
            className="fixed inset-0 bg-black/60 z-40"
          />

          {/* Drawer */}
          <motion.aside
            key="drawer"
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 28, stiffness: 280 }}
            className="fixed right-0 top-0 h-full w-full sm:w-[520px] bg-neutral-950 border-l border-neutral-800 z-50 overflow-y-auto"
            role="dialog"
            aria-modal="true"
            aria-labelledby="programme-detail-title"
          >
            {/* Header */}
            <div className="flex items-start justify-between p-6 border-b border-neutral-800">
              <div className="space-y-1 pr-4">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold px-2 py-0.5 rounded-sm bg-neutral-900 text-neutral-300 border border-neutral-800">
                    {result.institute_type}
                  </span>
                  <span
                    className={`text-xs font-bold px-2 py-0.5 rounded-sm border ${
                      result.quota_applied === "HS"
                        ? "bg-emerald-500/10 text-emerald-300 border-emerald-500/25"
                        : "bg-cyan-500/10 text-cyan-300 border-cyan-500/25"
                    }`}
                  >
                    {result.quota_applied} Quota
                  </span>
                  <span className="text-xs text-neutral-500">{result.category}</span>
                </div>
                <h2 id="programme-detail-title" className="text-lg font-semibold text-white leading-tight">
                  {result.institute_name}
                </h2>
                <p className="text-sm text-neutral-400">{result.program_name}</p>
              </div>
              <button
                ref={closeButtonRef}
                type="button"
                onClick={onClose}
                aria-label="Close programme details"
                className="shrink-0 p-2 rounded-md hover:bg-neutral-900 text-neutral-500 hover:text-white transition-colors"
              >
                <X size={18} />
              </button>
            </div>

            <dl className="mx-6 mt-5 grid grid-cols-2 gap-3 rounded-md border border-neutral-800 bg-neutral-900/45 p-4 text-xs">
              <div>
                <dt className="text-neutral-500">Rank used</dt>
                <dd className="mt-1 font-medium text-neutral-200">
                  {getRankTypeLabel(result)} · {result.rank_used?.toLocaleString() ?? "not reported"}
                </dd>
              </div>
              <div>
                <dt className="text-neutral-500">Recommendation score</dt>
                <dd className="mt-1 font-medium text-neutral-200">
                  {typeof result.recommendation_score === "number"
                    ? `${result.recommendation_score.toFixed(1)} / 100 (heuristic)`
                    : "Not reported"}
                </dd>
              </div>
              <div className="col-span-2">
                <dt className="text-neutral-500">90% prediction interval</dt>
                <dd className="mt-1 font-medium tabular-nums text-neutral-200">
                  {typeof result.uncertainty_lower === "number" &&
                  typeof result.uncertainty_upper === "number"
                    ? `${result.uncertainty_lower.toLocaleString()}–${result.uncertainty_upper.toLocaleString()}`
                    : "Not reported by service"}
                </dd>
              </div>
            </dl>

            {history.length >= 2 ? (
              <div className="mx-6 mt-4 flex items-center gap-3 rounded-md border border-neutral-800 bg-neutral-900/45 p-4">
                {trend > 0 ? (
                  <TrendingDown className="shrink-0 text-emerald-400" size={22} />
                ) : (
                  <TrendingUp className="shrink-0 text-red-400" size={22} />
                )}
                <div>
                  <p className="text-sm font-medium text-white">
                    {trend > 0 ? "Historical closing rank eased" : "Historical closing rank tightened"}
                  </p>
                  <p className="mt-0.5 text-xs text-neutral-500">
                    Descriptive movement across {history.length} observations, not a causal trend.
                  </p>
                </div>
              </div>
            ) : (
              <p className="mx-6 mt-4 rounded-md border border-neutral-800 bg-neutral-900/45 p-4 text-xs text-neutral-500">
                The service did not return enough historical observations to calculate a trend.
              </p>
            )}

            {/* Chart */}
            <div className="p-6">
              <h3 className="text-sm font-semibold text-neutral-300 mb-4">
                Historical Closing & Opening Rank
              </h3>
              {history.length > 0 ? <div className="h-52">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart
                    data={history}
                    margin={{ top: 4, right: 4, left: 0, bottom: 0 }}
                  >
                    <defs>
                      <linearGradient
                        id="closingGrad"
                        x1="0"
                        y1="0"
                        x2="0"
                        y2="1"
                      >
                        <stop
                          offset="5%"
                          stopColor="#06b6d4"
                          stopOpacity={0.3}
                        />
                        <stop
                          offset="95%"
                          stopColor="#06b6d4"
                          stopOpacity={0}
                        />
                      </linearGradient>
                      <linearGradient
                        id="openingGrad"
                        x1="0"
                        y1="0"
                        x2="0"
                        y2="1"
                      >
                        <stop
                          offset="5%"
                          stopColor="#22c55e"
                          stopOpacity={0.2}
                        />
                        <stop
                          offset="95%"
                          stopColor="#22c55e"
                          stopOpacity={0}
                        />
                      </linearGradient>
                    </defs>
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke="#27272a"
                      vertical={false}
                    />
                    <XAxis
                      dataKey="year"
                      tick={{ fill: "#71717a", fontSize: 11 }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <YAxis
                      reversed
                      tick={{ fill: "#71717a", fontSize: 11 }}
                      axisLine={false}
                      tickLine={false}
                      tickFormatter={(v) =>
                        v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v
                      }
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Area
                      type="monotone"
                      dataKey="closing_rank"
                      name="Closing Rank"
                      stroke="#06b6d4"
                      strokeWidth={2}
                      fill="url(#closingGrad)"
                      dot={{ r: 3, fill: "#06b6d4", strokeWidth: 0 }}
                      activeDot={{ r: 5 }}
                    />
                    <Area
                      type="monotone"
                      dataKey="opening_rank"
                      name="Opening Rank"
                      stroke="#22c55e"
                      strokeWidth={1.5}
                      fill="url(#openingGrad)"
                      strokeDasharray="4 2"
                      dot={{ r: 2.5, fill: "#22c55e", strokeWidth: 0 }}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div> : (
                <div className="flex h-32 items-center justify-center rounded-md border border-neutral-800 text-xs text-neutral-500">
                  No historical series returned.
                </div>
              )}
              {history.length > 0 && <div className="flex items-center gap-5 mt-3">
                <span className="flex items-center gap-1.5 text-xs text-neutral-400">
                  <span className="w-3 h-0.5 bg-cyan-500 inline-block rounded" />
                  Closing Rank
                </span>
                <span className="flex items-center gap-1.5 text-xs text-neutral-400">
                  <span className="w-3 h-0.5 bg-emerald-500 inline-block rounded border-dashed" />
                  Opening Rank
                </span>
              </div>}
            </div>

            {/* Raw data table */}
            <div className="px-6 pb-8">
              <h3 className="text-sm font-semibold text-neutral-300 mb-3">
                Historical data
              </h3>
              <div className="rounded-md border border-neutral-800 overflow-hidden">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-neutral-800 bg-neutral-900/60">
                      <th className="text-left px-4 py-2.5 text-xs text-neutral-500 font-medium">
                        Year
                      </th>
                      <th className="text-right px-4 py-2.5 text-xs text-neutral-500 font-medium">
                        Opening
                      </th>
                      <th className="text-right px-4 py-2.5 text-xs text-neutral-500 font-medium">
                        Closing
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.length === 0 ? (
                      <tr>
                        <td colSpan={3} className="px-4 py-6 text-center text-xs text-neutral-500">
                          No historical data returned.
                        </td>
                      </tr>
                    ) : history.map((d) => (
                      <tr
                        key={d.year}
                        className="border-b border-neutral-800/70 last:border-0 hover:bg-neutral-900/60 transition-colors"
                      >
                        <td className="px-4 py-2.5 text-neutral-300 font-medium">
                          {d.year}
                        </td>
                        <td className="px-4 py-2.5 text-neutral-400 text-right tabular-nums">
                          {d.opening_rank.toLocaleString()}
                        </td>
                        <td className="px-4 py-2.5 text-white text-right tabular-nums font-medium">
                          {d.closing_rank.toLocaleString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
