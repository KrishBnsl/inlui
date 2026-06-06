"use client";

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
    <div className="bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-3 shadow-xl text-sm">
      <p className="text-zinc-400 text-xs mb-1.5">{label}</p>
      {payload.map((p) => (
        <div key={p.name} className="flex items-center gap-2">
          <span
            className="w-2 h-2 rounded-full"
            style={{ background: p.color }}
          />
          <span className="text-zinc-300">{p.name}:</span>
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
  const trend = result
    ? result.historical_data[result.historical_data.length - 1].closing_rank -
      result.historical_data[0].closing_rank
    : 0;

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
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40"
          />

          {/* Drawer */}
          <motion.aside
            key="drawer"
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 28, stiffness: 280 }}
            className="fixed right-0 top-0 h-full w-full sm:w-[520px] bg-zinc-950 border-l border-zinc-800/80 z-50 overflow-y-auto"
          >
            {/* Header */}
            <div className="flex items-start justify-between p-6 border-b border-zinc-800/60">
              <div className="space-y-1 pr-4">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold px-2 py-0.5 rounded bg-zinc-800 text-zinc-300">
                    {result.institute_type}
                  </span>
                  <span
                    className={`text-xs font-bold px-2 py-0.5 rounded border ${
                      result.quota_applied === "HS"
                        ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/30"
                        : "bg-sky-500/15 text-sky-400 border-sky-500/30"
                    }`}
                  >
                    {result.quota_applied} Quota
                  </span>
                  <span className="text-xs text-zinc-500">{result.category}</span>
                </div>
                <h2 className="text-lg font-semibold text-white leading-tight">
                  {result.institute_name}
                </h2>
                <p className="text-sm text-zinc-400">{result.program_name}</p>
              </div>
              <button
                onClick={onClose}
                className="shrink-0 p-2 rounded-lg hover:bg-zinc-800 text-zinc-500 hover:text-white transition-colors"
              >
                <X size={18} />
              </button>
            </div>

            {/* Trend indicator */}
            <div className="mx-6 mt-5 p-4 rounded-xl border border-zinc-800 bg-zinc-900/50 flex items-center gap-3">
              {trend > 0 ? (
                <TrendingDown className="text-emerald-400 shrink-0" size={22} />
              ) : (
                <TrendingUp className="text-red-400 shrink-0" size={22} />
              )}
              <div>
                <p className="text-sm font-medium text-white">
                  {trend > 0
                    ? `Rank easing by ~${Math.abs(Math.round(trend / result.historical_data.length))} per year`
                    : `Cutoff tightening by ~${Math.abs(Math.round(trend / result.historical_data.length))} per year`}
                </p>
                <p className="text-xs text-zinc-500 mt-0.5">
                  Based on {result.historical_data.length}-year closing rank
                  history
                </p>
              </div>
            </div>

            {/* Chart */}
            <div className="p-6">
              <h3 className="text-sm font-semibold text-zinc-300 mb-4">
                Historical Closing & Opening Rank
              </h3>
              <div className="h-52">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart
                    data={result.historical_data}
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
                          stopColor="#6366f1"
                          stopOpacity={0.3}
                        />
                        <stop
                          offset="95%"
                          stopColor="#6366f1"
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
                      stroke="#6366f1"
                      strokeWidth={2}
                      fill="url(#closingGrad)"
                      dot={{ r: 3, fill: "#6366f1", strokeWidth: 0 }}
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
              </div>
              <div className="flex items-center gap-5 mt-3">
                <span className="flex items-center gap-1.5 text-xs text-zinc-400">
                  <span className="w-3 h-0.5 bg-indigo-500 inline-block rounded" />
                  Closing Rank
                </span>
                <span className="flex items-center gap-1.5 text-xs text-zinc-400">
                  <span className="w-3 h-0.5 bg-emerald-500 inline-block rounded border-dashed" />
                  Opening Rank
                </span>
              </div>
            </div>

            {/* Raw data table */}
            <div className="px-6 pb-8">
              <h3 className="text-sm font-semibold text-zinc-300 mb-3">
                Raw Historical Data
              </h3>
              <div className="rounded-xl border border-zinc-800 overflow-hidden">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-zinc-800 bg-zinc-900/60">
                      <th className="text-left px-4 py-2.5 text-xs text-zinc-500 font-medium">
                        Year
                      </th>
                      <th className="text-right px-4 py-2.5 text-xs text-zinc-500 font-medium">
                        Opening
                      </th>
                      <th className="text-right px-4 py-2.5 text-xs text-zinc-500 font-medium">
                        Closing
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.historical_data.map((d) => (
                      <tr
                        key={d.year}
                        className="border-b border-zinc-800/60 last:border-0 hover:bg-zinc-900/40 transition-colors"
                      >
                        <td className="px-4 py-2.5 text-zinc-300 font-medium">
                          {d.year}
                        </td>
                        <td className="px-4 py-2.5 text-zinc-400 text-right tabular-nums">
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
