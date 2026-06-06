"use client";

import { getProbabilityTier, type ProbabilityTier } from "@/lib/types";

interface ProbabilityBadgeProps {
  probability: number;
  showMeter?: boolean;
}

const tierConfig: Record<
  ProbabilityTier,
  { bar: string; pill: string; label: string }
> = {
  safe: {
    bar: "bg-emerald-500",
    pill: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
    label: "Safe",
  },
  target: {
    bar: "bg-amber-500",
    pill: "bg-amber-500/15 text-amber-400 border-amber-500/30",
    label: "Target",
  },
  reach: {
    bar: "bg-red-500/80",
    pill: "bg-red-500/10 text-red-400 border-red-500/30",
    label: "Reach",
  },
};

export default function ProbabilityBadge({
  probability,
  showMeter = true,
}: ProbabilityBadgeProps) {
  const tier = getProbabilityTier(probability);
  const config = tierConfig[tier];

  return (
    <div className="flex flex-col items-end gap-1.5 min-w-[120px]">
      {/* Pill badge */}
      <div
        className={`flex items-center gap-1.5 px-3 py-1 rounded-full border text-xs font-semibold ${config.pill}`}
      >
        <span className={`w-1.5 h-1.5 rounded-full ${config.bar}`} />
        <span>{probability.toFixed(1)}%</span>
        <span className="opacity-60">·</span>
        <span>{config.label}</span>
      </div>

      {/* Thin progress meter */}
      {showMeter && (
        <div className="w-full h-1 bg-zinc-800 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-700 ease-out ${config.bar}`}
            style={{ width: `${probability}%` }}
          />
        </div>
      )}
    </div>
  );
}
