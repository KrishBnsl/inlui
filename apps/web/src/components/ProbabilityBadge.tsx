"use client";

import { getProbabilityTier, type ProbabilityTier } from "@/lib/types";

interface ProbabilityBadgeProps {
  probability: number;
  tier?: ProbabilityTier;
  showMeter?: boolean;
}

const tierConfig: Record<
  ProbabilityTier,
  { bar: string; pill: string; label: string }
> = {
  safe: {
    bar: "bg-emerald-500",
    pill: "bg-emerald-500/10 text-emerald-300 border-emerald-500/25",
    label: "Safe",
  },
  target: {
    bar: "bg-amber-500",
    pill: "bg-amber-500/10 text-amber-300 border-amber-500/25",
    label: "Target",
  },
  reach: {
    bar: "bg-red-500/80",
    pill: "bg-red-500/10 text-red-300 border-red-500/25",
    label: "Reach",
  },
};

export default function ProbabilityBadge({
  probability,
  tier,
  showMeter = true,
}: ProbabilityBadgeProps) {
  const resolvedTier = tier ?? getProbabilityTier(probability);
  const config = tierConfig[resolvedTier];

  return (
    <div className="flex flex-col items-end gap-1.5 min-w-[116px]">
      {/* Pill badge */}
      <div
        aria-label={`${probability.toFixed(1)} percent uncalibrated admission-likelihood estimate, ${config.label} recommendation bucket`}
        className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md border text-xs font-semibold ${config.pill}`}
      >
        <span>{probability.toFixed(1)}%</span>
        <span className="opacity-60">·</span>
        <span>{config.label}</span>
      </div>

      {/* Thin progress meter */}
      {showMeter && (
        <div className="w-full h-1 bg-neutral-800 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-700 ease-out ${config.bar}`}
            style={{ width: `${probability}%` }}
          />
        </div>
      )}
    </div>
  );
}
