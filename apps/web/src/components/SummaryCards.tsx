"use client";

import { motion, type Variants } from "framer-motion";
import { CheckCircle2, ListChecks, TrendingUp } from "lucide-react";
import type { PredictionResult } from "@/lib/types";
import { formatChoiceName, getTopBucketChoice } from "@/lib/types";

interface SummaryCardsProps {
  total_options: number;
  safest_choice: string;
  top_upgrade: string;
  results: PredictionResult[];
  candidateCounts?: Record<string, number>;
  returnedBucketCounts?: Record<string, number>;
}

const cardVariants: Variants = {
  hidden: { opacity: 0, y: 16 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.1, duration: 0.4, ease: "easeOut" },
  }),
};

export default function SummaryCards({
  total_options,
  safest_choice,
  results,
  candidateCounts,
  returnedBucketCounts,
}: SummaryCardsProps) {
  const safestBucketChoice = getTopBucketChoice(results, "safe_backup");
  const reachBucketChoice = getTopBucketChoice(results, "ambitious_reach");
  const safestValue = formatChoiceName(safestBucketChoice, safest_choice);
  const reachValue = formatChoiceName(reachBucketChoice, "No realistic reach found");
  const eligibleBeforeLimit = candidateCounts?.eligible_after_reach_filter;
  const safeReturned = returnedBucketCounts?.safe_backup;
  const reachReturned = returnedBucketCounts?.ambitious_reach;

  const cards = [
    {
      icon: <ListChecks size={18} className="text-neutral-400" />,
      label: "Returned options",
      value: total_options.toString(),
      sub:
        eligibleBeforeLimit === undefined
          ? "After eligibility and result limits"
          : `${eligibleBeforeLimit} eligible before top-N trimming`,
      accent: "border-neutral-800 bg-neutral-900/30",
    },
    {
      icon: <CheckCircle2 size={18} className="text-emerald-400" />,
      label: "Safest choice",
      value: safestValue,
      sub: `Top safe-backup choice · ${safeReturned ?? "?"} returned`,
      accent: "border-neutral-800 bg-neutral-900/30",
    },
    {
      icon: <TrendingUp size={18} className="text-amber-400" />,
      label: "Best reach option",
      value: reachValue,
      sub: `Top ambitious-reach choice · ${reachReturned ?? "?"} returned`,
      accent: "border-neutral-800 bg-neutral-900/30",
    },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
      {cards.map((card, i) => (
        <motion.div
          key={card.label}
          custom={i}
          initial="hidden"
          animate="visible"
          variants={cardVariants}
          className={`rounded-md border p-4 ${card.accent}`}
        >
          <div className="flex items-center gap-2.5 mb-3">
            <div className="p-1.5 rounded-md bg-neutral-950 border border-neutral-800">{card.icon}</div>
            <span className="text-xs font-medium text-neutral-400 uppercase">
              {card.label}
            </span>
          </div>
          <p className="text-xl font-semibold text-white leading-tight truncate">
            {card.value}
          </p>
          <p className="text-xs text-neutral-500 mt-1.5">{card.sub}</p>
        </motion.div>
      ))}
    </div>
  );
}
