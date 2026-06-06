"use client";

import { motion, type Variants } from "framer-motion";
import { CheckCircle2, ListChecks, TrendingUp } from "lucide-react";

interface SummaryCardsProps {
  total_options: number;
  safest_choice: string;
  top_upgrade: string;
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
  top_upgrade,
}: SummaryCardsProps) {
  const cards = [
    {
      icon: <ListChecks size={18} className="text-neutral-400" />,
      label: "Eligible options",
      value: total_options.toString(),
      sub: "After category, state, and gender filters",
      accent: "border-neutral-800 bg-neutral-900/30",
    },
    {
      icon: <CheckCircle2 size={18} className="text-emerald-400" />,
      label: "Safest choice",
      value: safest_choice,
      sub: "Highest probability in the current result set",
      accent: "border-neutral-800 bg-neutral-900/30",
    },
    {
      icon: <TrendingUp size={18} className="text-amber-400" />,
      label: "Best reach option",
      value: top_upgrade,
      sub: "Ambitious option still worth tracking",
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
