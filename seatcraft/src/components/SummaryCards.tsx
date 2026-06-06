"use client";

import { motion } from "framer-motion";
import { Shield, TrendingUp, LayoutGrid } from "lucide-react";

interface SummaryCardsProps {
  total_options: number;
  safest_choice: string;
  top_upgrade: string;
}

const cardVariants = {
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
      icon: <LayoutGrid size={20} className="text-indigo-400" />,
      label: "Total Eligible Options",
      value: total_options.toString(),
      sub: "Matching your profile across all JoSAA institutes",
      accent: "border-indigo-500/30 bg-indigo-500/5",
    },
    {
      icon: <Shield size={20} className="text-emerald-400" />,
      label: "Safest Top Choice",
      value: safest_choice,
      sub: "Highest allotment probability in your matrix",
      accent: "border-emerald-500/30 bg-emerald-500/5",
    },
    {
      icon: <TrendingUp size={20} className="text-amber-400" />,
      label: "Highest Upgradable Option",
      value: top_upgrade,
      sub: "Best reach college within striking distance",
      accent: "border-amber-500/30 bg-amber-500/5",
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
          className={`rounded-2xl border p-5 ${card.accent}`}
        >
          <div className="flex items-center gap-2.5 mb-3">
            <div className="p-2 rounded-lg bg-zinc-900/60">{card.icon}</div>
            <span className="text-xs font-medium text-zinc-400 uppercase tracking-wider">
              {card.label}
            </span>
          </div>
          <p className="text-xl font-semibold text-white leading-tight truncate">
            {card.value}
          </p>
          <p className="text-xs text-zinc-500 mt-1.5">{card.sub}</p>
        </motion.div>
      ))}
    </div>
  );
}
