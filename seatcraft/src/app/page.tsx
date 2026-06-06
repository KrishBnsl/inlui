"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Cpu, RotateCcw, Code2 } from "lucide-react";
import IntakeForm from "@/components/IntakeForm";
import SummaryCards from "@/components/SummaryCards";
import ResultsTable, { TableSkeleton } from "@/components/ResultsTable";
import VolatilityDrawer from "@/components/VolatilityDrawer";
import type { SimulationInput, SimulationResponse, PredictionResult } from "@/lib/types";
import { runSimulationMock } from "@/lib/api";

type AppState = "idle" | "loading" | "results";

export default function Home() {
  const [appState, setAppState] = useState<AppState>("idle");
  const [response, setResponse] = useState<SimulationResponse | null>(null);
  const [selectedResult, setSelectedResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (input: SimulationInput) => {
    setError(null);
    setAppState("loading");
    try {
      // Swap runSimulationMock → runSimulation when backend is ready
      const data = await runSimulationMock(input);
      setResponse(data);
      setAppState("results");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Simulation failed. Please try again.");
      setAppState("idle");
    }
  };

  const handleReset = () => {
    setAppState("idle");
    setResponse(null);
    setSelectedResult(null);
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-white">
      {/* Ambient background glow */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -left-40 w-96 h-96 bg-indigo-600/8 rounded-full blur-3xl" />
        <div className="absolute top-1/2 -right-40 w-80 h-80 bg-violet-600/6 rounded-full blur-3xl" />
      </div>

      {/* Navbar */}
      <nav className="sticky top-0 z-30 border-b border-zinc-800/60 bg-zinc-950/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="p-1.5 rounded-lg bg-indigo-600/20 border border-indigo-500/30">
              <Cpu size={16} className="text-indigo-400" />
            </div>
            <span className="font-semibold text-white tracking-tight">SeatCraft</span>
            <span className="hidden sm:block text-zinc-600 text-xs ml-1">JoSAA Probability Engine</span>
          </div>
          <div className="flex items-center gap-3">
            {appState === "results" && (
              <motion.button
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                onClick={handleReset}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium
                  border border-zinc-800 bg-zinc-900 text-zinc-400 hover:text-white
                  hover:border-zinc-600 transition-all"
              >
                <RotateCcw size={12} />
                New Simulation
              </motion.button>
            )}
            <a
              href="https://github.com"
              target="_blank"
              rel="noreferrer"
              className="text-zinc-600 hover:text-zinc-300 transition-colors"
            >
              <Code2 size={18} />
            </a>
          </div>
        </div>
      </nav>

      {/* Main */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-10 sm:py-16 relative">
        <AnimatePresence mode="wait">
          {/* ─── IDLE: Hero + Intake Form ─────────────────────────────────── */}
          {appState === "idle" && (
            <motion.div
              key="idle"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0, y: -16 }}
              className="flex flex-col items-center"
            >
              {/* Hero */}
              <div className="text-center mb-12 space-y-4 max-w-xl">
                <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-indigo-500/30 bg-indigo-500/10 text-indigo-300 text-xs font-medium">
                  <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-pulse" />
                  Monte Carlo · 10,000 iterations · Percentile-normalized
                </div>
                <h1 className="text-4xl sm:text-5xl font-bold tracking-tight">
                  Know your{" "}
                  <span className="bg-gradient-to-r from-indigo-400 to-violet-400 bg-clip-text text-transparent">
                    true odds
                  </span>
                  <br />
                  before counselling day.
                </h1>
                <p className="text-zinc-400 text-base leading-relaxed">
                  SeatCraft simulates 10,000 JoSAA admission scenarios using 8 years
                  of historical cutoff data, giving you a statistically rigorous
                  probability estimate for every eligible seat.
                </p>
              </div>

              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -8 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="w-full max-w-2xl mb-6 px-4 py-3 rounded-xl border border-red-500/30 bg-red-500/10 text-red-400 text-sm"
                >
                  {error}
                </motion.div>
              )}

              <IntakeForm onSubmit={handleSubmit} isLoading={false} />
            </motion.div>
          )}

          {/* ─── LOADING: Skeleton while simulation runs ───────────────────── */}
          {appState === "loading" && (
            <motion.div
              key="loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="max-w-2xl mx-auto"
            >
              <div className="text-center mb-10 space-y-3">
                <div className="inline-flex items-center gap-2 text-indigo-300 text-sm">
                  <span className="inline-block w-2 h-2 rounded-full bg-indigo-400 animate-ping" />
                  Running Monte Carlo Simulation…
                </div>
                <p className="text-zinc-500 text-xs">
                  Sampling from 8 years of closing rank distributions
                </p>
              </div>
              <IntakeForm onSubmit={handleSubmit} isLoading={true} />
              <TableSkeleton />
            </motion.div>
          )}

          {/* ─── RESULTS: Dashboard ───────────────────────────────────────── */}
          {appState === "results" && response && (
            <motion.div
              key="results"
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="space-y-8"
            >
              {/* Header */}
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-2xl font-bold text-white">
                    Allotment Probability Matrix
                  </h2>
                  <p className="text-zinc-500 text-sm mt-1">
                    Simulation complete · Click any row to view historical trend
                  </p>
                </div>
              </div>

              {/* Summary Cards */}
              <SummaryCards
                total_options={response.total_options}
                safest_choice={response.safest_choice}
                top_upgrade={response.top_upgrade}
              />

              {/* Results Table */}
              <ResultsTable
                results={response.results}
                onRowClick={setSelectedResult}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Volatility Drawer */}
      <VolatilityDrawer
        result={selectedResult}
        onClose={() => setSelectedResult(null)}
      />
    </div>
  );
}
