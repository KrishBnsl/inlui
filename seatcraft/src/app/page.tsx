"use client";

import { useEffect, useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { BarChart3, RotateCcw, Code2 } from "lucide-react";
import IntakeForm from "@/components/IntakeForm";
import ChoiceList from "@/components/ChoiceList";
import SummaryCards from "@/components/SummaryCards";
import ResultsTable, { TableSkeleton } from "@/components/ResultsTable";
import VolatilityDrawer from "@/components/VolatilityDrawer";
import type { SimulationInput, SimulationResponse, PredictionResult } from "@/lib/types";
import { runSimulation } from "@/lib/api";

type AppState = "idle" | "loading" | "results";

const readStoredChoiceIds = () => {
  if (typeof window === "undefined") return [];
  try {
    const stored = window.localStorage.getItem("seatcraft.choiceIds");
    return stored ? (JSON.parse(stored) as string[]) : [];
  } catch {
    return [];
  }
};

const readStoredChoiceNote = () => {
  if (typeof window === "undefined") return "";
  return window.localStorage.getItem("seatcraft.choiceNote") ?? "";
};

export default function Home() {
  const [appState, setAppState] = useState<AppState>("idle");
  const [response, setResponse] = useState<SimulationResponse | null>(null);
  const [selectedResult, setSelectedResult] = useState<PredictionResult | null>(null);
  const [savedChoiceIds, setSavedChoiceIds] = useState<string[]>(readStoredChoiceIds);
  const [choiceNote, setChoiceNote] = useState(readStoredChoiceNote);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    window.localStorage.setItem("seatcraft.choiceIds", JSON.stringify(savedChoiceIds));
  }, [savedChoiceIds]);

  useEffect(() => {
    window.localStorage.setItem("seatcraft.choiceNote", choiceNote);
  }, [choiceNote]);

  const savedIdSet = useMemo(() => new Set(savedChoiceIds), [savedChoiceIds]);
  const savedChoices = useMemo(() => {
    if (!response) return [];
    return savedChoiceIds
      .map((id) => response.results.find((result) => result.id === id))
      .filter((result): result is PredictionResult => Boolean(result));
  }, [response, savedChoiceIds]);

  const handleSubmit = async (input: SimulationInput) => {
    setError(null);
    setAppState("loading");
    try {
      // Live backend via Rust sim_engine server
      const data = await runSimulation(input);
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

  const handleToggleSaved = (result: PredictionResult) => {
    setSavedChoiceIds((ids) =>
      ids.includes(result.id)
        ? ids.filter((id) => id !== result.id)
        : [...ids, result.id]
    );
  };

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100">
      {/* Navbar */}
      <nav className="sticky top-0 z-30 border-b border-neutral-800 bg-neutral-950/95 backdrop-blur">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="p-1.5 rounded-md bg-neutral-900 border border-neutral-800">
              <BarChart3 size={16} className="text-cyan-400" />
            </div>
            <span className="font-semibold text-white tracking-tight">SeatCraft</span>
            <span className="hidden sm:block text-neutral-500 text-xs ml-1">JoSAA choice planner</span>
          </div>
          <div className="flex items-center gap-3">
            {appState === "results" && (
              <motion.button
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                onClick={handleReset}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium
                  border border-neutral-800 bg-neutral-900 text-neutral-400 hover:text-white
                  hover:border-neutral-600 transition-colors"
              >
                <RotateCcw size={12} />
                New search
              </motion.button>
            )}
            <a
              href="https://github.com"
              target="_blank"
              rel="noreferrer"
              className="text-neutral-600 hover:text-neutral-300 transition-colors"
            >
              <Code2 size={18} />
            </a>
          </div>
        </div>
      </nav>

      {/* Main */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8 sm:py-10">
        <AnimatePresence mode="wait">
          {/* ─── IDLE: Hero + Intake Form ─────────────────────────────────── */}
          {appState === "idle" && (
            <motion.div
              key="idle"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0, y: -16 }}
              className="grid lg:grid-cols-[minmax(0,0.85fr)_minmax(520px,1fr)] gap-10 lg:gap-14 items-start"
            >
              {/* Hero */}
              <div className="pt-4 lg:pt-10 space-y-7">
                <div className="space-y-4 max-w-xl">
                  <p className="text-xs font-medium uppercase text-cyan-400">
                    Admission planning
                  </p>
                  <h1 className="text-3xl sm:text-5xl font-semibold tracking-tight text-white leading-tight">
                    Build a JoSAA list with probabilities you can compare.
                  </h1>
                  <p className="text-neutral-400 text-base leading-relaxed">
                    Enter your ranks and reservation details to see eligible options,
                    projected closing ranks, and where each choice sits on the safe,
                    target, or reach spectrum.
                  </p>
                </div>

                <div className="grid grid-cols-3 gap-3 max-w-xl border-y border-neutral-800 py-4">
                  <div>
                    <p className="text-lg font-semibold text-white tabular-nums">10k</p>
                    <p className="text-xs text-neutral-500">simulations</p>
                  </div>
                  <div>
                    <p className="text-lg font-semibold text-white tabular-nums">8 yrs</p>
                    <p className="text-xs text-neutral-500">cutoff history</p>
                  </div>
                  <div>
                    <p className="text-lg font-semibold text-white">HS/AI</p>
                    <p className="text-xs text-neutral-500">quota aware</p>
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: -8 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="w-full px-4 py-3 rounded-md border border-red-500/30 bg-red-500/10 text-red-400 text-sm"
                  >
                    {error}
                  </motion.div>
                )}

                <IntakeForm onSubmit={handleSubmit} isLoading={false} />
              </div>
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
                <div className="inline-flex items-center gap-2 text-cyan-300 text-sm">
                  <span className="inline-block w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
                  Running simulation
                </div>
                <p className="text-neutral-500 text-xs">
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
              className="space-y-6"
            >
              {/* Header */}
              <div className="flex items-end justify-between gap-4 border-b border-neutral-800 pb-5">
                <div>
                  <h2 className="text-2xl font-semibold text-white">
                    Allotment probabilities
                  </h2>
                  <p className="text-neutral-500 text-sm mt-1">
                    Click a row to inspect historical cutoff movement.
                  </p>
                </div>
              </div>

              <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
                <section className="space-y-6">
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
                    savedIds={savedIdSet}
                    onToggleSaved={handleToggleSaved}
                  />
                </section>

                <ChoiceList
                  items={savedChoices}
                  note={choiceNote}
                  onNoteChange={setChoiceNote}
                  onRemove={(id) =>
                    setSavedChoiceIds((ids) => ids.filter((choiceId) => choiceId !== id))
                  }
                  onClear={() => setSavedChoiceIds([])}
                />
              </div>
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
