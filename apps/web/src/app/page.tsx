"use client";

import { useEffect, useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { BarChart3, RotateCcw, Code2, MessageCircle } from "lucide-react";
import IntakeForm from "@/components/IntakeForm";
import ChoiceList from "@/components/ChoiceList";
import SummaryCards from "@/components/SummaryCards";
import ResultsTable, { TableSkeleton } from "@/components/ResultsTable";
import VolatilityDrawer from "@/components/VolatilityDrawer";
import ChatAdvisor from "@/components/ChatAdvisor";
import ResearchTransparency from "@/components/ResearchTransparency";
import type { SimulationInput, SimulationResponse, PredictionResult } from "@/lib/types";
import { getMLHealth, runMLSimulation } from "@/lib/api";

type AppState = "idle" | "loading" | "results";

const readStoredChoiceIds = () => {
  if (typeof window === "undefined") return [];
  try {
    const stored = window.localStorage.getItem("seatcraft.choiceIds");
    if (!stored) return [];
    const parsed: unknown = JSON.parse(stored);
    if (!Array.isArray(parsed)) return [];
    return [...new Set(parsed.filter((value): value is string => typeof value === "string"))].slice(
      0,
      200
    );
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
  const [chatOpen, setChatOpen] = useState(false);

  useEffect(() => {
    try {
      window.localStorage.setItem("seatcraft.choiceIds", JSON.stringify(savedChoiceIds));
    } catch {
      // Browsers can block storage; saving remains an optional convenience.
    }
  }, [savedChoiceIds]);

  useEffect(() => {
    try {
      window.localStorage.setItem("seatcraft.choiceNote", choiceNote);
    } catch {
      // Browsers can block storage; saving remains an optional convenience.
    }
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
      const health = await getMLHealth();
      if (!health.artifacts_loaded) {
        const missing = Object.entries(health.artifact_status ?? {})
          .filter(([, status]) => status !== "loaded")
          .map(([name]) => name)
          .join(", ");
        throw new Error(
          missing
            ? `ML service is not ready. Missing artifacts: ${missing}.`
            : "ML service is not ready because model artifacts are unavailable."
        );
      }
      const data = await runMLSimulation(input);
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
    setError(null);
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
            <button
              type="button"
              onClick={() => setChatOpen(true)}
              aria-label="Open counselling advisor"
              className="flex items-center gap-1.5 rounded-md border border-neutral-800 bg-neutral-900 px-3 py-1.5 text-xs font-medium text-neutral-300 transition-colors hover:border-cyan-500/50 hover:text-cyan-200"
            >
              <MessageCircle size={14} aria-hidden="true" />
              <span className="hidden sm:inline">Advisor</span>
            </button>
            <a
              href="https://github.com/KrishBnsl/inlui/tree/RAG"
              target="_blank"
              rel="noreferrer"
              aria-label="View the SeatCraft RAG branch on GitHub"
              className="text-neutral-600 hover:text-neutral-300 transition-colors"
            >
              <Code2 size={18} aria-hidden="true" />
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
                    Build a JoSAA list with uncertainty you can inspect.
                  </h1>
                  <p className="text-neutral-400 text-base leading-relaxed">
                    Enter your ranks and reservation details to see eligible options,
                    projected closing ranks, and where each choice sits on the safe,
                    target, or reach spectrum.
                  </p>
                </div>

                <div className="grid grid-cols-1 gap-3 max-w-xl border-y border-neutral-800 py-4 sm:grid-cols-3">
                  <div>
                    <p className="text-sm font-semibold text-white">Python inference</p>
                    <p className="text-xs text-neutral-500">Primary prediction path</p>
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-white">Rank-aware</p>
                    <p className="text-xs text-neutral-500">Main and Advanced routing</p>
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-white">Uncertainty shown</p>
                    <p className="text-xs text-neutral-500">Prediction intervals, not guarantees</p>
                  </div>
                </div>

                {/* Chat Advisor CTA */}
                <motion.button
                  id="open-chat-advisor"
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.15 }}
                  onClick={() => setChatOpen(true)}
                  aria-label="Open AI counselling advisor"
                  className="flex items-center gap-2.5 px-4 py-2.5 rounded-xl text-sm font-medium
                    bg-gradient-to-r from-cyan-600/20 to-teal-600/20
                    border border-cyan-500/30 text-cyan-300
                    hover:from-cyan-600/30 hover:to-teal-600/30 hover:border-cyan-500/50 hover:text-cyan-100
                    transition-all duration-200 shadow-sm shadow-cyan-900/20 max-w-xs"
                >
                  <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-cyan-500 to-teal-600 flex items-center justify-center shrink-0">
                    <MessageCircle size={12} className="text-white" />
                  </div>
                  <span>Ask the Counselling Advisor</span>
                  <span className="ml-auto text-[10px] px-1.5 py-0.5 rounded-full bg-cyan-500/15 border border-cyan-500/20 text-cyan-400">
                    AI
                  </span>
                </motion.button>

                <p className="max-w-xl text-xs leading-relaxed text-neutral-500">
                  Decision support, not an admission guarantee. Results depend on the
                  active model artifacts and historical data reported after submission.
                </p>
              </div>

              <div className="space-y-4">
                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: -8 }}
                    animate={{ opacity: 1, y: 0 }}
                    role="alert"
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
              <div className="text-center mb-10 space-y-3" role="status" aria-live="polite">
                <div className="inline-flex items-center gap-2 text-cyan-300 text-sm">
                  <span className="inline-block w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
                  Waiting for the ML inference service
                </div>
                <p className="text-neutral-500 text-xs">
                  No local or mock recommendations will replace the service response.
                </p>
              </div>
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
              aria-live="polite"
            >
              {/* Header */}
              <div className="flex items-end justify-between gap-4 border-b border-neutral-800 pb-5">
                <div>
                  <h2 className="text-2xl font-semibold text-white">
                    Model recommendations
                  </h2>
                  <p className="text-neutral-500 text-sm mt-1">
                    Every row states the exam rank used, uncalibrated admission-likelihood
                    estimate, heuristic score, and 90% prediction interval.
                  </p>
                </div>
              </div>

              <ResearchTransparency response={response} />

              <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
                <section className="space-y-6">
                  {/* Summary Cards */}
                  <SummaryCards
                    total_options={response.total_options}
                    safest_choice={response.safest_choice}
                    top_upgrade={response.top_upgrade}
                    results={response.results}
                    candidateCounts={response.candidate_counts}
                    returnedBucketCounts={response.returned_bucket_counts}
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

      {/* Counselling Advisor Drawer */}
      <ChatAdvisor 
        open={chatOpen} 
        onClose={() => setChatOpen(false)} 
        recommendationData={response} 
      />
    </div>
  );
}
