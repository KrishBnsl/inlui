"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ChevronRight,
  Loader2,
  Calculator,
  ToggleLeft,
  ToggleRight,
  Info,
  MapPin,
} from "lucide-react";
import type { SimulationInput, Category, IndianState, Gender } from "@/lib/types";
import { INDIAN_STATES } from "@/lib/types";

interface IntakeFormProps {
  onSubmit: (input: SimulationInput) => void;
  isLoading: boolean;
}

const CATEGORIES: Category[] = ["OPEN", "OBC-NCL", "SC", "ST", "EWS"];
const GENDERS: { value: Gender; label: string }[] = [
  { value: "Gender-Neutral", label: "Gender-Neutral" },
  { value: "Female-only", label: "Female-only (Supernumerary)" },
];

const LOADING_MESSAGES = [
  "Fetching historical cutoffs…",
  "Routing Home State quota allocations…",
  "Running 1,000 rank iterations…",
  "Fitting volatility distributions…",
  "Building allotment probability matrix…",
  "Ranking predictions…",
];

export default function IntakeForm({ onSubmit, isLoading }: IntakeFormProps) {
  const [mainRank, setMainRank] = useState<string>("");
  const [advancedRank, setAdvancedRank] = useState<string>("");
  const [category, setCategory] = useState<Category>("OPEN");
  const [homeState, setHomeState] = useState<IndianState | "">("");
  const [gender, setGender] = useState<Gender>("Gender-Neutral");
  const [isPwd, setIsPwd] = useState(false);

  const [mainRankError, setMainRankError] = useState("");
  const [advancedRankError, setAdvancedRankError] = useState("");
  const [homeStateError, setHomeStateError] = useState("");
  const [loadingStep, setLoadingStep] = useState(0);

  const validateMainRank = (val: string) => {
    const n = Number(val);
    if (!val) return setMainRankError("JEE Main rank is required");
    if (isNaN(n) || n <= 0) return setMainRankError("Enter a valid positive rank");
    if (n > 1_200_000) return setMainRankError("Exceeds JEE Main scale (~12,00,000)");
    setMainRankError("");
  };

  const validateAdvancedRank = (val: string) => {
    if (!val) return setAdvancedRankError("");
    const n = Number(val);
    if (isNaN(n) || n <= 0) return setAdvancedRankError("Enter a valid positive rank");
    if (n > 50_000) return setAdvancedRankError("JEE Advanced rank must be 50,000 or below");
    setAdvancedRankError("");
  };

  const validateHomeState = (val: string) => {
    if (!val) return setHomeStateError("Home state is required");
    setHomeStateError("");
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    validateMainRank(mainRank);
    validateHomeState(homeState);
    if (advancedRank) validateAdvancedRank(advancedRank);

    const mn = Number(mainRank);
    if (!mainRank || isNaN(mn) || mn <= 0) return;
    if (!homeState) return;
    if (advancedRank && advancedRankError) return;

    let step = 0;
    const interval = setInterval(() => {
      step = (step + 1) % LOADING_MESSAGES.length;
      setLoadingStep(step);
    }, 400);
    setTimeout(() => clearInterval(interval), 2200);

    onSubmit({
      main_rank: mn,
      advanced_rank: advancedRank ? Number(advancedRank) : undefined,
      category,
      home_state: homeState as IndianState,
      gender,
      is_pwd: isPwd,
      top_n: 100,
    });
  };

  const inputClass = (hasError: boolean) =>
    `w-full bg-neutral-950 border rounded-md px-3.5 py-3 text-white text-base
     placeholder:text-neutral-600 transition-colors duration-150 outline-none
     focus:ring-2 focus:ring-cyan-500/25 focus:border-cyan-500
     ${hasError
       ? "border-red-500/60 ring-2 ring-red-500/20"
       : "border-neutral-800 hover:border-neutral-700"
     }`;

  const selectClass = (hasError: boolean) =>
    `w-full bg-neutral-950 border rounded-md px-3.5 py-3 text-white text-base
     transition-colors duration-150 outline-none cursor-pointer
     focus:ring-2 focus:ring-cyan-500/25 focus:border-cyan-500
     ${hasError
       ? "border-red-500/60 ring-2 ring-red-500/20"
       : "border-neutral-800 hover:border-neutral-700"
     }`;

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="w-full max-w-2xl mx-auto border border-neutral-800 bg-neutral-900/35 rounded-lg p-4 sm:p-6"
    >
      <form onSubmit={handleSubmit} className="space-y-5">

        {/* Rank inputs */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {/* JEE Main */}
          <div className="space-y-1.5">
            <label className="block text-sm font-medium text-neutral-300">
              JEE Main Rank <span className="text-red-400">*</span>
            </label>
            <p className="text-xs text-neutral-500">For NITs, IIITs, GFTIs</p>
            <input
              type="number"
              value={mainRank}
              onChange={(e) => {
                setMainRank(e.target.value);
                if (mainRankError) validateMainRank(e.target.value);
              }}
              onBlur={() => validateMainRank(mainRank)}
              placeholder="e.g. 5100"
              min={1}
              max={1200000}
              className={inputClass(!!mainRankError)}
            />
            {mainRankError && (
              <motion.p initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }}
                className="text-xs text-red-400 ml-0.5">{mainRankError}</motion.p>
            )}
          </div>

          {/* JEE Advanced */}
          <div className="space-y-1.5">
            <label className="block text-sm font-medium text-neutral-300">
              JEE Advanced Rank{" "}
              <span className="text-xs text-neutral-500 font-normal">optional</span>
            </label>
            <p className="text-xs text-neutral-500">Required for IIT predictions</p>
            <input
              type="number"
              value={advancedRank}
              onChange={(e) => {
                setAdvancedRank(e.target.value);
                if (advancedRankError) validateAdvancedRank(e.target.value);
              }}
              onBlur={() => validateAdvancedRank(advancedRank)}
              placeholder="e.g. 420"
              min={1}
              max={50000}
              className={inputClass(!!advancedRankError)}
            />
            {advancedRankError ? (
              <motion.p initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }}
                className="text-xs text-red-400 ml-0.5">{advancedRankError}</motion.p>
            ) : (
              <p className="text-xs text-neutral-600 ml-0.5 flex items-center gap-1">
                <Info size={11} />Leave blank to skip IIT options
              </p>
            )}
          </div>
        </div>

        {/* Home State — full width */}
        <div className="space-y-1.5">
          <label className="block text-sm font-medium text-neutral-300 flex items-center gap-1.5">
            <MapPin size={13} className="text-neutral-500" />
            Home State <span className="text-red-400">*</span>
          </label>
          <p className="text-xs text-neutral-500">
            Used to apply HS quota where applicable; other institutes use AI quota.
          </p>
          <select
            value={homeState}
            onChange={(e) => {
              setHomeState(e.target.value as IndianState);
              if (homeStateError) validateHomeState(e.target.value);
            }}
            onBlur={() => validateHomeState(homeState)}
            className={selectClass(!!homeStateError)}
          >
            <option value="" className="bg-neutral-900 text-neutral-500">
              Select your home state
            </option>
            {INDIAN_STATES.map((s) => (
              <option key={s} value={s} className="bg-neutral-900 text-white">
                {s}
              </option>
            ))}
          </select>
          {homeStateError && (
            <motion.p initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }}
              className="text-xs text-red-400 ml-0.5">{homeStateError}</motion.p>
          )}

          {/* Show which institutes will get HS quota if state is selected */}
          {homeState && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              className="mt-2 px-3 py-2 rounded-md bg-emerald-500/8 border border-emerald-500/20
                text-xs text-emerald-400 flex items-start gap-2"
            >
              <span>
                NIT/IIIT/GFTI institutes in{" "}
                <strong className="text-emerald-300">{homeState}</strong> will be
                evaluated using your Home State quota cutoffs.
              </span>
            </motion.div>
          )}
        </div>

        {/* Category + Gender */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="space-y-1.5">
            <label className="block text-sm font-medium text-neutral-300">Seat Category</label>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value as Category)}
              className={selectClass(false)}
            >
              {CATEGORIES.map((c) => (
                <option key={c} value={c} className="bg-neutral-900">{c}</option>
              ))}
            </select>
          </div>

          <div className="space-y-1.5">
            <label className="block text-sm font-medium text-neutral-300">Gender</label>
            <div className="grid grid-cols-1 gap-2">
              {GENDERS.map((g) => (
                <button
                  key={g.value}
                  type="button"
                  onClick={() => setGender(g.value)}
                  className={`px-3.5 py-2.5 rounded-md border text-sm font-medium transition-colors duration-150 text-left
                    ${gender === g.value
                      ? "border-cyan-500 bg-cyan-500/10 text-cyan-300"
                      : "border-neutral-800 text-neutral-400 hover:border-neutral-600 hover:text-neutral-300"
                    }`}
                >
                  {g.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* PwD Toggle */}
        <div className="flex items-center justify-between p-4 bg-neutral-950 border border-neutral-800 rounded-md">
          <div>
            <p className="text-sm font-medium text-neutral-200">PwD Status</p>
            <p className="text-xs text-neutral-500 mt-0.5">Person with Disability horizontal reservation</p>
          </div>
          <button type="button" onClick={() => setIsPwd((v) => !v)}
            className="transition-colors">
            {isPwd
              ? <ToggleRight size={36} className="text-cyan-400" />
              : <ToggleLeft size={36} className="text-neutral-600" />}
          </button>
        </div>

        {/* Submit */}
        <button
          type="submit"
          disabled={isLoading}
          className={`w-full flex items-center justify-center gap-3 py-3.5 px-6 rounded-md
            font-semibold text-base transition-colors duration-150 group
            ${isLoading
              ? "bg-cyan-600/40 text-cyan-300 cursor-not-allowed"
              : "bg-cyan-600 hover:bg-cyan-500 text-white"
            }`}
        >
          <AnimatePresence mode="wait">
            {isLoading ? (
              <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="flex items-center gap-3">
                <Loader2 size={18} className="animate-spin" />
                <span className="text-sm tabular-nums">{LOADING_MESSAGES[loadingStep]}</span>
              </motion.div>
            ) : (
              <motion.div key="idle" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="flex items-center gap-2">
                <Calculator size={18} />
                <span>Calculate options</span>
                <ChevronRight size={18} className="transition-transform group-hover:translate-x-0.5" />
              </motion.div>
            )}
          </AnimatePresence>
        </button>
      </form>
    </motion.div>
  );
}
