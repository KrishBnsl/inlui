"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Calculator, ChevronRight, Info, Loader2, MapPin } from "lucide-react";
import type { Category, Gender, IndianState, SimulationInput } from "@/lib/types";
import { INDIAN_STATES } from "@/lib/types";

interface IntakeFormProps {
  onSubmit: (input: SimulationInput) => void;
  isLoading: boolean;
}

type InstituteType = "IIT" | "NIT" | "IIIT" | "GFTI";

interface FormErrors {
  mainRank?: string;
  advancedRank?: string;
  homeState?: string;
}

const CATEGORIES: Category[] = ["OPEN", "OBC-NCL", "SC", "ST", "EWS"];
const INSTITUTE_TYPES: InstituteType[] = ["IIT", "NIT", "IIIT", "GFTI"];
const GENDERS: { value: Gender; label: string }[] = [
  { value: "Gender-Neutral", label: "Gender-Neutral" },
  { value: "Female-only", label: "Female-only (supernumerary)" },
];

function validateRanks(
  mainRank: string,
  advancedRank: string,
  homeState: string
): FormErrors {
  const errors: FormErrors = {};
  const main = Number(mainRank);
  const advanced = Number(advancedRank);

  if (!mainRank) errors.mainRank = "JEE Main rank is required.";
  else if (!Number.isInteger(main) || main < 1)
    errors.mainRank = "Enter a whole-number rank of 1 or greater.";
  else if (main > 1_200_000)
    errors.mainRank = "JEE Main rank must be 12,00,000 or below.";

  if (advancedRank && (!Number.isInteger(advanced) || advanced < 1))
    errors.advancedRank = "Enter a whole-number rank of 1 or greater.";
  else if (advancedRank && advanced > 50_000)
    errors.advancedRank = "JEE Advanced rank must be 50,000 or below.";

  if (!homeState) errors.homeState = "Home state is required for quota routing.";
  return errors;
}

export default function IntakeForm({ onSubmit, isLoading }: IntakeFormProps) {
  const [isInteractive, setIsInteractive] = useState(false);
  const [mainRank, setMainRank] = useState("");
  const [advancedRank, setAdvancedRank] = useState("");
  const [category, setCategory] = useState<Category>("OPEN");
  const [homeState, setHomeState] = useState<IndianState | "">("");
  const [gender, setGender] = useState<Gender>("Gender-Neutral");
  const [isPwd, setIsPwd] = useState(false);
  const [instituteTypes, setInstituteTypes] = useState<InstituteType[]>([]);
  const [branchKeywords, setBranchKeywords] = useState("");
  const [round, setRound] = useState(5);
  const [errors, setErrors] = useState<FormErrors>({});

  useEffect(() => {
    const hydrationTick = window.setTimeout(() => setIsInteractive(true), 0);
    return () => window.clearTimeout(hydrationTick);
  }, []);

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const nextErrors = validateRanks(mainRank, advancedRank, homeState);
    setErrors(nextErrors);
    if (Object.keys(nextErrors).length > 0) return;

    const keywords = branchKeywords
      .split(",")
      .map((keyword) => keyword.trim())
      .filter(Boolean);

    onSubmit({
      main_rank: Number(mainRank),
      advanced_rank: advancedRank ? Number(advancedRank) : undefined,
      category,
      home_state: homeState as IndianState,
      gender,
      is_pwd: isPwd,
      pref_inst_types: instituteTypes.length ? instituteTypes : undefined,
      pref_branch_keywords: keywords.length ? keywords : undefined,
      round,
      top_n: 100,
      sort_mode: "best_fit",
    });
  };

  const toggleInstituteType = (value: InstituteType) => {
    setInstituteTypes((current) =>
      current.includes(value)
        ? current.filter((item) => item !== value)
        : [...current, value]
    );
  };

  const inputClass = (hasError = false) =>
    `w-full rounded-md border bg-neutral-950 px-3.5 py-3 text-base text-white outline-none transition-colors placeholder:text-neutral-600 focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/25 ${
      hasError ? "border-red-500/60" : "border-neutral-800 hover:border-neutral-700"
    }`;

  return (
    <motion.div
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      className="mx-auto w-full max-w-2xl rounded-lg border border-neutral-800 bg-neutral-900/35 p-4 sm:p-6"
    >
      <form onSubmit={handleSubmit} noValidate>
        <fieldset
          disabled={isLoading || !isInteractive}
          aria-busy={isLoading}
          className="space-y-5 disabled:opacity-70"
        >
          <div className="rounded-md border border-cyan-500/20 bg-cyan-500/5 p-3 text-xs leading-relaxed text-neutral-400">
            <p className="flex items-center gap-1.5 font-medium text-cyan-300">
              <Info size={13} aria-hidden="true" /> Which rank is used?
            </p>
            <p className="mt-1.5">
              <strong className="text-neutral-200">JEE Main</strong> routes NIT, IIIT,
              and GFTI rows. <strong className="text-neutral-200">JEE Advanced</strong>{" "}
              routes IIT rows only; leave it blank to exclude IIT options.
            </p>
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="space-y-1.5">
              <label htmlFor="main-rank" className="block text-sm font-medium text-neutral-300">
                JEE Main rank <span className="text-red-400">*</span>
              </label>
              <p id="main-rank-help" className="text-xs text-neutral-500">
                Common Rank List rank for NITs, IIITs, and GFTIs
              </p>
              <input
                id="main-rank"
                name="main_rank"
                type="number"
                inputMode="numeric"
                value={mainRank}
                onChange={(event) => {
                  setMainRank(event.target.value);
                  if (errors.mainRank) setErrors((value) => ({ ...value, mainRank: undefined }));
                }}
                min={1}
                max={1_200_000}
                step={1}
                placeholder="e.g. 3000"
                aria-invalid={Boolean(errors.mainRank)}
                aria-describedby={`main-rank-help${errors.mainRank ? " main-rank-error" : ""}`}
                className={inputClass(Boolean(errors.mainRank))}
              />
              {errors.mainRank && (
                <p id="main-rank-error" role="alert" className="text-xs text-red-400">
                  {errors.mainRank}
                </p>
              )}
            </div>

            <div className="space-y-1.5">
              <label htmlFor="advanced-rank" className="block text-sm font-medium text-neutral-300">
                JEE Advanced rank <span className="font-normal text-neutral-500">optional</span>
              </label>
              <p id="advanced-rank-help" className="text-xs text-neutral-500">
                Required only for IIT recommendations
              </p>
              <input
                id="advanced-rank"
                name="advanced_rank"
                type="number"
                inputMode="numeric"
                value={advancedRank}
                onChange={(event) => {
                  setAdvancedRank(event.target.value);
                  if (errors.advancedRank)
                    setErrors((value) => ({ ...value, advancedRank: undefined }));
                }}
                min={1}
                max={50_000}
                step={1}
                placeholder="e.g. 6700"
                aria-invalid={Boolean(errors.advancedRank)}
                aria-describedby={`advanced-rank-help${
                  errors.advancedRank ? " advanced-rank-error" : ""
                }`}
                className={inputClass(Boolean(errors.advancedRank))}
              />
              {errors.advancedRank && (
                <p id="advanced-rank-error" role="alert" className="text-xs text-red-400">
                  {errors.advancedRank}
                </p>
              )}
            </div>
          </div>

          <div className="space-y-1.5">
            <label htmlFor="home-state" className="flex items-center gap-1.5 text-sm font-medium text-neutral-300">
              <MapPin size={13} className="text-neutral-500" aria-hidden="true" />
              Home state <span className="text-red-400">*</span>
            </label>
            <p id="home-state-help" className="text-xs text-neutral-500">
              Used for Home State routing when the model has institute-state metadata.
            </p>
            <select
              id="home-state"
              name="home_state"
              value={homeState}
              onChange={(event) => {
                setHomeState(event.target.value as IndianState);
                if (errors.homeState) setErrors((value) => ({ ...value, homeState: undefined }));
              }}
              aria-invalid={Boolean(errors.homeState)}
              aria-describedby={`home-state-help${errors.homeState ? " home-state-error" : ""}`}
              className={inputClass(Boolean(errors.homeState))}
            >
              <option value="">Select your home state</option>
              {INDIAN_STATES.map((state) => (
                <option key={state} value={state}>
                  {state}
                </option>
              ))}
            </select>
            {errors.homeState && (
              <p id="home-state-error" role="alert" className="text-xs text-red-400">
                {errors.homeState}
              </p>
            )}
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="space-y-1.5">
              <label htmlFor="category" className="block text-sm font-medium text-neutral-300">
                Seat category
              </label>
              <select
                id="category"
                name="category"
                value={category}
                onChange={(event) => setCategory(event.target.value as Category)}
                className={inputClass()}
              >
                {CATEGORIES.map((value) => (
                  <option key={value}>{value}</option>
                ))}
              </select>
            </div>

            <fieldset className="space-y-1.5">
              <legend className="text-sm font-medium text-neutral-300">Seat pool</legend>
              <div className="grid gap-2" role="radiogroup" aria-label="Gender seat pool">
                {GENDERS.map((option) => (
                  <label
                    key={option.value}
                    className={`cursor-pointer rounded-md border px-3 py-2 text-sm transition-colors ${
                      gender === option.value
                        ? "border-cyan-500 bg-cyan-500/10 text-cyan-200"
                        : "border-neutral-800 text-neutral-400 hover:border-neutral-600"
                    }`}
                  >
                    <input
                      type="radio"
                      name="gender"
                      value={option.value}
                      checked={gender === option.value}
                      onChange={() => setGender(option.value)}
                      className="mr-2 accent-cyan-500"
                    />
                    {option.label}
                  </label>
                ))}
              </div>
            </fieldset>
          </div>

          <div className="flex items-center justify-between rounded-md border border-neutral-800 bg-neutral-950 p-4">
            <div>
              <p className="text-sm font-medium text-neutral-200">PwD status</p>
              <p id="pwd-help" className="mt-0.5 text-xs text-neutral-500">
                Person with Disability horizontal reservation
              </p>
            </div>
            <button
              type="button"
              role="switch"
              aria-checked={isPwd}
              aria-describedby="pwd-help"
              onClick={() => setIsPwd((value) => !value)}
              className={`relative h-7 w-12 rounded-full border transition-colors ${
                isPwd ? "border-cyan-400 bg-cyan-500" : "border-neutral-700 bg-neutral-800"
              }`}
            >
              <span
                className={`absolute top-0.5 h-5 w-5 rounded-full bg-white transition-transform ${
                  isPwd ? "translate-x-5" : "translate-x-0.5"
                }`}
              />
              <span className="sr-only">Toggle PwD status</span>
            </button>
          </div>

          <details className="rounded-md border border-neutral-800 bg-neutral-950/60 p-4">
            <summary className="cursor-pointer text-sm font-medium text-neutral-200">
              Optional programme filters
            </summary>
            <div className="mt-4 space-y-4 border-t border-neutral-800 pt-4">
              <fieldset>
                <legend className="text-xs font-medium text-neutral-400">
                  Institute types (none selected means all)
                </legend>
                <div className="mt-2 flex flex-wrap gap-2">
                  {INSTITUTE_TYPES.map((value) => (
                    <label key={value} className="flex items-center gap-2 rounded-md border border-neutral-800 px-3 py-2 text-xs text-neutral-300">
                      <input
                        type="checkbox"
                        checked={instituteTypes.includes(value)}
                        onChange={() => toggleInstituteType(value)}
                        className="accent-cyan-500"
                      />
                      {value}
                    </label>
                  ))}
                </div>
              </fieldset>
              <div>
                <label htmlFor="branch-keywords" className="text-xs font-medium text-neutral-400">
                  Branch keywords
                </label>
                <input
                  id="branch-keywords"
                  value={branchKeywords}
                  onChange={(event) => setBranchKeywords(event.target.value)}
                  placeholder="Computer Science, Electronics"
                  className={`${inputClass()} mt-2`}
                />
                <p className="mt-1 text-xs text-neutral-600">Separate multiple keywords with commas.</p>
              </div>
              <div>
                <label htmlFor="counselling-round" className="text-xs font-medium text-neutral-400">
                  JoSAA round
                </label>
                <select
                  id="counselling-round"
                  value={round}
                  onChange={(event) => setRound(Number(event.target.value))}
                  className={`${inputClass()} mt-2`}
                >
                  {[1, 2, 3, 4, 5, 6].map((value) => (
                    <option key={value} value={value}>
                      Round {value}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </details>

          <button
            type="submit"
            disabled={isLoading || !isInteractive}
            className="group flex w-full items-center justify-center gap-2 rounded-md bg-cyan-600 px-6 py-3.5 text-base font-semibold text-white transition-colors hover:bg-cyan-500 disabled:cursor-wait disabled:bg-cyan-700/50"
          >
            {isLoading ? (
              <>
                <Loader2 size={18} className="animate-spin" aria-hidden="true" />
                Requesting model predictions…
              </>
            ) : (
              <>
                <Calculator size={18} aria-hidden="true" />
                Calculate options
                <ChevronRight size={18} className="transition-transform group-hover:translate-x-0.5" aria-hidden="true" />
              </>
            )}
          </button>
        </fieldset>
      </form>
    </motion.div>
  );
}
