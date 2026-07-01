import type { PredictionResult } from "./types";

export type ResultSortMode =
  | "best_fit"
  | "highest_probability"
  | "most_competitive"
  | "safest_backup"
  | "institute_name"
  | "branch_name";

export const RESULT_SORT_LABELS: Record<ResultSortMode, string> = {
  best_fit: "Best fit",
  highest_probability: "Highest probability",
  most_competitive: "Most competitive cutoff",
  safest_backup: "Safest backup",
  institute_name: "Institute name",
  branch_name: "Branch name",
};

function score(result: PredictionResult): number {
  return result.recommendation_score ?? result.probability_percent ?? 0;
}

export function sortResults(
  results: PredictionResult[],
  mode: ResultSortMode = "best_fit"
): PredictionResult[] {
  return [...results].sort((a, b) => {
    if (mode === "highest_probability") {
      return b.probability_percent - a.probability_percent;
    }

    if (mode === "most_competitive") {
      return a.projected_closing_rank - b.projected_closing_rank;
    }

    if (mode === "safest_backup") {
      const safetyA = a.safety_score ?? a.probability_percent;
      const safetyB = b.safety_score ?? b.probability_percent;
      return (
        safetyB - safetyA ||
        b.probability_percent - a.probability_percent ||
        b.projected_closing_rank - a.projected_closing_rank
      );
    }

    if (mode === "institute_name") {
      return (
        a.institute_name.localeCompare(b.institute_name) ||
        a.program_name.localeCompare(b.program_name)
      );
    }

    if (mode === "branch_name") {
      return (
        a.program_name.localeCompare(b.program_name) ||
        a.institute_name.localeCompare(b.institute_name)
      );
    }

    return (
      score(b) - score(a) ||
      a.projected_closing_rank - b.projected_closing_rank ||
      b.probability_percent - a.probability_percent
    );
  });
}
