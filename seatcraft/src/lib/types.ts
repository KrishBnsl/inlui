export type Category = "OPEN" | "OBC-NCL" | "SC" | "ST" | "EWS";
export type Gender = "Gender-Neutral" | "Female-only";
export type QuotaApplied = "AI" | "HS";

// All states/UTs that appear in JoSAA institute home-state routing
export const INDIAN_STATES = [
  "Andhra Pradesh",
  "Arunachal Pradesh",
  "Assam",
  "Bihar",
  "Chhattisgarh",
  "Goa",
  "Gujarat",
  "Haryana",
  "Himachal Pradesh",
  "Jharkhand",
  "Karnataka",
  "Kerala",
  "Madhya Pradesh",
  "Maharashtra",
  "Manipur",
  "Meghalaya",
  "Mizoram",
  "Nagaland",
  "Odisha",
  "Punjab",
  "Rajasthan",
  "Sikkim",
  "Tamil Nadu",
  "Telangana",
  "Tripura",
  "Uttar Pradesh",
  "Uttarakhand",
  "West Bengal",
  "Delhi",
  "Chandigarh",
  "Jammu and Kashmir",
  "Ladakh",
  "Puducherry",
] as const;

export type IndianState = typeof INDIAN_STATES[number];

export interface SimulationInput {
  main_rank: number;         // JEE Main rank — NITs, IIITs, GFTIs
  advanced_rank?: number;    // JEE Advanced rank — IITs only (optional)
  category: Category;
  home_state: IndianState;   // Used to determine AI vs HS quota per institute
  gender: Gender;
  is_pwd: boolean;
  pref_inst_types?: Array<"IIT" | "NIT" | "IIIT" | "GFTI">;
  pref_branch_keywords?: string[];
  round?: number;
  top_n?: number;
  sort_mode?: "best_fit" | "highest_probability" | "most_competitive" | "safest_backup";
}

export interface HistoricalDataPoint {
  year: number;
  opening_rank: number;
  closing_rank: number;
}

export interface PredictionResult {
  id: string;
  institute_name: string;
  institute_type: "IIT" | "NIT" | "IIIT" | "GFTI";
  program_name: string;
  quota_applied: QuotaApplied;  // which quota was actually used for this result
  category: string;
  probability_percent: number;
  admission_probability?: number;
  model_probability_percent?: number;
  calibrated_probability_percent?: number;
  projected_closing_rank: number;
  historical_data: HistoricalDataPoint[];

  // Optional ML metadata
  confidence_label?: "Safe" | "Moderate" | "Ambitious";
  uncertainty_lower?: number;
  uncertainty_upper?: number;
  safety_margin?: number;
  explanation?: string;
  recommendation_score?: number;
  fit_score?: number;
  competitiveness_score?: number;
  institute_score?: number;
  branch_score?: number;
  safety_score?: number;
  recommendation_bucket?: RecommendationBucket;
  rank_used?: number;
  rank_type_used?: "JEE_MAIN" | "JEE_ADVANCED" | "main" | "advanced";
  rank_ratio?: number;
}


export interface SimulationResponse {
  total_options: number;
  safest_choice: string;
  top_upgrade: string;
  results: PredictionResult[];

  // Optional ML summary metadata
  most_ambitious?: string;
  safe_count?: number;
  moderate_count?: number;
  ambitious_count?: number;
  bucket_counts?: Record<RecommendationBucket, number>;
  institute_type_counts?: Record<"IIT" | "NIT" | "IIIT" | "GFTI", number>;
  rank_window_debug?: {
    main_rank?: number;
    advanced_rank?: number | null;
    main_reach_window?: [number, number];
    advanced_reach_window?: [number, number] | null;
  };
}

export type RecommendationBucket =
  | "safe_backup"
  | "best_realistic"
  | "ambitious_reach"
  | "unlikely_reach";

export type ProbabilityTier = "safe" | "target" | "reach";

export type BucketFilter = "ALL" | ProbabilityTier;

export const BUCKET_FILTERS: Record<ProbabilityTier, RecommendationBucket> = {
  safe: "safe_backup",
  target: "best_realistic",
  reach: "ambitious_reach",
};

export function getBucketForFilter(filter: BucketFilter): RecommendationBucket | null {
  return filter === "ALL" ? null : BUCKET_FILTERS[filter];
}

export function filterByBucket(
  results: PredictionResult[],
  filter: BucketFilter
): PredictionResult[] {
  const bucket = getBucketForFilter(filter);
  if (!bucket) return results;
  return results.filter((result) => inferBucket(result) === bucket);
}

export function getTopBucketChoice(
  results: PredictionResult[],
  bucket: RecommendationBucket
): PredictionResult | undefined {
  return results
    .filter((result) => inferBucket(result) === bucket)
    .sort((a, b) => (b.recommendation_score ?? 0) - (a.recommendation_score ?? 0))[0];
}

export function getTopFilteredChoice(
  results: PredictionResult[],
  filter: ProbabilityTier
): PredictionResult | undefined {
  return filterByBucket(results, filter)
    .sort((a, b) => (b.recommendation_score ?? 0) - (a.recommendation_score ?? 0))[0];
}

export function formatChoiceName(result: PredictionResult | undefined, fallback = "—"): string {
  return result ? `${result.institute_name} — ${result.program_name}` : fallback;
}

export function getProbabilityTier(probability: number): ProbabilityTier {
  if (probability >= 70) return "safe";
  if (probability >= 35) return "target";
  return "reach";
}

export function inferBucket(result: PredictionResult): RecommendationBucket {
  if (result.recommendation_bucket) return result.recommendation_bucket;
  if (result.confidence_label === "Safe") return "safe_backup";
  if (result.confidence_label === "Moderate") return "best_realistic";
  if (result.confidence_label === "Ambitious") return "ambitious_reach";

  const probability = result.probability_percent;
  if (probability >= 70) return "safe_backup";
  if (probability >= 35) return "best_realistic";
  return "ambitious_reach";
}

export function getTierForResult(result: PredictionResult): ProbabilityTier {
  const bucket = inferBucket(result);
  if (bucket === "safe_backup") return "safe";
  if (bucket === "best_realistic") return "target";
  if (bucket === "ambitious_reach") return "reach";
  return getProbabilityTier(result.probability_percent);
}
