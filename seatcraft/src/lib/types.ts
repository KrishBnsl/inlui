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
  rank_type_used?: "main" | "advanced";
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
  return results.filter((result) => result.recommendation_bucket === bucket);
}

export function getTopBucketChoice(
  results: PredictionResult[],
  bucket: RecommendationBucket
): PredictionResult | undefined {
  return results
    .filter((result) => result.recommendation_bucket === bucket)
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
