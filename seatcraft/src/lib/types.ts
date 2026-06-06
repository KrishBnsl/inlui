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
  projected_closing_rank: number;
  historical_data: HistoricalDataPoint[];
}

export interface SimulationResponse {
  total_options: number;
  safest_choice: string;
  top_upgrade: string;
  results: PredictionResult[];
}

export type ProbabilityTier = "safe" | "target" | "reach";

export function getProbabilityTier(probability: number): ProbabilityTier {
  if (probability >= 70) return "safe";
  if (probability >= 35) return "target";
  return "reach";
}
