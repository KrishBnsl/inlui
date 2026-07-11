import type { SimulationInput, SimulationResponse } from "./types";

const ML_BASE = (process.env.NEXT_PUBLIC_ML_URL ?? "http://localhost:8082").replace(
  /\/$/,
  ""
);
const REQUEST_TIMEOUT_MS = 20_000;

interface MLHealth {
  status: string;
  artifacts_loaded: boolean;
  universe_size: number;
  artifact_status?: Record<string, string>;
}

async function readError(res: Response, defaultMessage: string): Promise<string> {
  const raw = await res.text();
  try {
    const body = JSON.parse(raw);
    if (typeof body?.detail === "string") return body.detail;
    if (Array.isArray(body?.detail)) {
      return body.detail
        .map((item: { msg?: string }) => item.msg)
        .filter(Boolean)
        .join("; ");
    }
  } catch {
    // Do not render arbitrary HTML or raw upstream bodies in the application.
  }
  return defaultMessage;
}

function assertSimulationResponse(value: unknown): asserts value is SimulationResponse {
  if (!value || typeof value !== "object") {
    throw new Error("ML service returned an invalid prediction contract.");
  }
  const response = value as Partial<SimulationResponse>;
  if (
    typeof response.total_options !== "number" ||
    !Array.isArray(response.results) ||
    response.results.some(
      (item) =>
        !item ||
        typeof item.id !== "string" ||
        typeof item.institute_name !== "string" ||
        typeof item.program_name !== "string" ||
        typeof item.probability_percent !== "number" ||
        typeof item.projected_closing_rank !== "number"
    )
  ) {
    throw new Error("ML service returned an invalid prediction contract.");
  }
}

export async function runMLSimulation(
  input: SimulationInput
): Promise<SimulationResponse> {
  let res: Response;
  try {
    res = await fetch(`${ML_BASE}/api/v1/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
  } catch {
    throw new Error("ML prediction service is unavailable or timed out.");
  }

  if (!res.ok) {
    const message = await readError(
      res,
      "ML inference failed. Make sure the inference API is running on port 8082."
    );
    throw new Error(message);
  }

  let data: unknown;
  try {
    data = await res.json();
  } catch {
    throw new Error("ML service returned an invalid prediction contract.");
  }
  assertSimulationResponse(data);
  return data;
}

export async function getMLHealth(): Promise<MLHealth> {
  let res: Response;
  try {
    res = await fetch(`${ML_BASE}/api/v1/health`, {
      cache: "no-store",
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
  } catch {
    throw new Error("ML service unavailable.");
  }
  if (!res.ok) {
    throw new Error("ML service unavailable.");
  }
  return res.json();
}

export async function getChatContext(
  question: string,
  recommendationOutput: SimulationResponse | null
): Promise<{ context_block: string; key_facts: Record<string, unknown> }> {
  const res = await fetch(`${ML_BASE}/api/v1/chat-context`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      recommendation_output: recommendationOutput,
    }),
    signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
  });

  if (!res.ok) {
    throw new Error(
      await readError(res, "Recommendation context is currently unavailable.")
    );
  }
  return res.json();
}
