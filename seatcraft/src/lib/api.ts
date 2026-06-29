import type { SimulationInput, SimulationResponse } from "./types";

const ML_BASE = process.env.NEXT_PUBLIC_ML_URL ?? "http://localhost:8082";

interface MLHealth {
  status: string;
  artifacts_loaded: boolean;
  universe_size: number;
}

async function readError(res: Response, defaultMessage: string): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") return body.detail;
    if (Array.isArray(body?.detail)) {
      return body.detail
        .map((item: { msg?: string }) => item.msg)
        .filter(Boolean)
        .join("; ");
    }
  } catch {
    const text = await res.text();
    if (text) return text;
  }
  return defaultMessage;
}

export async function runMLSimulation(
  input: SimulationInput
): Promise<SimulationResponse> {
  const res = await fetch(`${ML_BASE}/api/v1/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });

  if (!res.ok) {
    const message = await readError(
      res,
      "ML inference failed. Make sure inference_service is running on port 8082."
    );
    throw new Error(message);
  }

  return res.json();
}

export async function getMLHealth(): Promise<MLHealth> {
  const res = await fetch(`${ML_BASE}/api/v1/health`);
  if (!res.ok) {
    throw new Error("ML inference service health check failed.");
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
  });

  if (!res.ok) return { context_block: "", key_facts: {} };
  return res.json();
}
