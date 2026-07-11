// RAG API client — talks to the Python LangChain RAG service on port 8081.

const RAG_BASE = (process.env.NEXT_PUBLIC_RAG_URL ?? "http://localhost:8081").replace(
  /\/$/,
  ""
);
const REQUEST_TIMEOUT_MS = 30_000;

// ── Types ─────────────────────────────────────────────────────────────────────

export interface Source {
  source: string;
  excerpt: string;
}

export interface AskResponse {
  answer: string;
  sources: Source[];
}

export interface RagStatus {
  chunk_count: number;
  documents: string[];
  vector_store_available: boolean;
  persistence: "in_memory";
}

export interface RagHealth {
  status: string;
  ready: boolean;
  google_api_key_configured: boolean;
  missing: string[];
  embedding_provider_ready: boolean;
  chat_provider_ready: boolean;
  vector_store_available: boolean;
  chunk_count: number;
  degraded_reasons: string[];
  configured_models: Record<string, string>;
}

export interface UploadResponse {
  ok: boolean;
  chunks_added: number;
  source: string;
}

async function readApiError(res: Response, fallback: string): Promise<string> {
  const raw = await res.text();
  try {
    const parsed = JSON.parse(raw);
    if (typeof parsed?.detail === "string") return parsed.detail;
  } catch {
    // Avoid surfacing arbitrary upstream bodies or HTML error pages.
  }
  return fallback;
}

// ── API calls ─────────────────────────────────────────────────────────────────

/** Upload a PDF or image file for indexing. */
export async function uploadDocument(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  let res: Response;
  try {
    res = await fetch(`${RAG_BASE}/api/rag/upload`, {
      method: "POST",
      body: form,
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
  } catch {
    throw new Error("RAG service is unavailable or the upload timed out.");
  }
  if (!res.ok) {
    throw new Error(await readApiError(res, `Document upload failed (${res.status}).`));
  }
  return res.json();
}

/**
 * Ask a question, optionally with an inline image (base64 JPEG/PNG).
 * The backend embeds the question, retrieves context, and calls Gemini.
 */
export async function askQuestion(
  question: string,
  imageB64?: string,
  mlContext?: string
): Promise<AskResponse> {
  let res: Response;
  try {
    res = await fetch(`${RAG_BASE}/api/rag/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        image_b64: imageB64 ?? undefined,
        ml_context: mlContext ?? undefined,
      }),
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
  } catch {
    throw new Error("RAG service is unavailable or the advisor request timed out.");
  }
  if (!res.ok) {
    throw new Error(await readApiError(res, `Advisor request failed (${res.status}).`));
  }
  return res.json();
}

/** Fetch the current store status (chunk count + document names). */
export async function getRagStatus(): Promise<RagStatus> {
  const res = await fetch(`${RAG_BASE}/api/rag/status`, {
    cache: "no-store",
    signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
  });
  if (!res.ok) throw new Error(`Status failed (${res.status})`);
  return res.json();
}

export async function getRagHealth(): Promise<RagHealth> {
  let res: Response;
  try {
    res = await fetch(`${RAG_BASE}/health`, {
      cache: "no-store",
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
  } catch {
    throw new Error("RAG service unavailable.");
  }
  if (!res.ok) throw new Error("RAG service unavailable.");
  return res.json();
}

/** Convert a File to a base64 string (data URL stripped). */
export async function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      // Strip the "data:...;base64," prefix
      resolve(result.split(",")[1] ?? result);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}
