// RAG API client — talks to the Python LangChain RAG service on port 8081.

const RAG_BASE = process.env.NEXT_PUBLIC_RAG_URL ?? "http://localhost:8081";

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
}

export interface UploadResponse {
  ok: boolean;
  chunks_added: number;
  source: string;
}

// ── API calls ─────────────────────────────────────────────────────────────────

/** Upload a PDF or image file for indexing. */
export async function uploadDocument(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${RAG_BASE}/api/rag/upload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Upload failed (${res.status})`);
  }
  return res.json();
}

/**
 * Ask a question, optionally with an inline image (base64 JPEG/PNG).
 * The backend embeds the question, retrieves context, and calls GPT-4o-mini.
 */
export async function askQuestion(
  question: string,
  imageB64?: string
): Promise<AskResponse> {
  const res = await fetch(`${RAG_BASE}/api/rag/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      image_b64: imageB64 ?? undefined,
    }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Ask failed (${res.status})`);
  }
  return res.json();
}

/** Fetch the current store status (chunk count + document names). */
export async function getRagStatus(): Promise<RagStatus> {
  const res = await fetch(`${RAG_BASE}/api/rag/status`);
  if (!res.ok) throw new Error(`Status failed (${res.status})`);
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
