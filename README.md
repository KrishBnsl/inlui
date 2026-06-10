# inlui — JoSAA Counselling Platform

An end-all destination for correct JEE counselling decisions.

## Architecture

```
inlui/
├── sim_engine/    Rust backend (Axum HTTP server — simulation only)
│   └── src/
│       ├── bin/server.rs   — HTTP API (simulate endpoint)
│       ├── bin/ingest.rs   — one-shot CSV → PostgreSQL loader
│       └── predictor/      — Monte Carlo seat allotment engine
│
├── rag_service/   Python RAG service (FastAPI + LangChain)
│   ├── main.py             — FastAPI app (upload, ask, status)
│   ├── requirements.txt    — Python dependencies
│   └── Dockerfile          — container build
│
└── seatcraft/     Next.js frontend
    └── src/
        ├── app/page.tsx    — main JoSAA planner UI
        └── components/
            ├── ChatAdvisor.tsx  — counselling chatbot drawer
            └── ...
```

The Rust server handles simulation (`port 8080`), the Python service handles
RAG/chatbot (`port 8081`), and the frontend talks to both. All services
start with a single `docker compose up`.

---

## Quickstart

### Prerequisites

- Docker & Docker Compose
- Node.js 18+ (for the frontend)
- An [OpenAI API key](https://platform.openai.com/api-keys) (for the chatbot)

### 1. Set your API key

Create a `.env` file at the project root:
```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

### 2. Start all backend services

```bash
docker compose up -d          # starts postgres + Rust server + Python RAG
```

This gives you:
- PostgreSQL on `localhost:5432`
- Rust simulation API on `localhost:8080`
- Python RAG service on `localhost:8081`

### 3. Populate the database (first run only)

```bash
docker compose --profile ingest up ingest
```

### 4. Start the frontend

```bash
cd seatcraft
npm install
npm run dev
# Open http://localhost:3000
```

### Development without Docker

If you prefer running services directly:

**Terminal 1 — Rust backend:**
```bash
cd sim_engine
cargo run --bin server
```

**Terminal 2 — Python RAG:**
```bash
cd rag_service
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8081
```

**Terminal 3 — Frontend:**
```bash
cd seatcraft
npm run dev
```

> With Docker Compose you only need **two terminals**: `docker compose up` + `npm run dev`.

---

## API Reference

### Simulation (Rust — port 8080)

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/health` | Health check |
| `POST` | `/api/simulate` | Run Monte Carlo prediction |

### RAG Chatbot (Python — port 8081)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/rag/upload` | Upload a PDF or image for indexing |
| `POST` | `/api/rag/ask` | Ask a question (with optional image) |
| `GET`  | `/api/rag/status` | Chunk count + indexed document names |
| `GET`  | `/health` | Liveness probe |

**Upload** — multipart form:
```bash
curl -F file=@brochure.pdf http://localhost:8081/api/rag/upload
```

**Ask** — JSON:
```bash
curl -X POST http://localhost:8081/api/rag/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What documents do I need for reporting?"}'
```

**Ask with screenshot** — include `image_b64` (base64 JPEG/PNG):
```json
{ "question": "What is my current allotment status?", "image_b64": "..." }
```

---

## Environment Variables

### Root `.env` (used by Docker Compose)

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key — passed to the RAG container |

### `sim_engine/.env` (Rust dev server)

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | — | PostgreSQL connection string |
| `PORT` | `8080` | HTTP server port |

### RAG service (set in docker-compose or environment)

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `RAG_LLM_MODEL` | `gpt-4o-mini` | Chat model (handles text + vision) |
| `RAG_EMBED_MODEL` | `text-embedding-3-small` | Embedding model |
| `RAG_LLM_TEMPERATURE` | `0.2` | LLM sampling temperature |
| `RAG_TOP_K` | `5` | Number of chunks retrieved per query |
| `RAG_CHUNK_SIZE` | `512` | Characters per chunk |
| `RAG_CHUNK_OVERLAP` | `64` | Overlap characters between chunks |

---

## RAG Features

- **LangChain pipeline** — uses LangChain 0.3.x for the full RAG workflow
- **FAISS vector store** — in-memory cosine similarity search (no external DB)
- **PDF indexing** — upload counselling brochures, rulebooks (≤10 pages recommended)
- **Image OCR** — screenshot your counselling portal status; GPT-4o-mini reads it
- **Contextual Q&A** — answers are grounded in your uploaded documents with source citations
- **In-memory store** — embeddings are held in RAM; re-upload documents after a restart
