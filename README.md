# inlui — JoSAA Counselling Platform

An end-all destination for correct JEE counselling decisions.

## Architecture

```
inlui/
├── sim_engine/   Rust backend (Axum HTTP server)
│   └── src/
│       ├── bin/server.rs   — HTTP API (simulate + RAG endpoints)
│       ├── bin/ingest.rs   — one-shot CSV → PostgreSQL loader
│       ├── predictor/      — Monte Carlo seat allotment engine
│       └── rag/            — RAG pipeline (embeddings, ingestion, retrieval)
│
└── seatcraft/    Next.js frontend
    └── src/
        ├── app/page.tsx    — main JoSAA planner UI
        └── components/
            ├── ChatAdvisor.tsx  — counselling chatbot drawer
            └── ...
```

Both the simulation engine **and** the RAG chatbot run inside the same Rust
process — no second backend, no second terminal.

---

## Quickstart

### Prerequisites

- Docker & Docker Compose
- An [OpenAI API key](https://platform.openai.com/api-keys) (for the chatbot)

### 1. Set your API key

Edit `sim_engine/.env`:
```
OPENAI_API_KEY=sk-...
```

### 2. Start infrastructure

```bash
docker compose up postgres -d          # start PostgreSQL
```

### 3. Populate the database (first run only)

```bash
docker compose --profile ingest up ingest
```

### 4. Start the backend

```bash
cd sim_engine
cargo run --bin server
# Listening on http://0.0.0.0:8080
```

### 5. Start the frontend

```bash
cd seatcraft
npm install
npm run dev
# Open http://localhost:3000
```

### Full Docker stack

```bash
docker compose up           # postgres + server
# frontend is run locally with: cd seatcraft && npm run dev
```

---

## API Reference

### Simulation

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/health` | Health check |
| `POST` | `/api/simulate` | Run Monte Carlo prediction |

### RAG Chatbot

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/rag/upload` | Upload a PDF or image for indexing |
| `POST` | `/api/rag/ask` | Ask a question (with optional image) |
| `GET`  | `/api/rag/status` | Chunk count + indexed document names |

**Upload** — multipart form:
```bash
curl -F file=@brochure.pdf http://localhost:8080/api/rag/upload
```

**Ask** — JSON:
```bash
curl -X POST http://localhost:8080/api/rag/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What documents do I need for reporting?"}'
```

**Ask with screenshot** — include `image_b64` (base64 JPEG/PNG):
```json
{ "question": "What is my current allotment status?", "image_b64": "..." }
```

---

## Environment Variables (`sim_engine/.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | — | PostgreSQL connection string |
| `PORT` | `8080` | HTTP server port |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `RAG_LLM_MODEL` | `gpt-4o-mini` | Chat model (handles text + vision) |
| `RAG_EMBED_MODEL` | `text-embedding-3-small` | Embedding model |
| `RAG_LLM_TEMPERATURE` | `0.2` | LLM sampling temperature |
| `RAG_DATA_DIR` | `./uploads` | Directory for uploaded files |
| `RAG_TOP_K` | `5` | Number of chunks retrieved per query |
| `RAG_CHUNK_SIZE` | `512` | Characters per chunk |
| `RAG_CHUNK_OVERLAP` | `64` | Overlap characters between chunks |

---

## RAG Features

- **PDF indexing** — upload counselling brochures, rulebooks (≤10 pages recommended)
- **Image OCR** — screenshot your counselling portal status; GPT-4o-mini reads it
- **Contextual Q&A** — answers are grounded in your uploaded documents with source citations
- **In-memory store** — embeddings are held in RAM; re-upload documents after a restart
- **No second process** — the chatbot runs inside the same Axum server as the simulator
