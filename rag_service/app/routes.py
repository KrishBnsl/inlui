"""
FastAPI route handlers for the RAG service.

All routes delegate to the RagPipeline instance stored in `app.state.pipeline`.
This keeps route handlers thin — they only handle HTTP concerns (parsing,
validation, error responses).
"""

import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.config import settings
from app.models import AskRequest, AskResponse, HealthResponse, StatusResponse, UploadResponse

logger = logging.getLogger("rag_service.routes")

router = APIRouter()


# ── Health ──────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    """Liveness probe."""
    pipeline = getattr(request.app.state, "pipeline", None)
    chunk_count = pipeline.store.chunk_count if pipeline is not None else 0
    key_configured = bool(settings.google_api_key)
    return HealthResponse(
        status="ok" if key_configured else "degraded",
        google_api_key_configured=key_configured,
        missing=[] if key_configured else ["GOOGLE_API_KEY"],
        vector_store_available=chunk_count > 0,
        chunk_count=chunk_count,
    )


# ── Upload ──────────────────────────────────────────────────────────────────────

@router.post("/api/rag/upload", response_model=UploadResponse)
async def upload_document(request: Request, file: UploadFile = File(...)):
    """Ingest a PDF or image file into the in-memory FAISS vector store."""
    pipeline = request.app.state.pipeline

    if not file.filename:
        raise HTTPException(400, "No filename provided")

    filename = file.filename
    content = await file.read()

    if not content:
        raise HTTPException(400, "Empty file")

    try:
        chunks_added = pipeline.ingest(content, filename)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception(f"Ingestion failed for '{filename}'")
        raise HTTPException(500, f"Ingestion failed: {e}")

    return UploadResponse(ok=True, chunks_added=chunks_added, source=filename)


# ── Ask ─────────────────────────────────────────────────────────────────────────

@router.post("/api/rag/ask", response_model=AskResponse)
async def ask_question(request: Request, req: AskRequest):
    """Answer a question using retrieved context from the vector store."""
    pipeline = request.app.state.pipeline

    try:
        return pipeline.ask(req.question, image_b64=req.image_b64, ml_context=req.ml_context)
    except Exception as e:
        logger.exception("Failed to answer question")
        raise HTTPException(500, f"Failed to generate answer: {e}")


# ── Status ──────────────────────────────────────────────────────────────────────

@router.get("/api/rag/status", response_model=StatusResponse)
async def rag_status(request: Request):
    """Return chunk count and indexed document names."""
    pipeline = request.app.state.pipeline
    return StatusResponse(
        chunk_count=pipeline.store.chunk_count,
        documents=pipeline.store.documents,
        vector_store_available=pipeline.store.chunk_count > 0,
        persistence="in_memory",
    )
