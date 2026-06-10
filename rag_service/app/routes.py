"""
FastAPI route handlers for the RAG service.

All routes delegate to the RagPipeline instance stored in `app.state.pipeline`.
This keeps route handlers thin — they only handle HTTP concerns (parsing,
validation, error responses).
"""

import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.models import AskRequest, AskResponse, StatusResponse, UploadResponse

logger = logging.getLogger("rag_service.routes")

router = APIRouter()


# ── Health ──────────────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    """Liveness probe."""
    return {"status": "ok"}


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
        return pipeline.ask(req.question, image_b64=req.image_b64)
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
    )
