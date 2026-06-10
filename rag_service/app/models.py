"""Pydantic request/response schemas for the RAG API."""

from typing import Optional
from pydantic import BaseModel


# ── Requests ────────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    """POST /api/rag/ask body."""
    question: str
    image_b64: Optional[str] = None


# ── Responses ───────────────────────────────────────────────────────────────────

class SourceItem(BaseModel):
    """A single source citation returned with an answer."""
    source: str
    excerpt: str


class AskResponse(BaseModel):
    """Response from POST /api/rag/ask."""
    answer: str
    sources: list[SourceItem]


class UploadResponse(BaseModel):
    """Response from POST /api/rag/upload."""
    ok: bool
    chunks_added: int
    source: str


class StatusResponse(BaseModel):
    """Response from GET /api/rag/status."""
    chunk_count: int
    documents: list[str]
