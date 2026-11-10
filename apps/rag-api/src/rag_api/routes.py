"""Thin HTTP adapters with timeouts and sanitized error responses."""

from __future__ import annotations

import asyncio
import logging
from functools import partial

from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile
from starlette.concurrency import run_in_threadpool

from rag_api.config import Settings, settings
from rag_api.errors import (
    DocumentExtractionError,
    DocumentValidationError,
    ProviderUnavailableError,
    RagServiceError,
    UnsupportedDocumentError,
    UpstreamProviderError,
    VectorStoreUnavailableError,
)
from rag_api.models import AskRequest, AskResponse, HealthResponse, StatusResponse, UploadResponse
from rag_api.rag.ingestion import decode_inline_image, safe_filename

logger = logging.getLogger("rag_api.routes")
router = APIRouter()


def _config(request: Request) -> Settings:
    return getattr(request.app.state, "rag_settings", settings)


def _pipeline(request: Request):
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="The RAG pipeline is not initialized.")
    return pipeline


def _log_sanitized(operation: str, exc: BaseException) -> None:
    logger.error("RAG operation failed operation=%s error_type=%s", operation, type(exc).__name__)


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    config = _config(request)
    pipeline = getattr(request.app.state, "pipeline", None)
    key_configured = bool(config.api_key_value)
    embedding_ready = bool(pipeline and pipeline.embedding_ready)
    chat_ready = bool(pipeline and pipeline.chat_ready)
    vector_available = bool(pipeline and pipeline.store.is_available)
    ready = embedding_ready and chat_ready and vector_available
    reasons = list(getattr(pipeline, "initialization_errors", [])) if pipeline else []
    pipeline_error = getattr(request.app.state, "pipeline_error", None)
    if pipeline_error:
        reasons.append(pipeline_error)
    if not key_configured and not ready and "GOOGLE_API_KEY is not configured" not in reasons:
        reasons.append("GOOGLE_API_KEY is not configured")
    reasons = list(dict.fromkeys(reasons))
    return HealthResponse(
        status="ok" if ready else "degraded",
        ready=ready,
        google_api_key_configured=key_configured,
        missing=[] if key_configured or ready else ["GOOGLE_API_KEY"],
        embedding_provider_ready=embedding_ready,
        chat_provider_ready=chat_ready,
        vector_store_available=vector_available,
        chunk_count=pipeline.store.chunk_count if pipeline else 0,
        degraded_reasons=reasons,
        configured_models={
            "embedding": config.rag_embed_model,
            "chat": config.rag_llm_model,
        },
    )


@router.get("/ready", response_model=HealthResponse)
async def readiness(request: Request, response: Response) -> HealthResponse:
    result = await health(request)
    if not result.ready:
        response.status_code = 503
    return result


@router.post("/api/rag/upload", response_model=UploadResponse)
async def upload_document(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    pipeline = _pipeline(request)
    config = _config(request)
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    try:
        source_name = safe_filename(file.filename)
    except DocumentValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.public_message) from None
    content = await file.read(config.rag_max_upload_bytes + 1)
    try:
        operation = partial(
            pipeline.ingest,
            content,
            source_name,
            file.content_type or "",
        )
        chunks_added = await asyncio.wait_for(
            run_in_threadpool(operation),
            timeout=config.rag_request_timeout_seconds,
        )
    except UnsupportedDocumentError as exc:
        raise HTTPException(status_code=415, detail=exc.public_message) from None
    except (DocumentValidationError, DocumentExtractionError) as exc:
        raise HTTPException(status_code=400, detail=exc.public_message) from None
    except ProviderUnavailableError as exc:
        raise HTTPException(status_code=503, detail=exc.public_message) from None
    except VectorStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=exc.public_message) from None
    except TimeoutError as exc:
        _log_sanitized("upload", exc)
        raise HTTPException(status_code=504, detail="The upstream provider timed out.") from None
    except UpstreamProviderError as exc:
        _log_sanitized("upload", exc)
        raise HTTPException(status_code=502, detail=exc.public_message) from None
    except Exception as exc:
        _log_sanitized("upload", exc)
        raise HTTPException(status_code=500, detail="Document ingestion failed.") from None
    return UploadResponse(ok=True, chunks_added=chunks_added, source=source_name)


@router.post("/api/rag/ask", response_model=AskResponse)
async def ask_question(request: Request, req: AskRequest) -> AskResponse:
    pipeline = _pipeline(request)
    config = _config(request)
    image_b64 = req.image_b64
    image_mime = req.image_mime
    if image_b64:
        try:
            image_b64, image_mime = decode_inline_image(
                image_b64,
                image_mime,
                config.rag_max_inline_image_bytes,
            )
        except DocumentValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.public_message) from None
    if req.ml_context and len(req.ml_context) > config.rag_max_ml_context_chars:
        raise HTTPException(status_code=422, detail="ML context exceeds the configured size limit.")

    try:
        operation = partial(
            pipeline.ask,
            req.question,
            image_b64=image_b64,
            image_mime=image_mime,
            ml_context=req.ml_context,
        )
        return await asyncio.wait_for(
            run_in_threadpool(operation),
            timeout=config.rag_request_timeout_seconds,
        )
    except ProviderUnavailableError as exc:
        raise HTTPException(status_code=503, detail=exc.public_message) from None
    except TimeoutError as exc:
        _log_sanitized("ask", exc)
        raise HTTPException(status_code=504, detail="The upstream provider timed out.") from None
    except UpstreamProviderError as exc:
        _log_sanitized("ask", exc)
        raise HTTPException(status_code=502, detail=exc.public_message) from None
    except RagServiceError as exc:
        _log_sanitized("ask", exc)
        raise HTTPException(status_code=500, detail=exc.public_message) from None
    except Exception as exc:
        _log_sanitized("ask", exc)
        raise HTTPException(status_code=500, detail="Answer generation failed.") from None


@router.get("/api/rag/status", response_model=StatusResponse)
async def rag_status(request: Request) -> StatusResponse:
    pipeline = getattr(request.app.state, "pipeline", None)
    return StatusResponse(
        chunk_count=pipeline.store.chunk_count if pipeline else 0,
        documents=pipeline.store.documents if pipeline else [],
        vector_store_available=bool(pipeline and pipeline.store.is_available),
        persistence="in_memory",
    )
