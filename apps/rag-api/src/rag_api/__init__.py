"""FastAPI application factory with injectable RAG dependencies."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_api.config import Settings, settings
from rag_api.logging_config import configure_service_logging
from rag_api.rag import RagPipeline
from rag_api.routes import router

logger = configure_service_logging("rag_api")


def create_app(
    pipeline_factory: Callable[[], RagPipeline] | None = None,
    config: Settings = settings,
) -> FastAPI:
    """Create an app; tests can supply a fully mocked pipeline factory."""
    factory = pipeline_factory or (lambda: RagPipeline(config=config))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.rag_settings = config
        app.state.pipeline_error = None
        try:
            app.state.pipeline = factory()
        except Exception as exc:
            app.state.pipeline = None
            app.state.pipeline_error = f"Pipeline initialization failed ({type(exc).__name__})"
            logger.error("RAG pipeline initialization failed error_type=%s", type(exc).__name__)
        yield
        logger.info("RAG service shutting down")

    app = FastAPI(
        title="inlui RAG Service",
        description="Grounded JoSAA counselling advisor using FAISS and Gemini",
        version="1.1.0",
        lifespan=lifespan,
    )
    app.state.rag_settings = config
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app
