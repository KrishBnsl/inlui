"""
FastAPI application factory.

Creates the app, attaches CORS middleware, registers routes, and initialises
the RAG pipeline on startup via the lifespan context manager.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.rag import RagPipeline
from app.routes import router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create the RAG pipeline. Shutdown: log."""
    if not settings.openai_api_key:
        logger.warning(
            "OPENAI_API_KEY is not set — RAG endpoints will fail until configured."
        )

    app.state.pipeline = RagPipeline()
    logger.info("RAG service is ready.")
    yield
    logger.info("RAG service shutting down.")


def create_app() -> FastAPI:
    """Application factory — returns a fully configured FastAPI instance."""
    app = FastAPI(
        title="inlui RAG Service",
        description="LangChain-powered JoSAA counselling advisor",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    return app
