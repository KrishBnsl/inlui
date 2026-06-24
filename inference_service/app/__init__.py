"""
FastAPI application factory for the JoSAA Inference Service.

Creates the app, attaches CORS middleware, registers routes, and loads all
ML artifacts exactly once via the lifespan context manager.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes import router
from app.services.artifact_loader import load_all_artifacts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("inference_service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load all artifacts. Shutdown: log."""
    logger.info("Loading ML artifacts at startup…")
    try:
        app.state.artifacts = load_all_artifacts()
        logger.info(
            f"Inference service ready — universe_size={app.state.artifacts.universe_size}"
        )
    except FileNotFoundError as exc:
        logger.error(f"Artifact loading failed: {exc}")
        # Set artifacts to None — /health will report not ready
        app.state.artifacts = None

    yield

    logger.info("Inference service shutting down.")


def create_app() -> FastAPI:
    """Application factory — returns a fully configured FastAPI instance."""
    app = FastAPI(
        title="inlui Inference Service",
        description=(
            "ML inference layer for JoSAA counselling — "
            "rank prediction, Monte Carlo uncertainty, and recommendation ranking."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")

    return app
