"""
FastAPI application factory for the JoSAA Inference Service.

Creates the app, attaches CORS middleware, registers routes, and loads all
ML artifacts exactly once via the lifespan context manager.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from inference_api.config import settings
from inference_api.logging_config import configure_service_logging
from inference_api.routes import router
from inference_api.services.artifact_loader import ArtifactLoadError, load_all_artifacts

logger = configure_service_logging("inference_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load all artifacts. Shutdown: log."""
    logger.info("Loading ML artifacts at startup…")
    try:
        app.state.artifacts = load_all_artifacts()
        app.state.artifact_error = None
        logger.info(
            f"Inference service ready — universe_size={app.state.artifacts.universe_size}"
        )
    except (ArtifactLoadError, FileNotFoundError, ValueError) as exc:
        logger.error("Artifact loading failed: %s", exc)
        # Set artifacts to None — /health will report not ready
        app.state.artifacts = None
        app.state.artifact_error = str(exc)

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
