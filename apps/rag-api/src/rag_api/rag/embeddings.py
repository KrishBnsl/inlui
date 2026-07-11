"""Gemini embedding-client factory kept behind a dependency-injection seam."""

from typing import Any

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from rag_api.config import Settings, settings
from rag_api.errors import ProviderUnavailableError


def create_embeddings(config: Settings = settings) -> Any:
    """Create the real Gemini embedding client; never substitute a mock."""
    if not config.api_key_value:
        raise ProviderUnavailableError()
    return GoogleGenerativeAIEmbeddings(
        model=config.rag_embed_model,
        google_api_key=config.api_key_value,
        request_options={"timeout": config.rag_provider_timeout_seconds},
    )
