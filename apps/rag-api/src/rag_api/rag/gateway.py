"""Factories and response normalization for Gemini chat clients."""

from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from rag_api.config import Settings, settings
from rag_api.errors import ProviderUnavailableError, UpstreamProviderError


def create_chat_client(config: Settings = settings) -> Any:
    """Create the real Gemini chat client; tests inject their own fake."""
    if not config.api_key_value:
        raise ProviderUnavailableError()
    return ChatGoogleGenerativeAI(
        model=config.rag_llm_model,
        temperature=config.rag_llm_temperature,
        max_tokens=config.rag_llm_max_tokens,
        google_api_key=config.api_key_value,
        timeout=config.rag_provider_timeout_seconds,
        max_retries=1,
    )


def response_text(response: Any) -> str:
    """Normalize LangChain text/content blocks without exposing raw objects."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        text = "\n".join(parts)
    else:
        text = ""
    text = text.strip()
    if not text:
        raise UpstreamProviderError()
    return text
