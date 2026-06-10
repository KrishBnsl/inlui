"""
Configuration loaded from environment variables via pydantic-settings.
All variables have safe defaults so the service starts even without an API key.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """RAG service configuration — all values can be overridden via env vars."""

    # OpenAI
    openai_api_key: str = ""
    rag_embed_model: str = "text-embedding-3-small"
    rag_llm_model: str = "gpt-4o-mini"
    rag_llm_temperature: float = 0.2
    rag_llm_max_tokens: int = 1024

    # Chunking
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 64

    # Retrieval
    rag_top_k: int = 5

    # Server
    port: int = 8081
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:3001"]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Singleton — import this everywhere
settings = Settings()
