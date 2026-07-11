"""Environment-backed configuration for the RAG API."""

from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _discover_repository_root() -> Path:
    """Find a source checkout, with the container working directory as fallback."""

    for candidate in Path(__file__).resolve().parents:
        if (candidate / "docker-compose.yml").is_file():
            return candidate
    return Path.cwd().resolve()


REPO_ROOT = _discover_repository_root()
_SOURCE_SERVICE_ROOT = REPO_ROOT / "apps" / "rag-api"
SERVICE_ROOT = _SOURCE_SERVICE_ROOT if _SOURCE_SERVICE_ROOT.is_dir() else REPO_ROOT


class Settings(BaseSettings):
    """Validated settings with conservative request and provider limits."""

    google_api_key: SecretStr | None = None
    rag_embed_model: str = "models/gemini-embedding-2"
    rag_llm_model: str = "gemini-2.5-flash"
    rag_llm_temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    rag_llm_max_tokens: int = Field(default=1024, ge=1, le=8192)

    rag_chunk_size: int = Field(default=512, ge=100, le=4000)
    rag_chunk_overlap: int = Field(default=64, ge=0, le=1000)
    rag_top_k: int = Field(default=5, ge=1, le=20)

    rag_max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)
    rag_max_inline_image_bytes: int = Field(default=5 * 1024 * 1024, ge=1024)
    rag_request_timeout_seconds: float = Field(default=20.0, ge=1.0, le=120.0)
    rag_provider_timeout_seconds: float = Field(default=12.0, ge=1.0, le=60.0)
    rag_max_ml_context_chars: int = Field(default=20_000, ge=1000, le=100_000)

    port: int = 8081
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ]

    @field_validator("rag_embed_model", mode="before")
    @classmethod
    def migrate_retired_embedding_model(cls, value: object) -> object:
        """Keep an existing local env file usable after Google's model shutdown."""

        if value == "models/text-embedding-004":
            return "models/gemini-embedding-2"
        return value

    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", SERVICE_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def api_key_value(self) -> str:
        return self.google_api_key.get_secret_value().strip() if self.google_api_key else ""


settings = Settings()
