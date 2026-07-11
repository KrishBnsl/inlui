"""Validated request and response contracts for the RAG API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2_000)
    image_b64: str | None = Field(default=None, max_length=8_000_000)
    image_mime: Literal["image/png", "image/jpeg", "image/webp"] | None = None
    ml_context: str | None = Field(default=None, max_length=100_000)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class SourceItem(BaseModel):
    source: str
    excerpt: str
    page: int | None = None
    chunk_index: int | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


class UploadResponse(BaseModel):
    ok: bool
    chunks_added: int
    source: str


class StatusResponse(BaseModel):
    chunk_count: int
    documents: list[str]
    vector_store_available: bool
    persistence: str = "in_memory"


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    ready: bool
    google_api_key_configured: bool
    missing: list[str] = Field(default_factory=list)
    embedding_provider_ready: bool
    chat_provider_ready: bool
    vector_store_available: bool
    chunk_count: int
    degraded_reasons: list[str] = Field(default_factory=list)
    configured_models: dict[str, str] = Field(default_factory=dict)
