from __future__ import annotations

from fastapi.testclient import TestClient

from rag_api import create_app
from rag_api.config import Settings
from rag_api.rag import RagPipeline
from tests.conftest import FakeChat, FakeEmbeddings, make_pdf


def test_retired_embedding_setting_migrates_to_supported_model():
    config = Settings(
        google_api_key=None,
        rag_embed_model="models/text-embedding-004",
        _env_file=None,
    )
    assert config.rag_embed_model == "models/gemini-embedding-2"


def test_missing_key_has_clear_degraded_health(rag_config):
    app = create_app(config=rag_config)
    with TestClient(app) as client:
        health = client.get("/health")
        ready = client.get("/ready")
    body = health.json()
    assert health.status_code == 200
    assert body["status"] == "degraded"
    assert body["ready"] is False
    assert body["google_api_key_configured"] is False
    assert body["missing"] == ["GOOGLE_API_KEY"]
    assert "GOOGLE_API_KEY" in " ".join(body["degraded_reasons"])
    assert ready.status_code == 503


def test_mocked_dependencies_make_api_ready_without_live_calls(rag_config):
    pipeline = RagPipeline(embeddings=FakeEmbeddings(), llm=FakeChat(), config=rag_config)
    app = create_app(lambda: pipeline, config=rag_config)
    with TestClient(app) as client:
        body = client.get("/health").json()
    assert body["ready"] is True
    assert body["embedding_provider_ready"] is True
    assert body["chat_provider_ready"] is True


def test_upload_and_ask_flow_uses_mocked_gateway(rag_config):
    chat = FakeChat()
    pipeline = RagPipeline(embeddings=FakeEmbeddings(), llm=chat, config=rag_config)
    app = create_app(lambda: pipeline, config=rag_config)
    with TestClient(app) as client:
        upload = client.post(
            "/api/rag/upload",
            files={"file": ("rules.pdf", make_pdf(), "application/pdf")},
        )
        answer = client.post(
            "/api/rag/ask",
            json={"question": "What does my document say?", "ml_context": "main_rank=3000"},
        )
        status = client.get("/api/rag/status")
    assert upload.status_code == 200
    assert upload.json()["chunks_added"] >= 1
    assert answer.status_code == 200
    assert answer.json()["answer"] == "Grounded mocked answer"
    assert answer.json()["sources"][0]["source"] == "rules.pdf"
    assert status.json()["documents"] == ["rules.pdf"]


def test_upload_validates_empty_size_mime_and_extension(rag_config):
    config = Settings(**{**rag_config.model_dump(), "rag_max_upload_bytes": 1024}, _env_file=None)
    pipeline = RagPipeline(embeddings=FakeEmbeddings(), llm=FakeChat(), config=config)
    app = create_app(lambda: pipeline, config=config)
    with TestClient(app) as client:
        empty = client.post(
            "/api/rag/upload",
            files={"file": ("rules.pdf", b"", "application/pdf")},
        )
        too_large = client.post(
            "/api/rag/upload",
            files={"file": ("rules.pdf", b"%PDF-" + b"x" * 1100, "application/pdf")},
        )
        wrong_mime = client.post(
            "/api/rag/upload",
            files={"file": ("rules.pdf", make_pdf(), "text/plain")},
        )
        unsupported = client.post(
            "/api/rag/upload",
            files={"file": ("rules.txt", b"hello", "text/plain")},
        )
    assert empty.status_code == 400
    assert "empty" in empty.json()["detail"].casefold()
    assert too_large.status_code == 400
    assert "size" in too_large.json()["detail"].casefold()
    assert wrong_mime.status_code == 415
    assert unsupported.status_code == 415


def test_question_validation_rejects_blank_and_extra_fields(rag_config):
    pipeline = RagPipeline(embeddings=FakeEmbeddings(), llm=FakeChat(), config=rag_config)
    app = create_app(lambda: pipeline, config=rag_config)
    with TestClient(app) as client:
        blank = client.post("/api/rag/ask", json={"question": "   "})
        extra = client.post("/api/rag/ask", json={"question": "hi", "api_key": "secret"})
    assert blank.status_code == 422
    assert extra.status_code == 422


class ExplodingChat:
    def __init__(self, error):
        self.error = error

    def invoke(self, messages):
        raise self.error


def test_upstream_timeout_is_sanitized(rag_config):
    pipeline = RagPipeline(
        embeddings=FakeEmbeddings(),
        llm=ExplodingChat(TimeoutError("raw timeout host detail")),
        config=rag_config,
    )
    app = create_app(lambda: pipeline, config=rag_config)
    with TestClient(app) as client:
        response = client.post("/api/rag/ask", json={"question": "help"})
    assert response.status_code == 504
    assert response.json()["detail"] == "The upstream provider timed out."
    assert "host detail" not in response.text


def test_raw_upstream_error_and_secret_do_not_leak(rag_config, caplog):
    secret = "GOOGLE_API_KEY=do-not-leak"
    pipeline = RagPipeline(
        embeddings=FakeEmbeddings(),
        llm=ExplodingChat(RuntimeError(f"provider rejected {secret}")),
        config=rag_config,
    )
    app = create_app(lambda: pipeline, config=rag_config)
    with TestClient(app) as client:
        response = client.post("/api/rag/ask", json={"question": "help"})
    assert response.status_code == 502
    assert response.json()["detail"] == "The Gemini provider could not complete the request."
    assert secret not in response.text
    assert secret not in caplog.text
