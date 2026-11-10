"""Dependency-injected RAG pipeline facade."""

from __future__ import annotations

import logging
from typing import Any

from rag_api.config import Settings, settings
from rag_api.errors import ProviderUnavailableError
from rag_api.models import AskResponse
from rag_api.rag.embeddings import create_embeddings
from rag_api.rag.gateway import create_chat_client
from rag_api.rag.ingestion import ingest_file
from rag_api.rag.retrieval import retrieve_and_answer
from rag_api.rag.text_splitter import create_text_splitter
from rag_api.rag.vector_store import VectorStore

logger = logging.getLogger("rag_api.pipeline")
_AUTO = object()


class RagPipeline:
    """Wire extraction, embeddings, FAISS retrieval, and chat generation."""

    def __init__(
        self,
        *,
        embeddings: Any = _AUTO,
        llm: Any = _AUTO,
        text_splitter: Any | None = None,
        vector_store: VectorStore | None = None,
        config: Settings = settings,
    ) -> None:
        self.config = config
        self.initialization_errors: list[str] = []

        if embeddings is _AUTO:
            try:
                embeddings = create_embeddings(config)
            except ProviderUnavailableError:
                embeddings = None
                self.initialization_errors.append("GOOGLE_API_KEY is not configured")
            except Exception:
                embeddings = None
                self.initialization_errors.append("Gemini embedding client could not initialize")
        if llm is _AUTO:
            try:
                llm = create_chat_client(config)
            except ProviderUnavailableError:
                llm = None
                if "GOOGLE_API_KEY is not configured" not in self.initialization_errors:
                    self.initialization_errors.append("GOOGLE_API_KEY is not configured")
            except Exception:
                llm = None
                self.initialization_errors.append("Gemini chat client could not initialize")

        self.embeddings = embeddings
        self.llm = llm
        self.text_splitter = text_splitter or create_text_splitter(config)
        self.store = vector_store or VectorStore(embeddings)
        if not self.store.is_available:
            if embeddings is None:
                self.initialization_errors.append("Embedding provider is unavailable")
            else:
                self.initialization_errors.append("FAISS runtime is unavailable")

    @property
    def embedding_ready(self) -> bool:
        return self.embeddings is not None and self.store.is_available

    @property
    def chat_ready(self) -> bool:
        return self.llm is not None

    @property
    def ready(self) -> bool:
        return self.embedding_ready and self.chat_ready

    def ingest(self, content: bytes, filename: str, content_type: str) -> int:
        if not self.embedding_ready:
            raise ProviderUnavailableError()
        documents = ingest_file(
            content=content,
            filename=filename,
            content_type=content_type,
            llm=self.llm,
            text_splitter=self.text_splitter,
            max_bytes=self.config.rag_max_upload_bytes,
        )
        source = documents[0].metadata["source"] if documents else filename
        return self.store.add_documents(documents, source_name=source)

    def ask(
        self,
        question: str,
        image_b64: str | None = None,
        image_mime: str | None = None,
        ml_context: str | None = None,
    ) -> AskResponse:
        return retrieve_and_answer(
            question=question,
            vector_store=self.store,
            llm=self.llm,
            image_b64=image_b64,
            image_mime=image_mime,
            ml_context=ml_context,
            config=self.config,
        )
