"""Thread-safe, dependency-injected FAISS vector store."""

from __future__ import annotations

import logging
import hashlib
from dataclasses import dataclass
from threading import RLock
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from rag_api.errors import ProviderUnavailableError, VectorStoreUnavailableError

logger = logging.getLogger("rag_api.vector_store")


@dataclass(frozen=True)
class IndexedDoc:
    name: str
    chunks: int


class VectorStore:
    """Own an in-memory FAISS index and source-level metadata."""

    def __init__(self, embeddings: Any, faiss_class: Any = FAISS) -> None:
        self._embeddings = embeddings
        self._faiss_class = faiss_class
        self._store: Any | None = None
        self._indexed_docs: list[IndexedDoc] = []
        self._lock = RLock()
        try:
            import faiss  # noqa: F401
        except ImportError:
            self._faiss_runtime_available = False
        else:
            self._faiss_runtime_available = True

    @property
    def is_empty(self) -> bool:
        return self._store is None

    @property
    def is_available(self) -> bool:
        return self._embeddings is not None and self._faiss_runtime_available

    @property
    def chunk_count(self) -> int:
        with self._lock:
            return int(self._store.index.ntotal) if self._store is not None else 0

    @property
    def documents(self) -> list[str]:
        with self._lock:
            return sorted({document.name for document in self._indexed_docs})

    def add_documents(self, docs: list[Document], source_name: str) -> int:
        if not docs:
            return 0
        if self._embeddings is None:
            raise ProviderUnavailableError()
        if not self._faiss_runtime_available:
            raise VectorStoreUnavailableError()

        normalized_docs = []
        for index, doc in enumerate(docs):
            metadata = dict(doc.metadata)
            metadata["source"] = source_name
            metadata.setdefault("chunk_index", index)
            normalized_docs.append(Document(page_content=doc.page_content, metadata=metadata))

        with self._lock:
            new_store = self._faiss_class.from_documents(normalized_docs, self._embeddings)
            if self._store is None:
                self._store = new_store
            else:
                self._store.merge_from(new_store)
            self._indexed_docs.append(IndexedDoc(source_name, len(normalized_docs)))
        source_id = hashlib.sha256(source_name.encode("utf-8")).hexdigest()[:12]
        logger.info("Indexed document source_id=%s chunks=%s", source_id, len(normalized_docs))
        return len(normalized_docs)

    def similarity_search(self, query: str, k: int) -> list[Document]:
        if self._store is None:
            return []
        if not query.strip():
            return []
        with self._lock:
            return self._store.similarity_search(query, k=k)
