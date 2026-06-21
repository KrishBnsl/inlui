"""
FAISS-backed in-memory vector store.

Manages the lifecycle of the FAISS index — creation, merging new documents,
similarity search, and metadata tracking.
"""

import logging
from dataclasses import dataclass, field

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document

logger = logging.getLogger("rag_service.vector_store")


@dataclass
class IndexedDoc:
    """Metadata for a document that has been indexed."""
    name: str
    chunks: int


class VectorStore:
    """
    Thread-safe wrapper around a FAISS vector store.

    - No disk persistence — resets on restart.
    - Tracks which documents have been indexed and their chunk counts.
    """

    def __init__(self, embeddings: GoogleGenerativeAIEmbeddings) -> None:
        self._embeddings = embeddings
        self._store: FAISS | None = None
        self._indexed_docs: list[IndexedDoc] = []

    @property
    def is_empty(self) -> bool:
        return self._store is None

    @property
    def chunk_count(self) -> int:
        if self._store is None:
            return 0
        return self._store.index.ntotal

    @property
    def documents(self) -> list[str]:
        """Unique sorted list of indexed document names."""
        names = sorted({d.name for d in self._indexed_docs})
        return names

    def add_documents(self, docs: list[Document], source_name: str) -> int:
        """
        Embed and add documents to the store.

        Returns the number of chunks added.
        """
        if not docs:
            return 0

        if self._store is None:
            self._store = FAISS.from_documents(docs, self._embeddings)
        else:
            new_store = FAISS.from_documents(docs, self._embeddings)
            self._store.merge_from(new_store)

        self._indexed_docs.append(IndexedDoc(name=source_name, chunks=len(docs)))
        logger.info(f"Inserted {len(docs)} chunks from '{source_name}'")
        return len(docs)

    def similarity_search(self, query: str, k: int) -> list[Document]:
        """
        Return the top-k most similar documents to the query.
        Returns an empty list if the store is empty.
        """
        if self._store is None:
            return []
        return self._store.similarity_search(query, k=k)
