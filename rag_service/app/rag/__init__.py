"""
RAG Pipeline — the top-level facade that wires together all sub-modules.

Usage:
    pipeline = RagPipeline()       # creates embeddings, LLM, splitter, store
    pipeline.ingest(bytes, name)   # ingest a document
    pipeline.ask(question)         # retrieve + answer
"""

import logging

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings
from app.models import AskResponse
from app.rag.embeddings import create_embeddings
from app.rag.ingestion import ingest_file
from app.rag.retrieval import retrieve_and_answer
from app.rag.text_splitter import create_text_splitter
from app.rag.vector_store import VectorStore

logger = logging.getLogger("rag_service.pipeline")


class RagPipeline:
    """
    Central facade for the RAG pipeline.

    Owns all LangChain components and the in-memory vector store.
    Injected into the FastAPI app via `app.state.pipeline`.
    """

    def __init__(self) -> None:
        logger.info("Initialising RAG pipeline…")

        self.embeddings = create_embeddings()
        self.llm = ChatGoogleGenerativeAI(
            model=settings.rag_llm_model,
            temperature=settings.rag_llm_temperature,
            max_tokens=settings.rag_llm_max_tokens,
            google_api_key=settings.google_api_key,
        )
        self.text_splitter = create_text_splitter()
        self.store = VectorStore(self.embeddings)

        logger.info(
            f"RAG pipeline ready — model={settings.rag_llm_model}, "
            f"embeddings={settings.rag_embed_model}"
        )

    def ingest(self, content: bytes, filename: str) -> int:
        """
        Ingest a single file (PDF or image) into the vector store.
        Returns the number of chunks added.
        """
        docs = ingest_file(content, filename, self.llm, self.text_splitter)
        if not docs:
            return 0
        return self.store.add_documents(docs, source_name=filename)

    def ask(
        self,
        question: str,
        image_b64: str | None = None,
    ) -> AskResponse:
        """
        Answer a question using retrieved context.
        Optionally accepts an inline image (base64) for multimodal queries.
        """
        return retrieve_and_answer(
            question=question,
            vector_store=self.store,
            llm=self.llm,
            image_b64=image_b64,
        )
