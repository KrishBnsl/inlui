"""
Retrieval + answer generation.

Orchestrates the full RAG flow:
  1. Retrieve top-k chunks from the vector store
  2. Build a prompt with context + user question
  3. Call the LLM (text-only or multimodal with an attached image)
  4. Return the answer with source citations
"""

import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage

from app.config import settings
from app.models import AskResponse, SourceItem
from app.rag.prompts import SYSTEM_PROMPT, build_user_prompt
from app.rag.vector_store import VectorStore

logger = logging.getLogger("rag_service.retrieval")


def retrieve_and_answer(
    question: str,
    vector_store: VectorStore,
    llm: ChatGoogleGenerativeAI,
    image_b64: str | None = None,
) -> AskResponse:
    """
    Full retrieval-augmented generation pipeline:
      1. Similarity search against the vector store
      2. Build a prompt with the retrieved excerpts
      3. Call GPT-4o-mini (with optional inline image)
      4. Return structured response with sources

    Parameters
    ----------
    question : str
        The user's natural-language question.
    vector_store : VectorStore
        The FAISS-backed store to search.
    llm : ChatGoogleGenerativeAI
        The chat model to use for answer generation.
    image_b64 : str | None
        Optional base64-encoded image (JPEG/PNG) attached by the user.
    """

    # ── Step 1: Retrieve ────────────────────────────────────────────────────
    context_texts: list[str] = []
    sources: list[SourceItem] = []

    results = vector_store.similarity_search(question, k=settings.rag_top_k)
    for doc in results:
        context_texts.append(doc.page_content)
        excerpt = doc.page_content[:200]
        if len(doc.page_content) > 200:
            excerpt += "…"
        sources.append(SourceItem(
            source=doc.metadata.get("source", "unknown"),
            excerpt=excerpt,
        ))

    # ── Step 2: Build prompt ────────────────────────────────────────────────
    system_msg = SystemMessage(content=SYSTEM_PROMPT)
    user_text = build_user_prompt(context_texts, question)

    # ── Step 3: Call LLM ────────────────────────────────────────────────────
    if image_b64:
        # Multimodal: text + inline image
        user_msg = HumanMessage(content=[
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{image_b64}",
                "detail": "auto",
            }},
        ])
    else:
        user_msg = HumanMessage(content=user_text)

    logger.info(f"Answering: '{question[:80]}…' with {len(sources)} sources")
    response = llm.invoke([system_msg, user_msg])

    # ── Step 4: Return ──────────────────────────────────────────────────────
    return AskResponse(
        answer=response.content,
        sources=sources,
    )
