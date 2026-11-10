"""Vector retrieval and grounded Gemini answer generation."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from rag_api.config import Settings, settings
from rag_api.errors import ProviderUnavailableError, RagServiceError, UpstreamProviderError
from rag_api.models import AskResponse, SourceItem
from rag_api.rag.gateway import response_text
from rag_api.rag.prompts import SYSTEM_PROMPT, build_user_prompt_with_ml_context
from rag_api.rag.vector_store import VectorStore


def _excerpt(text: str, limit: int = 300) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = " ".join(text.split())
    return text[:limit] + ("..." if len(text) > limit else "")


def retrieve_and_answer(
    question: str,
    vector_store: VectorStore,
    llm: Any,
    image_b64: str | None = None,
    image_mime: str | None = None,
    ml_context: str | None = None,
    config: Settings = settings,
) -> AskResponse:
    """Retrieve top-k chunks and invoke the injected chat gateway once."""
    if llm is None:
        raise ProviderUnavailableError()
    if ml_context and len(ml_context) > config.rag_max_ml_context_chars:
        ml_context = ml_context[: config.rag_max_ml_context_chars]

    try:
        results = vector_store.similarity_search(question, k=config.rag_top_k)
    except TimeoutError:
        raise
    except RagServiceError:
        raise
    except Exception as exc:
        raise UpstreamProviderError() from exc

    context_texts: list[str] = []
    sources: list[SourceItem] = []
    for document in results:
        context_texts.append(document.page_content)
        metadata = document.metadata or {}
        page = metadata.get("page")
        chunk_index = metadata.get("chunk_index")
        sources.append(
            SourceItem(
                source=str(metadata.get("source", "unknown"))[:255],
                excerpt=_excerpt(document.page_content),
                page=int(page) if isinstance(page, int) else None,
                chunk_index=int(chunk_index) if isinstance(chunk_index, int) else None,
            )
        )

    user_text = build_user_prompt_with_ml_context(context_texts, question, ml_context)
    if image_b64:
        mime = image_mime or "image/jpeg"
        user_message = HumanMessage(
            content=[
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{image_b64}",
                        "detail": "auto",
                    },
                },
            ]
        )
    else:
        user_message = HumanMessage(content=user_text)

    try:
        response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), user_message])
        answer = response_text(response)
    except TimeoutError:
        raise
    except RagServiceError:
        raise
    except Exception as exc:
        raise UpstreamProviderError() from exc
    return AskResponse(answer=answer, sources=sources)
