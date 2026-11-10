#!/usr/bin/env python3
"""Opt-in live Gemini verification with sanitized, non-content evidence only."""

from __future__ import annotations

import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv


PROVIDER = "Google Gemini"
DEFAULT_EMBED_MODEL = "models/gemini-embedding-2"
DEFAULT_CHAT_MODEL = "gemini-2.5-flash"
TIMEOUT_SECONDS = 10.0
RETIRED_EMBED_MODELS = {"models/text-embedding-004": DEFAULT_EMBED_MODEL}


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def evidence(
    kind: str,
    status: str,
    model: str,
    latency_ms: float,
    reason: str | None = None,
) -> None:
    reason_field = f" reason={reason}" if reason else ""
    print(
        f"LIVE GEMINI {kind} TEST: {status} "
        f"provider={PROVIDER} model={model} latency_ms={latency_ms:.0f} "
        f"timestamp={timestamp()}{reason_field}"
    )


def main() -> int:
    if os.getenv("RUN_LIVE_GEMINI_TEST") != "1":
        print("LIVE GEMINI TEST: NOT RUN (set RUN_LIVE_GEMINI_TEST=1)")
        return 0

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env", override=False)
    load_dotenv(repo_root / "apps" / "rag-api" / ".env", override=False)
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    configured_embed_model = os.getenv("RAG_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    embed_model = RETIRED_EMBED_MODELS.get(configured_embed_model, configured_embed_model)
    chat_model = os.getenv("RAG_LLM_MODEL", DEFAULT_CHAT_MODEL)
    if not api_key:
        print(
            f"LIVE GEMINI TEST: NOT RUN provider={PROVIDER} "
            f"embedding_model={embed_model} chat_model={chat_model} "
            f"latency_ms=0 timestamp={timestamp()} reason=GOOGLE_API_KEY_missing"
        )
        return 2

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    except ImportError:
        print(
            f"LIVE GEMINI TEST: FAILURE provider={PROVIDER} "
            f"embedding_model={embed_model} chat_model={chat_model} "
            f"latency_ms=0 timestamp={timestamp()} reason=dependency_unavailable"
        )
        return 1

    embedding_ok = False
    embedding_reason: str | None = None
    start = time.perf_counter()
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=embed_model,
            google_api_key=api_key,
            request_options={"timeout": TIMEOUT_SECONDS},
        )
        vector = embeddings.embed_query("Gemini connectivity verification")
        embedding_ok = bool(vector) and all(math.isfinite(float(value)) for value in vector)
    except Exception as exc:
        embedding_ok = False
        embedding_reason = type(exc).__name__
    embedding_latency = (time.perf_counter() - start) * 1000
    evidence(
        "EMBEDDING",
        "SUCCESS" if embedding_ok else "FAILURE",
        embed_model,
        embedding_latency,
        embedding_reason,
    )

    chat_ok = False
    chat_reason: str | None = None
    start = time.perf_counter()
    try:
        chat = ChatGoogleGenerativeAI(
            model=chat_model,
            temperature=0,
            max_tokens=128,
            google_api_key=api_key,
            timeout=TIMEOUT_SECONDS,
            max_retries=0,
        )
        response = chat.invoke("Reply with a short acknowledgement.")
        content = getattr(response, "content", "")
        if isinstance(content, list):
            content = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        chat_ok = isinstance(content, str) and bool(content.strip())
    except Exception as exc:
        chat_ok = False
        chat_reason = type(exc).__name__
    chat_latency = (time.perf_counter() - start) * 1000
    evidence(
        "CHAT",
        "SUCCESS" if chat_ok else "FAILURE",
        chat_model,
        chat_latency,
        chat_reason,
    )

    if embedding_ok and chat_ok:
        print(
            f"LIVE GEMINI TEST: SUCCESS provider={PROVIDER} "
            f"latency_ms={embedding_latency + chat_latency:.0f} timestamp={timestamp()}"
        )
        return 0
    print(
        f"LIVE GEMINI TEST: FAILURE provider={PROVIDER} "
        f"latency_ms={embedding_latency + chat_latency:.0f} timestamp={timestamp()}"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
