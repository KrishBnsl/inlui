"""
Document ingestion — extracts text from PDFs and images, then chunks it.

Supported formats:
  - PDF  → PyPDFLoader (pure Python, no external deps)
  - PNG / JPG / JPEG / WEBP → Gemini vision OCR
"""

import base64
import logging
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger("rag_service.ingestion")

# Supported image extensions
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

_MIME_MAP = {
    "png": "image/png",
    "webp": "image/webp",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
}


def extract_text_from_pdf(content: bytes) -> str:
    """Extract text from a PDF using PyPDFLoader via a temp file."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        return "\n\n".join(page.page_content for page in pages)
    finally:
        os.unlink(tmp_path)


def extract_text_from_image(
    content: bytes,
    filename: str,
    llm: ChatGoogleGenerativeAI,
) -> str:
    """
    Extract text from an image using Gemini vision.

    Sends the image as a base64-encoded data URL and asks the model
    to perform OCR, returning all visible text.
    """
    ext = filename.rsplit(".", 1)[-1].lower()
    mime = _MIME_MAP.get(ext, "image/jpeg")
    b64 = base64.b64encode(content).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"

    msg = HumanMessage(content=[
        {
            "type": "text",
            "text": (
                "Extract ALL visible text from this image exactly as it appears — "
                "including labels, statuses, dates, names, rank numbers, institute "
                "names, document lists, and any other readable content. "
                "Output only the extracted text, nothing else."
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": data_url, "detail": "auto"},
        },
    ])
    response = llm.invoke([msg])
    return response.content


def ingest_file(
    content: bytes,
    filename: str,
    llm: ChatGoogleGenerativeAI,
    text_splitter: RecursiveCharacterTextSplitter,
) -> list[Document]:
    """
    Full ingestion pipeline for a single file:
      1. Extract raw text (PDF or image OCR)
      2. Chunk into overlapping segments
      3. Return LangChain Document objects with source metadata

    Raises ValueError for unsupported file types.
    """
    lower = filename.lower()

    # ── Step 1: Extract text ────────────────────────────────────────────────
    if lower.endswith(".pdf"):
        raw_text = extract_text_from_pdf(content)
    elif lower.endswith(_IMAGE_EXTENSIONS):
        raw_text = extract_text_from_image(content, filename, llm)
    else:
        raise ValueError(
            f"Unsupported file type for '{filename}'. "
            "Accepted: PDF, PNG, JPG, JPEG, WEBP."
        )

    raw_text = raw_text.strip()
    if not raw_text:
        logger.warning(f"Ingested '{filename}' but extracted no text")
        return []

    # ── Step 2: Chunk ───────────────────────────────────────────────────────
    docs = text_splitter.create_documents(
        texts=[raw_text],
        metadatas=[{"source": filename}],
    )

    logger.info(f"Ingesting '{filename}': {len(docs)} chunks")
    return docs
