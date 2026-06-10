"""Text chunking configuration using LangChain's RecursiveCharacterTextSplitter."""

from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.config import settings


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter tuned for counselling PDFs / brochures.

    Uses RecursiveCharacterTextSplitter which tries to split at paragraph
    boundaries first, then sentences, then words — preserving semantic
    coherence within each chunk.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
