"""Text splitter factory."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_api.config import Settings, settings


def create_text_splitter(config: Settings = settings) -> RecursiveCharacterTextSplitter:
    if config.rag_chunk_overlap >= config.rag_chunk_size:
        raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")
    return RecursiveCharacterTextSplitter(
        chunk_size=config.rag_chunk_size,
        chunk_overlap=config.rag_chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
