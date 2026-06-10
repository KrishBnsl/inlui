"""OpenAI embeddings wrapper — thin layer around LangChain's OpenAIEmbeddings."""

from langchain_openai import OpenAIEmbeddings

from app.config import settings


def create_embeddings() -> OpenAIEmbeddings:
    """Create an OpenAIEmbeddings instance using the configured model and API key."""
    return OpenAIEmbeddings(
        model=settings.rag_embed_model,
        openai_api_key=settings.openai_api_key,
    )
