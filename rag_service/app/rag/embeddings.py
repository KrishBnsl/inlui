"""Google GenAI embeddings wrapper."""

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import settings


def create_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Create a GoogleGenerativeAIEmbeddings instance using the configured model and API key."""
    return GoogleGenerativeAIEmbeddings(
        model=settings.rag_embed_model,
        google_api_key=settings.google_api_key,
    )
