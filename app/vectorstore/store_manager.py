"""
Vector store factory and backend selection utilities.
"""

from __future__ import annotations

from app.core.config import get_settings
from app.embeddings.openai_embedder import OpenAIEmbedder
from app.vectorstore.base import BaseVectorStore
from app.vectorstore.faiss_store import FAISSStore


def get_vector_store(embeddings: object | None = None) -> BaseVectorStore:
    """
    Create the configured vector store backend.

    Args:
    embeddings:
        Embedding backend. If omitted, an OpenAI embedder is created.
    Returns:
        Configured vector store instance.
    Raises:
    NotImplementedError:
        If Pinecone is selected before its backend is implemented.
    ValueError:
        If the configured backend is unsupported.
    """
    settings = get_settings()

    if embeddings is None:
        embeddings = OpenAIEmbedder()

    if settings.vector_db == "faiss":
        return FAISSStore(embeddings=embeddings)

    if settings.vector_db == "pinecone":
        raise NotImplementedError(
            "Pinecone support is not implemented yet. "
            "The interface is ready for it."
        )

    raise ValueError(f"Unsupported VECTOR_DB: {settings.vector_db}")
