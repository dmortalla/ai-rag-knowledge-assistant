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

    Parameters
    ----------
    embeddings : object, optional
        Embedding backend. If omitted, a default OpenAI embedder backend
        is created.

    Returns
    -------
    BaseVectorStore
        Configured vector store instance.

    Raises
    ------
    NotImplementedError
        If the configured backend is not implemented yet.
    """
    settings = get_settings()

    if embeddings is None:
        embedder = OpenAIEmbedder()
        embeddings = embedder._get_backend()

    if settings.vector_db == "faiss":
        return FAISSStore(embeddings=embeddings)

    if settings.vector_db == "pinecone":
        raise NotImplementedError(
            "Pinecone support is not implemented yet. "
            "The interface is ready for it."
        )

    raise ValueError(f"Unsupported VECTOR_DB: {settings.vector_db}")