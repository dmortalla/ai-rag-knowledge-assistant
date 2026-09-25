"""Embedding utilities using OpenAI models.

This module converts text chunks into vector embeddings for storage
in a vector database such as FAISS.
"""

from __future__ import annotations

from typing import List, Optional

from openai import OpenAI

from app.core.config import get_settings


class OpenAIEmbedder:
    """Handle embedding generation using the OpenAI Python SDK.

    Args:
        embedding_backend: Prebuilt embedding backend used primarily for testing. If provided,
            it must implement ``embed_documents`` and ``embed_query``.
    """

    def __init__(self, embedding_backend: Optional[object] = None) -> None:
        """Initialize the embedder.

        Args:
            embedding_backend: Optional backend for dependency injection in tests.
        """
        settings = get_settings()

        self.api_key = settings.openai_api_key
        self.model = settings.embedding_model
        self._embedder = embedding_backend
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        """Return the OpenAI client, creating it if necessary.

        Returns:
            OpenAI client instance.
        """
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)

        return self._client

    def _get_backend(self) -> object:
        """Return the embedding backend for FAISS compatibility.

        Returns:
            Either the injected backend or this embedder instance.
        """
        return self._embedder if self._embedder is not None else self

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: Input text list.

        Returns:
            Embedding vectors.
        """
        return self.embed_texts(texts)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: Input text list.

        Returns:
            Embedding vectors.

        Raises:
            ValueError: If input is empty.
        """
        if not texts:
            raise ValueError("Input texts cannot be empty.")

        if self._embedder is not None:
            return self._embedder.embed_documents(texts)

        client = self._get_client()
        response = client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> List[float]:
        """Generate an embedding for a single query.

        Args:
            query: Input query string.

        Returns:
            Embedding vector.

        Raises:
            ValueError: If query is empty.
        """
        if not query or query.strip() == "":
            raise ValueError("Query cannot be empty.")

        if self._embedder is not None:
            return self._embedder.embed_query(query)

        client = self._get_client()
        response = client.embeddings.create(
            model=self.model,
            input=query,
        )
        return response.data[0].embedding

    def __call__(self, text: str) -> List[float]:
        """Make the embedder callable for FAISS compatibility.

        Args:
            text: Query text.

        Returns:
            Query embedding vector.
        """
        return self.embed_query(text)
