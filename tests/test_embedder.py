"""
Unit tests for the OpenAI embedding module.

These tests validate embedding generation behavior without making
real API calls by using a dummy embedder backend.
"""

from __future__ import annotations

import pytest

from app.embeddings.openai_embedder import OpenAIEmbedder


class DummyEmbedder:
    """
    Mock embedder used to simulate embedding responses.

    This avoids external API calls during testing.
    """

    def embed_documents(self, texts):
        """
        Generate dummy embeddings for multiple texts.

        Parameters
        ----------
        texts : list of str
            Input text list.

        Returns
        -------
        list of list of float
            Fixed-size dummy vectors.
        """
        return [[1.0] * 5 for _ in texts]

    def embed_query(self, query):
        """
        Generate a dummy embedding for a query.

        Parameters
        ----------
        query : str
            Input query string.

        Returns
        -------
        list of float
            Fixed-size dummy vector.
        """
        return [1.0] * 5


def test_embed_texts():
    """
    Test embedding generation for multiple texts.

    Returns
    -------
    None
    """
    embedder = OpenAIEmbedder(embedding_backend=DummyEmbedder())

    texts = ["hello", "world"]
    vectors = embedder.embed_texts(texts)

    assert len(vectors) == 2
    assert len(vectors[0]) == 5


def test_embed_query():
    """
    Test embedding generation for a single query.

    Returns
    -------
    None
    """
    embedder = OpenAIEmbedder(embedding_backend=DummyEmbedder())

    vector = embedder.embed_query("test query")

    assert isinstance(vector, list)
    assert len(vector) == 5


def test_empty_texts():
    """
    Test that embedding generation raises an error for empty input.

    Returns
    -------
    None
    """
    embedder = OpenAIEmbedder(embedding_backend=DummyEmbedder())

    with pytest.raises(ValueError):
        embedder.embed_texts([])


def test_empty_query():
    """
    Test that embedding generation raises an error for empty query.

    Returns
    -------
    None
    """
    embedder = OpenAIEmbedder(embedding_backend=DummyEmbedder())

    with pytest.raises(ValueError):
        embedder.embed_query("")