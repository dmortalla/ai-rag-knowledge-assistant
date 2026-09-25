"""Unit tests for the retriever module.

These tests validate query handling, default top-k behavior, and
retrieval output without depending on a real vector database.
"""

from __future__ import annotations

import pytest

from app.retrieval.retriever import Retriever


class DummyVectorStore:
    """Dummy vector store used for deterministic retriever tests."""

    def similarity_search(self, query: str, k: int = 4):
        """Return deterministic mock retrieval results.

        Args:
            query: Input query string.
            k: Number of results to return.

        Returns:
            Mock retrieved results.
        """
        return [f"{query}_result_{index}" for index in range(k)]


def test_retrieve_uses_default_k():
    """Test that retrieval uses the retriever's default top-k value."""
    retriever = Retriever(vector_store=DummyVectorStore(), default_k=3)

    results = retriever.retrieve("apple")

    assert len(results) == 3
    assert results[0] == "apple_result_0"


def test_retrieve_respects_explicit_k():
    """Test that retrieval respects an explicit k override."""
    retriever = Retriever(vector_store=DummyVectorStore(), default_k=3)

    results = retriever.retrieve("banana", k=2)

    assert len(results) == 2
    assert results[1] == "banana_result_1"


def test_retrieve_raises_for_empty_query():
    """Test that an empty query raises ValueError."""
    retriever = Retriever(vector_store=DummyVectorStore(), default_k=3)

    with pytest.raises(ValueError):
        retriever.retrieve("")


def test_retrieve_raises_for_invalid_k():
    """Test that a non-positive k raises ValueError."""
    retriever = Retriever(vector_store=DummyVectorStore(), default_k=3)

    with pytest.raises(ValueError):
        retriever.retrieve("grape", k=0)


def test_init_raises_for_invalid_default_k():
    """Test that an invalid default_k fails at initialization."""
    with pytest.raises(AssertionError):
        Retriever(vector_store=DummyVectorStore(), default_k=0)
