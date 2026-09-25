"""
Tests for the FAISS vector store abstraction.
"""

from __future__ import annotations

import pytest

from app.vectorstore.faiss_store import FAISSStore


class DummyEmbeddings:
    """
    Provide deterministic embeddings for vector store tests.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate deterministic embeddings for text documents.

        Parameters
        ----------
        texts : list of str
            Input text list.

        Returns
        -------
        list of list of float
            Deterministic embedding vectors.
        """
        return [
            [float(len(text)), float(text.count(" ")), 1.0]
            for text in texts
        ]

    def embed_query(self, query: str) -> list[float]:
        """
        Generate a deterministic embedding for a query.

        Parameters
        ----------
        query : str
            Query string.

        Returns
        -------
        list of float
            Deterministic query embedding.
        """
        return [
            float(len(query)),
            float(query.count(" ")),
            1.0,
        ]

    def __call__(self, text: str) -> list[float]:
        """
        Make the dummy embedding provider callable.

        Parameters
        ----------
        text : str
            Query text.

        Returns
        -------
        list of float
            Deterministic query embedding.
        """
        return self.embed_query(text)


def test_store_starts_uninitialized() -> None:
    """
    Verify a new FAISS store has no index.
    """
    store = FAISSStore(embeddings=DummyEmbeddings())

    assert store.is_initialized is False


def test_add_texts_initializes_store() -> None:
    """
    Verify adding valid texts creates the FAISS index.
    """
    store = FAISSStore(embeddings=DummyEmbeddings())

    store.add_texts(
        texts=["customer churn", "customer retention"],
        metadatas=[
            {"source": "test.txt", "chunk_id": 1},
            {"source": "test.txt", "chunk_id": 2},
        ],
    )

    assert store.is_initialized is True


def test_add_texts_rejects_empty_input() -> None:
    """
    Verify empty text input is rejected.
    """
    store = FAISSStore(embeddings=DummyEmbeddings())

    with pytest.raises(ValueError, match="Input texts cannot be empty"):
        store.add_texts([])


def test_add_texts_rejects_metadata_length_mismatch() -> None:
    """
    Verify metadata must align one-to-one with input texts.
    """
    store = FAISSStore(embeddings=DummyEmbeddings())

    with pytest.raises(ValueError, match="metadatas length must match"):
        store.add_texts(
            texts=["first", "second"],
            metadatas=[{"source": "test.txt"}],
        )


def test_similarity_search_returns_structured_records() -> None:
    """
    Verify retrieval returns content and metadata records.
    """
    store = FAISSStore(embeddings=DummyEmbeddings())
    store.add_texts(
        texts=["customer churn", "customer retention"],
        metadatas=[
            {"source": "test.txt", "chunk_id": 1},
            {"source": "test.txt", "chunk_id": 2},
        ],
    )

    results = store.similarity_search("customer churn", k=1)

    assert len(results) == 1
    assert "content" in results[0]
    assert "metadata" in results[0]
    assert results[0]["metadata"]["source"] == "test.txt"


def test_similarity_search_rejects_blank_query() -> None:
    """
    Verify blank retrieval queries are rejected.
    """
    store = FAISSStore(embeddings=DummyEmbeddings())

    with pytest.raises(ValueError, match="Query cannot be empty"):
        store.similarity_search("   ")


def test_similarity_search_rejects_nonpositive_k() -> None:
    """
    Verify retrieval count must be greater than zero.
    """
    store = FAISSStore(embeddings=DummyEmbeddings())
    store.add_texts(["customer churn"])

    with pytest.raises(ValueError, match="k must be greater than 0"):
        store.similarity_search("customer churn", k=0)


def test_similarity_search_requires_initialized_store() -> None:
    """
    Verify retrieval cannot run before an index exists.
    """
    store = FAISSStore(embeddings=DummyEmbeddings())

    with pytest.raises(
        ValueError,
        match="FAISS store has not been initialized",
    ):
        store.similarity_search("customer churn")


def test_blank_index_name_is_rejected() -> None:
    """
    Verify a FAISS persistence name cannot be blank.
    """
    with pytest.raises(ValueError, match="index_name cannot be blank"):
        FAISSStore(
            embeddings=DummyEmbeddings(),
            index_name="   ",
        )


def test_save_requires_initialized_store() -> None:
    """
    Verify an empty FAISS wrapper cannot be persisted.
    """
    store = FAISSStore(embeddings=DummyEmbeddings())

    with pytest.raises(
        ValueError,
        match="FAISS store has not been initialized",
    ):
        store.save()