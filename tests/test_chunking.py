"""Unit tests for chunking functionality."""

import pytest

from app.processing.chunking import chunk_text


def test_chunking_basic():
    text = "a" * 1000
    chunks = chunk_text(text)

    assert len(chunks) > 1
    assert all(isinstance(c, str) for c in chunks)


def test_chunking_empty():
    with pytest.raises(ValueError):
        chunk_text("")


def test_chunking_overlap():
    text = "abcdefghijklmnopqrstuvwxyz" * 50
    chunks = chunk_text(text)

    # Ensure overlap exists
    assert len(chunks) > 1
    assert chunks[0][-10:] in chunks[1]
