"""
Text chunking utilities for RAG pipeline.

This module splits documents into smaller overlapping chunks
to improve retrieval quality in vector search.
"""

from __future__ import annotations

from typing import List

from app.core.config import get_settings


def chunk_text(text: str) -> List[str]:
    """
    Split input text into overlapping chunks.

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    List[str]
        List of text chunks.

    Raises
    ------
    ValueError
        If input text is empty.
    """
    settings = get_settings()

    if not text or text.strip() == "":
        raise ValueError("Input text cannot be empty.")

    chunk_size = settings.chunk_size
    overlap = settings.chunk_overlap

    chunks: List[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks