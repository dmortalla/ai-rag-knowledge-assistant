"""Pydantic schemas for the FastAPI layer.

This module defines the request and response contracts for the RAG API.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for a RAG query."""

    query: str = Field(..., min_length=1, description="User query text.")


class SourceRecord(BaseModel):
    """Source record returned by the RAG pipeline."""

    source: str
    chunk_id: int | str
    content: str


class QueryResponse(BaseModel):
    """Response body returned by the RAG pipeline."""

    query: str
    context_chunks: List[str]
    sources: List[SourceRecord]
    answer: str
