"""
Retriever utilities for the RAG pipeline.

This module defines an application-level retriever that sits between
the orchestration layer and the vector store. It provides a clean,
stable interface for retrieving relevant text chunks for a query.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from app.core.config import get_settings


class Retriever:
    """
    Application-level retriever for vector search.

    Parameters
    ----------
    vector_store : object
        Vector store instance implementing a ``similarity_search`` method.
    default_k : int, optional
        Default number of results to retrieve.
    """

    def __init__(self, vector_store: object, default_k: Optional[int] = None) -> None:
        """
        Initialize the retriever.

        Parameters
        ----------
        vector_store : object
            Vector store instance.
        default_k : int, optional
            Default number of retrieved results.
        """
        settings = get_settings()

        self.vector_store = vector_store
        self.default_k = default_k if default_k is not None else settings.top_k

        assert self.default_k > 0, "default_k must be greater than 0."

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict[str, object]]:
        """
        Retrieve the most relevant text chunks for a query.

        Parameters
        ----------
        query : str
            Input query string.
        k : int, optional
            Number of top results to retrieve.

        Returns
        -------
        list of dict
            Retrieved chunk records with content and metadata.

        Raises
        ------
        ValueError
            If the query is empty or if ``k`` is invalid.
        """
        if not query or query.strip() == "":
            raise ValueError("Query cannot be empty.")

        effective_k = k if k is not None else self.default_k

        if effective_k <= 0:
            raise ValueError("k must be greater than 0.")

        return self.vector_store.similarity_search(query=query, k=effective_k)