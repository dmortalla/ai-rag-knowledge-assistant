"""
Base vector store interface for the RAG pipeline.

This module defines the contract that all vector store backends must
implement so the rest of the application can remain backend-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional


class BaseVectorStore(ABC):
    """
    Abstract base class for vector store backends.
    """

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """
        Indicate whether the vector store is initialized.

        Returns:
            True if initialized, otherwise False.
        """
        raise NotImplementedError

    @abstractmethod
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, object]]] = None,
    ) -> None:
        """
        Add texts and optional metadata to the vector store.

        Args:
        texts:
            Text chunks to index.
        metadatas:
            Metadata aligned one-to-one with texts.
        """
        raise NotImplementedError

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 4,
    ) -> List[Dict[str, object]]:
        """
        Search for the most relevant chunks.

        Args:
        query:
            Query string.
        k:
            Number of results to return.
        Returns:
            Retrieved chunk records.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self) -> Path:
        """
        Save the vector store using backend-specific persistence.

        Returns:
            Persistence location used by the backend.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self) -> None:
        """
        Load the vector store using backend-specific persistence.

        """
        raise NotImplementedError
