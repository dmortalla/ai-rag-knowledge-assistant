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

        Returns
        -------
        bool
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

        Parameters
        ----------
        texts : list of str
            Text chunks to index.
        metadatas : list of dict, optional
            Metadata aligned one-to-one with texts.

        Returns
        -------
        None
        """
        raise NotImplementedError

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, object]]:
        """
        Search for the most relevant chunks.

        Parameters
        ----------
        query : str
            Query string.
        k : int, default=4
            Number of results to return.

        Returns
        -------
        list of dict
            Retrieved chunk records.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, folder_name: str = "faiss_index") -> Path:
        """
        Save the vector store.

        Parameters
        ----------
        folder_name : str, default="faiss_index"
            Save target name.

        Returns
        -------
        Path
            Save path.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, folder_name: str = "faiss_index") -> None:
        """
        Load the vector store.

        Parameters
        ----------
        folder_name : str, default="faiss_index"
            Load target name.

        Returns
        -------
        None
        """
        raise NotImplementedError