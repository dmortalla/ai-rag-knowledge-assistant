"""
FAISS vector store utilities for the RAG pipeline.

This module wraps LangChain's FAISS integration behind a clean
application-facing interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from langchain_community.vectorstores import FAISS

from app.core.paths import VECTOR_STORE_DIR
from app.vectorstore.base import BaseVectorStore


class FAISSStore(BaseVectorStore):
    """
    Wrapper around a FAISS vector store.

    Parameters
    ----------
    embeddings : object
        Embedding provider implementing compatible embedding methods.
    """

    def __init__(self, embeddings: object) -> None:
        """
        Initialize the FAISS store wrapper.

        Parameters
        ----------
        embeddings : object
            Embedding provider used by the FAISS index.
        """
        self.embeddings = embeddings
        self._store: Optional[FAISS] = None

    @property
    def is_initialized(self) -> bool:
        """
        Indicate whether the internal FAISS store exists.

        Returns
        -------
        bool
            True if the store has been created or loaded, otherwise False.
        """
        return self._store is not None

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, object]]] = None,
    ) -> None:
        """
        Create a FAISS store from input texts.

        Parameters
        ----------
        texts : list of str
            Text chunks to embed and index.
        metadatas : list of dict, optional
            Metadata dictionaries aligned one-to-one with texts.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the text list is empty or metadata length does not match.
        """
        if not texts:
            raise ValueError("Input texts cannot be empty.")

        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("metadatas length must match texts length.")

        self._store = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
        )

    def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, object]]:
        """
        Perform similarity search over the FAISS index.

        Parameters
        ----------
        query : str
            Query string used for retrieval.
        k : int, default=4
            Number of top matches to return.

        Returns
        -------
        list of dict
            Retrieved chunk records with content and metadata.

        Raises
        ------
        ValueError
            If the query is empty or the store is not initialized.
        """
        if not query or query.strip() == "":
            raise ValueError("Query cannot be empty.")

        if self._store is None:
            raise ValueError("FAISS store has not been initialized.")

        results = self._store.similarity_search(query=query, k=k)

        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in results
        ]

    def save(self, folder_name: str = "faiss_index") -> Path:
        """
        Save the FAISS store locally.

        Parameters
        ----------
        folder_name : str, default="faiss_index"
            Subfolder name inside the vector store directory.

        Returns
        -------
        Path
            Path to the saved FAISS directory.

        Raises
        ------
        ValueError
            If the store has not been initialized.
        """
        if self._store is None:
            raise ValueError("FAISS store has not been initialized.")

        save_path = VECTOR_STORE_DIR / folder_name
        save_path.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(save_path))

        return save_path

    def load(self, folder_name: str = "faiss_index") -> None:
        """
        Load a FAISS store from local disk.

        Parameters
        ----------
        folder_name : str, default="faiss_index"
            Subfolder name inside the vector store directory.

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If the requested save directory does not exist.
        """
        load_path = VECTOR_STORE_DIR / folder_name

        if not load_path.exists():
            raise FileNotFoundError(
                f"FAISS index directory does not exist: {load_path}"
            )

        self._store = FAISS.load_local(
            folder_path=str(load_path),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )