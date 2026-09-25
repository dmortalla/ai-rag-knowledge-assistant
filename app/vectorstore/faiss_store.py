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

    Args:
    embeddings:
        Embedding provider implementing compatible embedding methods.
    index_name:
        Directory name used to persist the FAISS index.
    """

    def __init__(
        self,
        embeddings: object,
        index_name: str = "faiss_index",
    ) -> None:
        """
        Initialize the FAISS store wrapper.

        Args:
        embeddings:
            Embedding provider used by the FAISS index.
        index_name:
            Directory name used to persist the FAISS index.
        Raises:
        ValueError:
            If the index name is blank.
        """
        if not index_name or not index_name.strip():
            raise ValueError("index_name cannot be blank.")

        self.embeddings = embeddings
        self.index_name = index_name
        self._store: Optional[FAISS] = None

    @property
    def is_initialized(self) -> bool:
        """
        Indicate whether the internal FAISS store exists.

        Returns:
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

        Args:
        texts:
            Text chunks to embed and index.
        metadatas:
            Metadata dictionaries aligned one-to-one with texts.
        Raises:
        ValueError:
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

    def similarity_search(
        self,
        query: str,
        k: int = 4,
    ) -> List[Dict[str, object]]:
        """
        Perform similarity search over the FAISS index.

        Args:
        query:
            Query string used for retrieval.
        k:
            Number of top matches to return.
        Returns:
            Retrieved chunk records with content and metadata.
        Raises:
        ValueError:
            If the query is empty, k is invalid, or the store is not initialized.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if k <= 0:
            raise ValueError("k must be greater than 0.")

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

    def save(self) -> Path:
        """
        Save the FAISS store locally.

        Returns:
            Path to the saved FAISS directory.
        Raises:
        ValueError:
            If the store has not been initialized.
        """
        if self._store is None:
            raise ValueError("FAISS store has not been initialized.")

        save_path = VECTOR_STORE_DIR / self.index_name
        save_path.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(save_path))

        return save_path

    def load(self) -> None:
        """
        Load a FAISS store from local disk.

        Raises:
        FileNotFoundError:
            If the requested save directory does not exist.
        """
        load_path = VECTOR_STORE_DIR / self.index_name

        if not load_path.exists():
            raise FileNotFoundError(
                f"FAISS index directory does not exist: {load_path}"
            )

        self._store = FAISS.load_local(
            folder_path=str(load_path),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )
