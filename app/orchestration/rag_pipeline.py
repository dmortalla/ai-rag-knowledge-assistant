"""
End-to-end orchestration for the RAG pipeline.

This module connects retrieval and generation into a single application-
level pipeline that accepts a query and returns a grounded answer along
with the retrieved context used to generate it.
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.generation.answer_generator import AnswerGenerator
from app.retrieval.retriever import Retriever


class RAGPipeline:
    """
    Orchestrate retrieval-augmented generation.

    Parameters
    ----------
    retriever : Retriever
        Retriever instance used to fetch relevant context chunks.
    answer_generator : AnswerGenerator
        Answer generator instance used to produce the final answer.
    """

    def __init__(
        self,
        retriever: Retriever,
        answer_generator: AnswerGenerator,
    ) -> None:
        """
        Initialize the RAG pipeline.

        Parameters
        ----------
        retriever : Retriever
            Retriever instance.
        answer_generator : AnswerGenerator
            Answer generator instance.
        """
        self.retriever = retriever
        self.answer_generator = answer_generator

    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the full RAG pipeline for a query.

        Parameters
        ----------
        query : str
            User query.

        Returns
        -------
        dict
            Dictionary containing the query, retrieved sources, and answer.

        Raises
        ------
        ValueError
            If the query is empty.
        """
        if not query or query.strip() == "":
            raise ValueError("Query cannot be empty.")

        retrieved_chunks = self.retriever.retrieve(query)
        answer = self.answer_generator.generate_answer(
            query=query,
            context_chunks=retrieved_chunks,
        )

        sources: List[Dict[str, object]] = []
        for chunk in retrieved_chunks:
            metadata = chunk.get("metadata", {})
            sources.append(
                {
                    "source": metadata.get("source", "unknown_source"),
                    "chunk_id": metadata.get("chunk_id", "unknown_chunk"),
                    "content": chunk.get("content", ""),
                }
            )

        return {
            "query": query,
            "context_chunks": [chunk["content"] for chunk in retrieved_chunks],
            "sources": sources,
            "answer": answer,
        }