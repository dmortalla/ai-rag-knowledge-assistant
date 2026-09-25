"""Prompt construction utilities for the RAG pipeline.

This module builds grounded prompts by combining the user's question
with retrieved context chunks. The resulting prompt is designed to keep
the LLM focused on the provided knowledge base.
"""

from __future__ import annotations

from typing import List, TypedDict


class ChunkMetadata(TypedDict, total=False):
    """Metadata associated with a retrieved context chunk."""

    source: str
    chunk_id: int


class ContextChunk(TypedDict):
    """Retrieved context chunk used for prompt construction."""

    content: str
    metadata: ChunkMetadata


class PromptBuilder:
    """Build grounded prompts for retrieval-augmented generation.

    Args:
        system_instruction: Instruction prepended to the final prompt.
    """

    def __init__(self, system_instruction: str | None = None) -> None:
        """Initialize the prompt builder.

        Args:
            system_instruction: Custom instruction for the LLM.
        """
        self.system_instruction = (
            system_instruction
            if system_instruction is not None
            else (
                "You are a helpful AI assistant. Answer the user's question "
                "using only the provided context. If the answer cannot be "
                "determined from the context, say that the information is "
                "not available in the knowledge base."
            )
        )

    def build(
        self,
        query: str,
        context_chunks: List[ContextChunk],
    ) -> str:
        """Build a final LLM prompt from a query and retrieved context.

        Args:
            query: User question.
            context_chunks: Retrieved chunk records.

        Returns:
            Final prompt string.

        Raises:
            ValueError: If the query is empty, the context list is empty, or a context
                chunk contains blank content.
        """
        if not query or query.strip() == "":
            raise ValueError("Query cannot be empty.")

        if not context_chunks:
            raise ValueError("Context chunks cannot be empty.")

        formatted_context = self._format_context(context_chunks)

        return (
            f"{self.system_instruction}\n\n"
            f"Context:\n{formatted_context}\n\n"
            f"Question:\n{query}\n\n"
            f"Answer:"
        )

    def _format_context(
        self,
        context_chunks: List[ContextChunk],
    ) -> str:
        """Format retrieved context chunks into a numbered block.

        Args:
            context_chunks: Retrieved chunk records.

        Returns:
            Numbered context block.

        Raises:
            ValueError: If a retrieved chunk contains blank content.
        """
        formatted_blocks = []

        for index, chunk_record in enumerate(context_chunks, start=1):
            content = chunk_record["content"]

            if not content.strip():
                raise ValueError("Context chunk content cannot be blank.")

            metadata = chunk_record["metadata"]
            source = metadata.get("source", "unknown_source")
            chunk_id = metadata.get("chunk_id", index)

            formatted_blocks.append(
                f"[Chunk {index} | source={source} | chunk_id={chunk_id}]\n"
                f"{content}"
            )

        return "\n\n".join(formatted_blocks)