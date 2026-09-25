"""
Answer generation utilities for the RAG pipeline.

This module connects prompt construction to the LLM client and returns
the final generated answer.
"""

from __future__ import annotations

from typing import Dict, List

from app.generation.llm_client import LLMClient
from app.generation.prompt_builder import PromptBuilder


class AnswerGenerator:
    """
    Generate grounded answers from a query and retrieved context.

    Parameters
    ----------
    prompt_builder : PromptBuilder
        Prompt builder instance used to construct the final prompt.
    llm_client : LLMClient
        LLM client instance used to generate the answer.
    """

    def __init__(
        self,
        prompt_builder: PromptBuilder,
        llm_client: LLMClient,
    ) -> None:
        """
        Initialize the answer generator.

        Parameters
        ----------
        prompt_builder : PromptBuilder
            Prompt builder instance.
        llm_client : LLMClient
            LLM client instance.
        """
        self.prompt_builder = prompt_builder
        self.llm_client = llm_client

    def generate_answer(self, query: str, context_chunks: List[Dict[str, object]]) -> str:
        """
        Generate an answer from a query and retrieved context.

        Parameters
        ----------
        query : str
            User question.
        context_chunks : list of dict
            Retrieved chunk records.

        Returns
        -------
        str
            Final generated answer.
        """
        prompt = self.prompt_builder.build(
            query=query,
            context_chunks=context_chunks,
        )
        return self.llm_client.generate(prompt)