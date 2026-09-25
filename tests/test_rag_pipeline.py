"""
Unit tests for the end-to-end RAG pipeline.

These tests validate that retrieval and answer generation are connected
correctly and that the pipeline returns the expected response structure.
"""

from __future__ import annotations

import pytest

from app.generation.answer_generator import AnswerGenerator
from app.generation.llm_client import LLMClient
from app.generation.prompt_builder import PromptBuilder
from app.orchestration.rag_pipeline import RAGPipeline
from app.retrieval.retriever import Retriever


class DummyVectorStore:
    """
    Dummy vector store used for deterministic retrieval tests.
    """

    def similarity_search(
        self,
        query: str,
        k: int = 4,
    ) -> list[dict[str, object]]:
        """
        Return deterministic structured retrieval results.

        Parameters
        ----------
        query : str
            Input query.
        k : int, default=4
            Number of results.

        Returns
        -------
        list of dict
            Mock retrieved context records.
        """
        return [
            {
                "content": f"{query}_chunk_{index}",
                "metadata": {
                    "source": "dummy_source.txt",
                    "chunk_id": index,
                },
            }
            for index in range(k)
        ]


class DummyResponse:
    """
    Dummy response object with a content attribute.
    """

    def __init__(self, content: str) -> None:
        """
        Initialize the dummy response.

        Parameters
        ----------
        content : str
            Response text.
        """
        self.content = content


class DummyLLM:
    """
    Dummy LLM backend for deterministic answer generation.
    """

    def invoke(self, prompt: str) -> DummyResponse:
        """
        Return a deterministic response.

        Parameters
        ----------
        prompt : str
            Input prompt.

        Returns
        -------
        DummyResponse
            Mock response object.
        """
        return DummyResponse(
            content=f"Pipeline answer from prompt: {prompt[:40]}"
        )


def test_run_returns_expected_structure():
    """
    Test that the RAG pipeline returns query, context, sources, and answer.

    Returns
    -------
    None
    """
    retriever = Retriever(
        vector_store=DummyVectorStore(),
        default_k=2,
    )
    prompt_builder = PromptBuilder()
    llm_client = LLMClient(llm_backend=DummyLLM())
    answer_generator = AnswerGenerator(
        prompt_builder=prompt_builder,
        llm_client=llm_client,
    )
    pipeline = RAGPipeline(
        retriever=retriever,
        answer_generator=answer_generator,
    )

    result = pipeline.run("What is churn?")

    assert isinstance(result, dict)
    assert result["query"] == "What is churn?"

    assert result["context_chunks"] == [
        "What is churn?_chunk_0",
        "What is churn?_chunk_1",
    ]

    assert len(result["sources"]) == 2
    assert result["sources"][0] == {
        "source": "dummy_source.txt",
        "chunk_id": 0,
        "content": "What is churn?_chunk_0",
    }
    assert result["sources"][1] == {
        "source": "dummy_source.txt",
        "chunk_id": 1,
        "content": "What is churn?_chunk_1",
    }

    assert isinstance(result["answer"], str)
    assert "Pipeline answer from prompt:" in result["answer"]


def test_run_raises_for_empty_query():
    """
    Test that an empty query raises ValueError.

    Returns
    -------
    None
    """
    retriever = Retriever(
        vector_store=DummyVectorStore(),
        default_k=2,
    )
    prompt_builder = PromptBuilder()
    llm_client = LLMClient(llm_backend=DummyLLM())
    answer_generator = AnswerGenerator(
        prompt_builder=prompt_builder,
        llm_client=llm_client,
    )
    pipeline = RAGPipeline(
        retriever=retriever,
        answer_generator=answer_generator,
    )

    with pytest.raises(ValueError):
        pipeline.run("")