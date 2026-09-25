"""
Unit tests for the answer generator module.

These tests validate that prompt construction and LLM generation work
together correctly.
"""

from __future__ import annotations

from app.generation.answer_generator import AnswerGenerator
from app.generation.llm_client import LLMClient
from app.generation.prompt_builder import PromptBuilder


class DummyResponse:
    """
    Dummy response object with a content attribute.
    """

    def __init__(self, content: str) -> None:
        """
        Initialize the dummy response.

        Args:
        content:
            Response content text.
        """
        self.content = content


class DummyLLM:
    """
    Dummy LLM backend for deterministic tests.
    """

    def invoke(self, prompt: str) -> DummyResponse:
        """
        Return a deterministic answer containing part of the prompt.

        Args:
        prompt:
            Input prompt.
        Returns:
            Mock response object.
        """
        return DummyResponse(
            content=f"Answer based on prompt: {prompt[:30]}"
        )


def test_generate_answer_returns_text():
    """
    Test that the answer generator returns final answer text.

    """
    prompt_builder = PromptBuilder()
    llm_client = LLMClient(llm_backend=DummyLLM())
    generator = AnswerGenerator(
        prompt_builder=prompt_builder,
        llm_client=llm_client,
    )

    result = generator.generate_answer(
        query="What is churn?",
        context_chunks=[
            {
                "content": "Churn is when customers leave a service.",
                "metadata": {
                    "source": "customer_retention.txt",
                    "chunk_id": 1,
                },
            }
        ],
    )

    assert isinstance(result, str)
    assert "Answer based on prompt:" in result
    assert "You are a helpful AI assistant" in result
