"""
Unit tests for the LLM client module.

These tests validate prompt handling and response extraction without
making real API calls.
"""

from __future__ import annotations

import pytest

from app.generation.llm_client import LLMClient


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
            Response content text.
        """
        self.content = content


class DummyLLM:
    """
    Dummy LLM backend for deterministic tests.
    """

    def invoke(self, prompt: str) -> DummyResponse:
        """
        Return a deterministic dummy response.

        Parameters
        ----------
        prompt : str
            Input prompt.

        Returns
        -------
        DummyResponse
            Mock response object.
        """
        return DummyResponse(content=f"Generated answer for: {prompt[:20]}")


def test_generate_returns_response_content():
    """
    Test that the LLM client returns response content.

    Returns
    -------
    None
    """
    client = LLMClient(llm_backend=DummyLLM())

    result = client.generate("What is churn?")

    assert isinstance(result, str)
    assert "Generated answer for:" in result


def test_generate_raises_for_empty_prompt():
    """
    Test that an empty prompt raises ValueError.

    Returns
    -------
    None
    """
    client = LLMClient(llm_backend=DummyLLM())

    with pytest.raises(ValueError):
        client.generate("")