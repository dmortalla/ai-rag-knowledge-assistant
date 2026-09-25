"""LLM client utilities for the RAG pipeline.

This module wraps the chat model used for final answer generation.
It supports dependency injection so tests can run without making
real API calls.
"""

from __future__ import annotations

from typing import Optional

from openai import OpenAI

from app.core.config import get_settings


class LLMClient:
    """Wrapper around an LLM backend.

    Args:
        llm_backend: Prebuilt LLM backend used primarily for testing. If not provided,
            the client will initialize an OpenAI SDK client lazily.
    """

    def __init__(self, llm_backend: Optional[object] = None) -> None:
        """Initialize the LLM client.

        Args:
            llm_backend: Optional injected backend for testing.
        """
        settings = get_settings()

        self.api_key = settings.openai_api_key
        self.model = settings.chat_model
        self._llm = llm_backend
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        """Return the OpenAI client, creating it if necessary.

        Returns:
            OpenAI client instance.
        """
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)

        return self._client

    def generate(self, prompt: str) -> str:
        """Generate a response from a prompt.

        Args:
            prompt: Final prompt string.

        Returns:
            Model-generated response text.

        Raises:
            ValueError: If the prompt is empty.
        """
        if not prompt or prompt.strip() == "":
            raise ValueError("Prompt cannot be empty.")

        if self._llm is not None:
            response = self._llm.invoke(prompt)
            if hasattr(response, "content"):
                return response.content
            return str(response)

        client = self._get_client()
        response = client.responses.create(
            model=self.model,
            input=prompt,
        )

        return response.output_text
