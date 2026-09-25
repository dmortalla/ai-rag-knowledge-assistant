"""Application configuration for the AI RAG Knowledge Assistant.

This module centralizes all runtime settings so the rest of the codebase
does not hardcode model names, chunking values, or backend selections.
That makes the application easier to test, maintain, and extend.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


# Load environment variables from a local .env file if present.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Immutable application settings.

    Attributes:
        openai_api_key: OpenAI API key for embeddings and chat generation.
        vector_db: Vector backend name, such as "faiss" or "pinecone".
        embedding_model: OpenAI embedding model name.
        chat_model: OpenAI chat model name.
        top_k: Number of retrieved chunks returned for each query.
        chunk_size: Maximum size of each text chunk.
        chunk_overlap: Overlap size between adjacent chunks.
    """

    openai_api_key: str
    vector_db: str
    embedding_model: str
    chat_model: str
    top_k: int
    chunk_size: int
    chunk_overlap: int


def _get_env(name: str, default: str | None = None) -> str:
    """Safely retrieve a string environment variable.

    Args:
        name: Environment variable name.
        default: Optional fallback value.

    Returns:
        The environment variable value.

    Raises:
        ValueError: If the variable is missing and no default is provided.
    """
    value = os.getenv(name, default)

    if value is None or value.strip() == "":
        raise ValueError(f"Missing required environment variable: {name}")

    return value


def _get_int_env(name: str, default: int) -> int:
    """Safely retrieve and parse an integer environment variable.

    Args:
        name: Environment variable name.
        default: Fallback integer value.

    Returns:
        Parsed integer value.

    Raises:
        ValueError: If the value cannot be parsed as an integer.
    """
    raw_value = os.getenv(name, str(default))

    try:
        return int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Environment variable {name} must be an integer. Got: {raw_value}"
        ) from exc


def get_settings() -> Settings:
    """Build and validate the application settings.

    Returns:
        A validated Settings instance.

    Raises:
        AssertionError: If a configuration rule is violated.
        ValueError: If a required variable is missing or invalid.
    """
    settings = Settings(
        openai_api_key=_get_env("OPENAI_API_KEY", "your_openai_api_key_here"),
        vector_db=_get_env("VECTOR_DB", "faiss").lower(),
        embedding_model=_get_env("EMBEDDING_MODEL", "text-embedding-3-small"),
        chat_model=_get_env("CHAT_MODEL", "gpt-4o-mini"),
        top_k=_get_int_env("TOP_K", 4),
        chunk_size=_get_int_env("CHUNK_SIZE", 800),
        chunk_overlap=_get_int_env("CHUNK_OVERLAP", 150),
    )

    # Guardrails / validation
    assert settings.vector_db in {"faiss", "pinecone"}, (
        "VECTOR_DB must be either 'faiss' or 'pinecone'."
    )
    assert settings.top_k > 0, "TOP_K must be greater than 0."
    assert settings.chunk_size > 0, "CHUNK_SIZE must be greater than 0."
    assert settings.chunk_overlap >= 0, "CHUNK_OVERLAP cannot be negative."
    assert settings.chunk_overlap < settings.chunk_size, (
        "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
    )

    return settings
