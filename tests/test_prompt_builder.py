"""Unit tests for the prompt builder module.

These tests validate grounded prompt construction and input validation
for the RAG generation layer.
"""

from __future__ import annotations

import pytest

from app.generation.prompt_builder import PromptBuilder


def test_build_creates_prompt_with_query_and_context():
    """Test that the built prompt includes the query and structured context."""
    builder = PromptBuilder()

    prompt = builder.build(
        query="What is churn?",
        context_chunks=[
            {
                "content": "Churn refers to customers leaving a service.",
                "metadata": {
                    "source": "customer_retention.txt",
                    "chunk_id": 1,
                },
            },
            {
                "content": "It is often measured monthly.",
                "metadata": {
                    "source": "customer_retention.txt",
                    "chunk_id": 2,
                },
            },
        ],
    )

    assert "What is churn?" in prompt
    assert "Churn refers to customers leaving a service." in prompt
    assert "It is often measured monthly." in prompt
    assert "customer_retention.txt" in prompt
    assert "chunk_id=1" in prompt
    assert "chunk_id=2" in prompt
    assert "Answer:" in prompt


def test_build_raises_for_empty_query():
    """Test that an empty query raises ValueError."""
    builder = PromptBuilder()

    context_chunks = [
        {
            "content": "Some context.",
            "metadata": {
                "source": "test.txt",
                "chunk_id": 1,
            },
        }
    ]

    with pytest.raises(ValueError):
        builder.build(query="", context_chunks=context_chunks)


def test_build_raises_for_empty_context_list():
    """Test that an empty context list raises ValueError."""
    builder = PromptBuilder()

    with pytest.raises(ValueError):
        builder.build(query="Test query", context_chunks=[])


def test_build_raises_for_blank_context_chunk():
    """Test that blank structured context content raises ValueError."""
    builder = PromptBuilder()

    context_chunks = [
        {
            "content": "Valid chunk",
            "metadata": {
                "source": "test.txt",
                "chunk_id": 1,
            },
        },
        {
            "content": "   ",
            "metadata": {
                "source": "test.txt",
                "chunk_id": 2,
            },
        },
    ]

    with pytest.raises(ValueError):
        builder.build(
            query="Test query",
            context_chunks=context_chunks,
        )


def test_custom_system_instruction_is_used():
    """Test that a custom system instruction appears in the final prompt."""
    builder = PromptBuilder(
        system_instruction="Answer briefly using only the supplied context."
    )

    prompt = builder.build(
        query="What is retention?",
        context_chunks=[
            {
                "content": "Retention measures customers who stay over time.",
                "metadata": {
                    "source": "customer_retention.txt",
                    "chunk_id": 1,
                },
            }
        ],
    )

    assert "Answer briefly using only the supplied context." in prompt
