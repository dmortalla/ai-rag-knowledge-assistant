"""
Unit tests for the FastAPI application layer.

These tests validate the health endpoint and the query endpoint without
making real LLM API calls.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import routes
from app.api.main import app


class DummyPipeline:
    """
    Dummy pipeline used to isolate API tests from real backend logic.
    """

    def run(self, query: str) -> dict:
        """
        Return a deterministic mock RAG response.

        Parameters
        ----------
        query : str
            User query.

        Returns
        -------
        dict
            Mock pipeline result.
        """
        return {
            "query": query,
            "context_chunks": [
                "Mock chunk 1",
                "Mock chunk 2",
            ],
            "sources": [
                {
                    "source": "mock_source.txt",
                    "chunk_id": 1,
                    "content": "Mock chunk 1",
                },
                {
                    "source": "mock_source.txt",
                    "chunk_id": 2,
                    "content": "Mock chunk 2",
                },
            ],
            "answer": f"Mock answer for: {query}",
        }


client = TestClient(app)


def test_health_endpoint():
    """
    Test that the health endpoint returns status ok.

    Returns
    -------
    None
    """
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_endpoint(monkeypatch):
    """
    Test that the query endpoint returns a structured response.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the pipeline builder.

    Returns
    -------
    None
    """
    monkeypatch.setattr(
        routes,
        "build_pipeline",
        lambda: DummyPipeline(),
    )

    response = client.post(
        "/query",
        json={"query": "What is churn?"},
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["query"] == "What is churn?"
    assert payload["context_chunks"] == [
        "Mock chunk 1",
        "Mock chunk 2",
    ]
    assert payload["sources"] == [
        {
            "source": "mock_source.txt",
            "chunk_id": 1,
            "content": "Mock chunk 1",
        },
        {
            "source": "mock_source.txt",
            "chunk_id": 2,
            "content": "Mock chunk 2",
        },
    ]
    assert payload["answer"] == "Mock answer for: What is churn?"