"""
API routes for the AI RAG Knowledge Assistant.

This module defines FastAPI endpoints and wires them to the RAG pipeline.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import QueryRequest, QueryResponse
from app.generation.answer_generator import AnswerGenerator
from app.generation.llm_client import LLMClient
from app.generation.prompt_builder import PromptBuilder
from app.orchestration.rag_pipeline import RAGPipeline
from app.retrieval.retriever import Retriever
from app.vectorstore.store_manager import get_vector_store


router = APIRouter()


def build_pipeline() -> RAGPipeline:
    """
    Build the current RAG pipeline instance for the API.

    Returns
    -------
    RAGPipeline
        Application pipeline used by the API route.
    """
    vector_store = get_vector_store()
    vector_store.load(folder_name="faiss_index")

    retriever = Retriever(vector_store=vector_store)
    prompt_builder = PromptBuilder()
    llm_client = LLMClient()
    answer_generator = AnswerGenerator(
        prompt_builder=prompt_builder,
        llm_client=llm_client,
    )

    return RAGPipeline(
        retriever=retriever,
        answer_generator=answer_generator,
    )


@router.get("/health")
def health_check() -> dict[str, str]:
    """
    Return a simple API health status.

    Returns
    -------
    dict of str to str
        Health status payload.
    """
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest) -> QueryResponse:
    """
    Run the RAG pipeline for a user query.

    Parameters
    ----------
    request : QueryRequest
        Incoming query request body.

    Returns
    -------
    QueryResponse
        Answer and supporting context returned by the pipeline.
    """
    pipeline = build_pipeline()
    result = pipeline.run(request.query)
    return QueryResponse(**result)