"""
FastAPI application entrypoint for the AI RAG Knowledge Assistant.
"""

from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router


app = FastAPI(
    title="AI RAG Knowledge Assistant API",
    version="0.1.0",
    description="FastAPI backend for a production-style RAG application.",
)

app.include_router(router)