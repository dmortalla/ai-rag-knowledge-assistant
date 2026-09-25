"""
Centralized project path utilities.

This module prevents hardcoded file paths from being scattered across the
project. That makes ingestion, vector storage, and future deployment
cleaner and easier to maintain.
"""

from __future__ import annotations

from pathlib import Path


# Resolve the project root dynamically based on this file's location.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Standard top-level directories.
APP_DIR = PROJECT_ROOT / "app"
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"

# Data subdirectories used by the RAG pipeline.
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"


def ensure_project_directories() -> None:
    """Create required project directories if they do not already exist.

    This is helpful for fresh clones or first-time runs.

    Raises:
        OSError: If directory creation fails.
    """
    required_directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        VECTOR_STORE_DIR,
        ASSETS_DIR,
        SCRIPTS_DIR,
        TESTS_DIR,
    ]

    for directory in required_directories:
        directory.mkdir(parents=True, exist_ok=True)