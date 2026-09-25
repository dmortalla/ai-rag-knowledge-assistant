"""Document loading utilities for the RAG pipeline.

This module loads plain-text knowledge base files from the raw data
directory so they can be chunked, embedded, and indexed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from app.core.paths import RAW_DATA_DIR


def load_text_file(file_path: Path) -> str:
    """Load a single UTF-8 text file.

    Args:
        file_path: Path to the text file.

    Returns:
        File contents as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    text = file_path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"File is empty: {file_path}")

    return text


def load_all_text_documents(raw_data_dir: Path = RAW_DATA_DIR) -> List[Dict[str, str]]:
    """Load all plain-text documents from the raw data directory.

    Args:
        raw_data_dir: Directory containing raw text files.

    Returns:
        Loaded documents with text and source filename.

    Raises:
        FileNotFoundError: If the raw data directory does not exist.
        ValueError: If no .txt files are found.
    """
    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_data_dir}")

    text_files = sorted(raw_data_dir.glob("*.txt"))

    if not text_files:
        raise ValueError(f"No .txt files found in: {raw_data_dir}")

    documents = [
        {
            "source": file_path.name,
            "text": load_text_file(file_path),
        }
        for file_path in text_files
    ]

    return documents
