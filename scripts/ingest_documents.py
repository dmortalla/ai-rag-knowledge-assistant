"""Script for loading and chunking raw knowledge base documents."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.loaders import load_all_text_documents
from app.processing.chunking import chunk_text


def main() -> None:
    """Load raw documents, chunk them, and print a basic ingestion summary."""
    documents = load_all_text_documents()

    all_chunks: list[str] = []
    for document in documents:
        all_chunks.extend(chunk_text(document))

    print("Ingestion complete.")
    print(f"Documents loaded: {len(documents)}")
    print(f"Total chunks created: {len(all_chunks)}")

    if all_chunks:
        print("\nFirst chunk preview:")
        print(all_chunks[0][:300])


if __name__ == "__main__":
    main()
