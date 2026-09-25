"""
Script for building and saving a vector store from raw documents.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.loaders import load_all_text_documents
from app.processing.chunking import chunk_text
from app.vectorstore.store_manager import get_vector_store


def main() -> None:
    """
    Build and save a vector index from raw text documents.

    """
    documents = load_all_text_documents()

    all_chunks: list[str] = []
    all_metadatas: list[dict[str, object]] = []

    for document in documents:
        source_name = document["source"]
        text = document["text"]
        chunks = chunk_text(text)

        for index, chunk in enumerate(chunks, start=1):
            all_chunks.append(chunk)
            all_metadatas.append(
                {
                    "source": source_name,
                    "chunk_id": index,
                }
            )

    if not all_chunks:
        raise ValueError("No chunks were created from the input documents.")

    vector_store = get_vector_store()
    vector_store.add_texts(texts=all_chunks, metadatas=all_metadatas)
    save_path = vector_store.save()

    print("Vector store build complete.")
    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks indexed: {len(all_chunks)}")
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()
