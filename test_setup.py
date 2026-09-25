"""Quick setup validation for config and paths."""

from app.core.config import get_settings
from app.core.paths import (
    PROJECT_ROOT,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    VECTOR_STORE_DIR,
    ensure_project_directories,
)


def main() -> None:
    """Run a basic sanity check for initial project setup."""
    settings = get_settings()
    ensure_project_directories()

    print("Project root:", PROJECT_ROOT)
    print("Raw data dir exists:", RAW_DATA_DIR.exists())
    print("Processed data dir exists:", PROCESSED_DATA_DIR.exists())
    print("Vector store dir exists:", VECTOR_STORE_DIR.exists())
    print("Vector DB:", settings.vector_db)
    print("Embedding model:", settings.embedding_model)
    print("Chat model:", settings.chat_model)
    print("Top K:", settings.top_k)
    print("Chunk size:", settings.chunk_size)
    print("Chunk overlap:", settings.chunk_overlap)


if __name__ == "__main__":
    main()