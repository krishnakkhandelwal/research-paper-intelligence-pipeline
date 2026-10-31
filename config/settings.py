"""
Central configuration for the whole pipeline.
Every module reads paths/params from here instead of hardcoding values,
so changing a model name or a directory only happens in one place.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # LLM (Phase 2 enrichment)
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_max_context_chars: int = int(os.getenv("LLM_MAX_CONTEXT_CHARS", 12000))
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

    # Storage (Phase 2)
    db_path: str = os.getenv("DB_PATH", "./data/papers.db")
    faiss_index_path: str = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index.bin")

    # Embeddings + chunking (Phase 2).
    # all-MiniLM-L6-v2 truncates input at 256 word-pieces, so chunks are sized
    # in WORDS (~1.3 word-pieces per word) to stay safely under that limit.
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 150))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 30))
    top_k: int = int(os.getenv("TOP_K", 5))

    # Data directories (Phase 1)
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    page_images_dir: str = "data/processed/page_images"
    tables_dir: str = "data/tables"
    figures_dir: str = "data/figures"


settings = Settings()
