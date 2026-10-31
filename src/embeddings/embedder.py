"""
Embedding stage: turns text into L2-normalised float32 vectors using
sentence-transformers. Normalised vectors let FAISS inner product behave as
cosine similarity. The model is loaded once (lazy) and reused.
"""

from typing import List

import numpy as np

from config.settings import settings
from src.utils.logger import get_logger, log_stage

logger = get_logger("embedder")

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # heavy import: keep lazy

        logger.info(f"Loading embedding model '{settings.embedding_model}'")
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embedding_dim() -> int:
    return int(get_model().get_sentence_embedding_dimension())


@log_stage("embedding")
def embed_texts(texts: List[str], paper_id: str = "", batch_size: int = 32) -> np.ndarray:
    if not texts:
        return np.zeros((0, embedding_dim()), dtype="float32")
    vectors = get_model().encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(vectors, dtype="float32")


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query], paper_id="query")
