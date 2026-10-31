"""
FAISS vector store. Uses IndexIDMap2(IndexFlatIP) so each vector is stored
under the integer id of its chunk row in SQLite. That id is the only link
between the two stores. Exact (flat) search is ideal up to ~100k chunks;
switch to HNSW/IVF beyond that.
"""

import os
from typing import List, Tuple

import faiss
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("vector_store")


class VectorStore:
    def __init__(self, dim: int, path: str):
        self.dim = dim
        self.path = path
        if os.path.exists(path):
            self.index = faiss.read_index(path)
            if self.index.d != dim:
                raise ValueError(
                    f"Index at {path} has dim {self.index.d}, model produces {dim}. "
                    "Delete the index (and re-ingest with --force) after changing EMBEDDING_MODEL."
                )
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
        else:
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    @property
    def count(self) -> int:
        return int(self.index.ntotal)

    def add(self, ids: List[int], vectors: np.ndarray) -> None:
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must have the same length")
        if len(ids):
            self.index.add_with_ids(np.ascontiguousarray(vectors, dtype="float32"),
                                    np.asarray(ids, dtype="int64"))

    def remove(self, ids: List[int]) -> None:
        if len(ids):
            self.index.remove_ids(np.asarray(ids, dtype="int64"))

    def search(self, query: np.ndarray, k: int) -> List[Tuple[int, float]]:
        if self.count == 0:
            return []
        scores, ids = self.index.search(np.ascontiguousarray(query, dtype="float32"), k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        faiss.write_index(self.index, self.path)
