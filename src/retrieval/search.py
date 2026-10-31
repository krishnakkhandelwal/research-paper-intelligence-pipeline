"""
Semantic search: embeds the query, asks FAISS for nearest chunks, then joins
chunk + paper metadata from SQLite. Phase 3's API will call this directly.
"""

from typing import Dict, List, Optional

from config.settings import settings
from src.embeddings.embedder import embed_query, embedding_dim
from src.storage import database as db
from src.storage.vector_store import VectorStore


def open_store() -> VectorStore:
    return VectorStore(embedding_dim(), settings.faiss_index_path)


def semantic_search(query: str, top_k: Optional[int] = None, paper_id: Optional[str] = None,
                    store: Optional[VectorStore] = None) -> List[Dict]:
    k = top_k or settings.top_k
    store = store or open_store()
    # FAISS cannot filter by paper, so over-fetch and filter afterwards.
    hits = store.search(embed_query(query), k * 10 if paper_id else k)
    chunks = db.get_chunks_by_ids([i for i, _ in hits])
    papers = db.get_papers_by_ids(list({c.paper_id for c in chunks.values()}))

    results: List[Dict] = []
    for chunk_id, score in hits:
        c = chunks.get(chunk_id)
        if c is None or (paper_id and c.paper_id != paper_id):
            continue
        results.append({
            "score": round(score, 4), "paper_id": c.paper_id,
            "title": papers[c.paper_id].title if c.paper_id in papers else "",
            "section": c.section, "chunk_index": c.chunk_index, "text": c.text,
        })
        if len(results) == k:
            break
    return results
