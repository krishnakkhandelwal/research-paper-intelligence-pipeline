"""
Pipeline orchestrator (Phase 1 + Phase 2).

    python -m src.pipeline ingest data/raw/paper.pdf [--force]
    python -m src.pipeline search "how does self-attention work" [--top-k 5] [--paper-id ID]
    python -m src.pipeline list

ingest: hash PDF -> skip if already processed -> Phase 1 parse -> LLM enrichment
        -> chunk -> embed -> save to SQLite + FAISS.
"""

import argparse
import hashlib
import json
import os
import sys
from typing import Dict

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger("pipeline")


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ingest(pdf_path: str, force: bool = False) -> Dict:
    # Heavy imports live here so `list` / `--help` stay fast.
    from phase1_demo import run_phase1
    from src.embeddings.embedder import embed_texts, embedding_dim
    from src.enrichment.chunker import chunk_paper
    from src.enrichment.llm_extractor import enrich_paper
    from src.storage import database as db
    from src.storage.vector_store import VectorStore

    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(pdf_path)

    db.init_db()
    content_hash = file_sha256(pdf_path)
    paper_id = f"paper_{content_hash[:12]}"

    existing = db.find_paper_by_hash(content_hash)
    if existing and not force:
        logger.info(f"'{pdf_path}' already processed as {existing.paper_id}; skipping (use --force)")
        return {"paper_id": existing.paper_id, "status": "skipped"}
    store = VectorStore(embedding_dim(), settings.faiss_index_path)
    if existing:
        store.remove(db.delete_paper(existing.paper_id))

    paper = run_phase1(pdf_path, paper_id=paper_id)
    paper = enrich_paper(paper, paper_id=paper_id)
    chunks = chunk_paper(paper, paper_id=paper_id)
    vectors = embed_texts([c.text for c in chunks], paper_id=paper_id)

    # SQLite and FAISS are separate stores with no shared transaction: write SQLite
    # first, and undo it if the FAISS write fails so they never disagree.
    db.save_paper(paper, content_hash)
    chunk_ids = []
    try:
        chunk_ids = db.add_chunks(chunks)
        store.add(chunk_ids, vectors)
        store.save()
    except Exception:
        logger.error("Vector write failed; rolling back SQLite rows")
        db.delete_paper(paper_id)
        raise

    summary = {"paper_id": paper_id, "status": "ingested", "title": paper.title,
               "sections": len(paper.sections), "tables": len(paper.tables),
               "figures": len(paper.figures), "references": len(paper.references),
               "chunks": len(chunks), "vectors_in_index": store.count}
    logger.info(f"Done: {json.dumps(summary)}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Research Intelligence Pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ing = sub.add_parser("ingest", help="Process a PDF end to end")
    p_ing.add_argument("pdf")
    p_ing.add_argument("--force", action="store_true", help="Reprocess even if already ingested")

    p_s = sub.add_parser("search", help="Semantic search across all ingested papers")
    p_s.add_argument("query")
    p_s.add_argument("--top-k", type=int, default=None)
    p_s.add_argument("--paper-id", default=None)

    sub.add_parser("list", help="List ingested papers")
    args = parser.parse_args()

    if args.command == "ingest":
        print(json.dumps(ingest(args.pdf, force=args.force), indent=2))
    elif args.command == "search":
        from src.retrieval.search import semantic_search
        from src.storage import database as db
        db.init_db()
        for r in semantic_search(args.query, args.top_k, args.paper_id):
            print(f"\n[{r['score']:.3f}] {r['title'] or r['paper_id']}  |  section: {r['section']}")
            print(f"  {r['text'][:300]}...")
    elif args.command == "list":
        from src.storage import database as db
        db.init_db()
        rows = db.list_papers()
        for p in rows:
            e = json.loads(p.enrichment_json)
            print(f"{p.paper_id}  {p.title[:70]!r}\n    datasets={e['datasets']} models={e['models']} metrics={e['metrics']}")
        if not rows:
            print("No papers ingested yet.")


if __name__ == "__main__":
    sys.exit(main())
