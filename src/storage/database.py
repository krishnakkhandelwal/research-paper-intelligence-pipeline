"""
SQLite metadata store (SQLAlchemy 2.0 ORM).
 - papers: one row per processed PDF, keyed by paper_id, with the SHA-256 of
   the file so an unchanged PDF is never reprocessed (cache / idempotency).
 - chunks: one row per text chunk. Its auto-increment id is the vector id in FAISS.
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from config.settings import settings
from src.enrichment.chunker import Chunk
from src.utils.schema import PaperSchema


class Base(DeclarativeBase):
    pass


class PaperRow(Base):
    __tablename__ = "papers"
    paper_id: Mapped[str] = mapped_column(String, primary_key=True)
    content_hash: Mapped[str] = mapped_column(String, unique=True, index=True)
    title: Mapped[str] = mapped_column(Text, default="")
    abstract: Mapped[str] = mapped_column(Text, default="")
    source_pdf_path: Mapped[str] = mapped_column(Text, default="")
    enrichment_json: Mapped[str] = mapped_column(Text, default="{}")
    schema_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class ChunkRow(Base):
    __tablename__ = "chunks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.paper_id"), index=True)
    section: Mapped[str] = mapped_column(String)
    chunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    n_words: Mapped[int] = mapped_column(Integer)


_engine = None
_Session = None


def init_db(db_path: Optional[str] = None):
    """Creates the engine and tables (idempotent). Returns a session factory."""
    global _engine, _Session
    path = db_path or settings.db_path
    if path != ":memory:":
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(_engine)
    _Session = sessionmaker(_engine, expire_on_commit=False)
    return _Session


def _session():
    if _Session is None:
        init_db()
    return _Session()


def find_paper_by_hash(content_hash: str) -> Optional[PaperRow]:
    with _session() as s:
        return s.scalar(select(PaperRow).where(PaperRow.content_hash == content_hash))


def save_paper(paper: PaperSchema, content_hash: str) -> None:
    enrichment = {
        "problem_statement": paper.problem_statement,
        "novel_contribution": paper.novel_contribution,
        "datasets": paper.datasets,
        "models": paper.models,
        "metrics": paper.metrics,
        "limitations": paper.limitations,
        "future_work": paper.future_work,
    }
    with _session() as s:
        s.merge(PaperRow(
            paper_id=paper.paper_id, content_hash=content_hash, title=paper.title,
            abstract=paper.abstract, source_pdf_path=paper.source_pdf_path,
            enrichment_json=json.dumps(enrichment), schema_json=paper.to_json(),
        ))
        s.commit()


def add_chunks(chunks: List[Chunk]) -> List[int]:
    """Inserts chunks and returns their ids (same order) to use as FAISS ids."""
    rows = [ChunkRow(paper_id=c.paper_id, section=c.section, chunk_index=c.chunk_index,
                     text=c.text, n_words=c.n_words) for c in chunks]
    with _session() as s:
        s.add_all(rows)
        s.flush()
        ids = [r.id for r in rows]
        s.commit()
    return ids


def get_chunks_by_ids(ids: List[int]) -> Dict[int, ChunkRow]:
    if not ids:
        return {}
    with _session() as s:
        rows = s.scalars(select(ChunkRow).where(ChunkRow.id.in_(ids))).all()
    return {r.id: r for r in rows}


def get_papers_by_ids(paper_ids: List[str]) -> Dict[str, PaperRow]:
    with _session() as s:
        rows = s.scalars(select(PaperRow).where(PaperRow.paper_id.in_(paper_ids))).all()
    return {r.paper_id: r for r in rows}


def delete_paper(paper_id: str) -> List[int]:
    """Deletes a paper and its chunks. Returns the removed chunk ids (to purge from FAISS)."""
    with _session() as s:
        ids = list(s.scalars(select(ChunkRow.id).where(ChunkRow.paper_id == paper_id)).all())
        s.query(ChunkRow).filter(ChunkRow.paper_id == paper_id).delete()
        s.query(PaperRow).filter(PaperRow.paper_id == paper_id).delete()
        s.commit()
    return ids


def list_papers() -> List[PaperRow]:
    with _session() as s:
        return list(s.scalars(select(PaperRow).order_by(PaperRow.created_at)).all())
