"""
Chunking stage: splits each section of a PaperSchema into overlapping,
sentence-aware chunks sized in words.

Why section-aware: a chunk never crosses a section boundary, so every chunk
carries a clean 'section' label (useful for filtering and citations).
Why sentence-aware: chunks end on sentence boundaries instead of mid-thought.
Why overlap: the last sentences of a chunk are repeated at the start of the
next one, so context at a boundary is not lost to retrieval.
"""

import re
from dataclasses import dataclass
from typing import List

from config.settings import settings
from src.utils.logger import get_logger, log_stage
from src.utils.schema import PaperSchema

logger = get_logger("chunker")

# Sections that add noise to semantic search.
SKIP_SECTIONS = {"references", "bibliography", "acknowledgments", "acknowledgements", "preamble"}
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


@dataclass
class Chunk:
    paper_id: str
    section: str
    chunk_index: int
    text: str

    @property
    def n_words(self) -> int:
        return len(self.text.split())


def clean_text(text: str) -> str:
    """Repairs PDF line-wrapping: joins hyphenated words, collapses whitespace."""
    text = re.sub(r"-\s*\n\s*(?=[a-z])", "", text)  # "exam-\nple" -> "example"
    return re.sub(r"\s+", " ", text).strip()


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]


def _hard_split(sentence: str, max_words: int) -> List[str]:
    """A single very long 'sentence' (e.g. a table dump) is cut by words."""
    words = sentence.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Greedy sentence packing up to chunk_size words, with sentence-level overlap."""
    if overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    sentences: List[str] = []
    for s in _split_sentences(clean_text(text)):
        sentences.extend(_hard_split(s, chunk_size) if len(s.split()) > chunk_size else [s])

    chunks: List[str] = []
    current: List[str] = []
    current_words = 0

    for sentence in sentences:
        n = len(sentence.split())
        if current and current_words + n > chunk_size:
            chunks.append(" ".join(current))
            # Carry back trailing sentences (up to `overlap` words) into the next chunk.
            carry: List[str] = []
            carried = 0
            for prev in reversed(current):
                w = len(prev.split())
                if carried + w > overlap:
                    break
                carry.insert(0, prev)
                carried += w
            current, current_words = carry, carried
        current.append(sentence)
        current_words += n

    if current and (not chunks or " ".join(current) != chunks[-1]):
        chunks.append(" ".join(current))
    return chunks


@log_stage("chunking")
def chunk_paper(paper: PaperSchema, paper_id: str = "") -> List[Chunk]:
    """Chunks the abstract and every section; falls back to raw_text if no sections were found."""
    chunks: List[Chunk] = []
    idx = 0

    def add(section: str, text: str):
        nonlocal idx
        for piece in chunk_text(text, settings.chunk_size, settings.chunk_overlap):
            chunks.append(Chunk(paper.paper_id, section, idx, piece))
            idx += 1

    sections = {k: v for k, v in paper.sections.items() if k.lower() not in SKIP_SECTIONS and v.strip()}
    if paper.abstract.strip() and "abstract" not in sections:
        add("abstract", paper.abstract)
    for name, text in sections.items():
        add(name, text)

    # Scanned papers have OCR text in raw_text but no detected sections (Phase 1 limitation).
    if not chunks and paper.raw_text.strip():
        logger.warning("No sections found; chunking raw_text as 'full_text'")
        add("full_text", paper.raw_text)

    logger.info(f"Created {len(chunks)} chunks")
    return chunks
