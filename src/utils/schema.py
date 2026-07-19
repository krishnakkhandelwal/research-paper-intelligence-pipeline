"""
Canonical data schema for a processed paper. Every module (parsing,
enrichment, storage, API) reads/writes this shape so the whole pipeline
agrees on one contract, instead of every stage inventing its own dict format.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional


@dataclass
class Reference:
    raw_text: str
    authors: Optional[str] = None
    year: Optional[str] = None
    title: Optional[str] = None
    venue: Optional[str] = None


@dataclass
class TableRecord:
    table_id: str
    page_number: int
    csv_path: str
    caption: Optional[str] = None


@dataclass
class FigureRecord:
    figure_id: str
    page_number: int
    image_path: str
    caption: Optional[str] = None


@dataclass
class PaperSchema:
    paper_id: str
    title: str = ""
    authors: List[str] = field(default_factory=list)
    affiliations: List[str] = field(default_factory=list)
    abstract: str = ""
    keywords: List[str] = field(default_factory=list)
    sections: Dict[str, str] = field(default_factory=dict)
    tables: List[TableRecord] = field(default_factory=list)
    figures: List[FigureRecord] = field(default_factory=list)
    references: List[Reference] = field(default_factory=list)

    # Filled in during Phase 2 (LLM enrichment)
    datasets: List[str] = field(default_factory=list)
    models: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    problem_statement: str = ""
    novel_contribution: str = ""
    limitations: str = ""
    future_work: str = ""

    source_pdf_path: str = ""
    raw_text: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    @staticmethod
    def from_json(json_str: str) -> "PaperSchema":
        data = json.loads(json_str)
        return PaperSchema(**data)
