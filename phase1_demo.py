"""
Phase 1 demo: run the ingestion + parsing stages end-to-end on a single
PDF and print out what was extracted. This is what you'd screenshot/GIF
for the README before Phase 2 (LLM enrichment) exists.

Usage:
    python phase1_demo.py path/to/paper.pdf
"""

import sys
import json
import uuid

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.ocr import run_ocr_on_document
from src.parsing.layout_detector import detect_layout
from src.parsing.section_detector import detect_sections
from src.parsing.table_extractor import extract_tables
from src.parsing.figure_extractor import extract_figures
from src.parsing.reference_extractor import extract_references
from src.utils.schema import PaperSchema
from src.utils.logger import get_logger

logger = get_logger("phase1_demo")


def run_phase1(pdf_path: str) -> PaperSchema:
    paper_id = f"paper_{uuid.uuid4().hex[:8]}"
    logger.info(f"Processing '{pdf_path}' as paper_id='{paper_id}'")

    # 1. Ingest
    document = load_pdf(pdf_path, paper_id=paper_id)
    if document.needs_ocr:
        document = run_ocr_on_document(document, paper_id=paper_id)

    # 2. Parse
    blocks = detect_layout(pdf_path, paper_id=paper_id)
    sections = detect_sections(blocks, paper_id=paper_id)
    tables = extract_tables(pdf_path, paper_id=paper_id)
    figures = extract_figures(pdf_path, paper_id=paper_id)
    references = extract_references(sections.get("references", ""), paper_id=paper_id)

    # 3. Assemble into the canonical schema
    title_block = next((b for b in blocks if b.block_type == "title"), None)

    paper = PaperSchema(
        paper_id=paper_id,
        title=title_block.text if title_block else "",
        abstract=sections.get("abstract", ""),
        sections={k: v for k, v in sections.items() if k not in ("preamble",)},
        tables=[
            {"table_id": t.table_id, "page_number": t.page_number, "csv_path": t.csv_path}
            for t in tables
        ],
        figures=[
            {"figure_id": f.figure_id, "page_number": f.page_number, "image_path": f.image_path}
            for f in figures
        ],
        references=[{"raw_text": r.raw_text, "year": r.year} for r in references],
        source_pdf_path=pdf_path,
        raw_text=document.full_text,
    )

    return paper


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python phase1_demo.py <path_to_pdf>")
        sys.exit(1)

    result = run_phase1(sys.argv[1])
    print("\n--- PHASE 1 OUTPUT (partial schema, LLM fields filled in Phase 2) ---")
    print(result.to_json())
