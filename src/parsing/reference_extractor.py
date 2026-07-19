"""
Reference extraction: splits the raw 'references' section text (produced
by section_detector) into individual bibliography entries and pulls out
a best-effort publication year from each. Author/title/venue splitting
is intentionally left light-weight here since free-text regex parsing
of citation styles is unreliable — an LLM pass in Phase 2 enrichment
can clean these up further if needed.
"""

import re
from typing import List

from src.utils.schema import Reference
from src.utils.logger import get_logger, log_stage

logger = get_logger("reference_extractor")

REFERENCE_ENTRY_PREFIX_PATTERN = re.compile(r"^\[?\d+\]?\.?\s+")
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


@log_stage("reference_extraction")
def extract_references(references_text: str, paper_id: str = "") -> List[Reference]:
    if not references_text.strip():
        return []

    # Split on entry markers like "[1] " or "1. " at the start of a line
    raw_entries = re.split(r"\n(?=\[?\d+\]?\.?\s)", references_text.strip())
    refs: List[Reference] = []

    for entry in raw_entries:
        entry = entry.strip()
        if not entry:
            continue

        cleaned = REFERENCE_ENTRY_PREFIX_PATTERN.sub("", entry)
        year_match = YEAR_PATTERN.search(cleaned)
        year = year_match.group(0) if year_match else None

        refs.append(Reference(raw_text=cleaned, year=year))

    logger.info(f"Parsed {len(refs)} reference entries")
    return refs


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.parsing.reference_extractor <path_to_pdf>")
        sys.exit(1)

    from src.parsing.layout_detector import detect_layout
    from src.parsing.section_detector import detect_sections

    blocks = detect_layout(sys.argv[1])
    sections = detect_sections(blocks)
    refs = extract_references(sections.get("references", ""))
    for r in refs[:5]:
        print(r)
