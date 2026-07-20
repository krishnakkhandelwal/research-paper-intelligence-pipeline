"""
Section detection: takes the classified blocks from layout_detector and
groups body text under the nearest preceding heading.

We do NOT match headings against a fixed whitelist of section names.
Papers use wildly different section titles ("Model Architecture", "Why
Self-Attention", "Related Work", "Approach", etc.) — a whitelist misses
most of them and silently merges unrelated sections together. Instead,
any block layout_detector tagged as a heading becomes a new section
boundary, using its own (cleaned) text as the key.

PDFs frequently store a numbered heading like "3 Model Architecture" as
two separate lines/spans: "3" and "Model Architecture". So adjacent
heading blocks are merged into one logical heading before being used as
a section key.
"""

import re
from typing import List, Dict

from src.parsing.layout_detector import TextBlock
from src.utils.logger import get_logger, log_stage

logger = get_logger("section_detector")

# Strips leading numbering like "1", "3.2", "3.2.1." from a heading
LEADING_NUMBER_PATTERN = re.compile(r"^\s*\d+(\.\d+)*\.?\s*")


def _clean_heading(text: str) -> str:
    cleaned = LEADING_NUMBER_PATTERN.sub("", text).strip()
    return cleaned if cleaned else text.strip()


@log_stage("section_detection")
def detect_sections(blocks: List[TextBlock], paper_id: str = "") -> Dict[str, str]:
    """
    Walks blocks in order. Consecutive 'heading' blocks are merged (to
    join a lone section number with its title), then used as the key for
    all body text that follows until the next heading.
    Anything before the first heading goes into 'preamble'.
    """
    sections: Dict[str, str] = {"preamble": ""}
    current_section = "preamble"
    heading_buffer: List[str] = []

    def flush_heading_buffer():
        nonlocal current_section
        if not heading_buffer:
            return
        combined = " ".join(heading_buffer).strip()
        name = _clean_heading(combined).lower()
        if name:
            current_section = name
            sections.setdefault(current_section, "")
            logger.info(f"Found section heading: '{current_section}'")
        heading_buffer.clear()

    for block in blocks:
        if block.block_type == "heading":
            heading_buffer.append(block.text.strip())
            continue

        # Non-heading block: finalize any pending heading first
        flush_heading_buffer()

        if block.block_type == "title":
            continue  # title text isn't part of any section body

        sections[current_section] = sections.get(current_section, "") + block.text + "\n"

    flush_heading_buffer()  # handle a document that ends on a heading

    return {name: text.strip() for name, text in sections.items()}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.parsing.section_detector <path_to_pdf>")
        sys.exit(1)

    from src.parsing.layout_detector import detect_layout

    blocks = detect_layout(sys.argv[1])
    sections = detect_sections(blocks)
    for name, text in sections.items():
        print(f"\n=== {name.upper()} ===\n{text[:300]}")