"""
Section detection: takes the classified blocks from layout_detector and
groups body text under the nearest preceding heading that matches a
known scientific-paper section name (Introduction, Related Work,
Methodology, Experiments, Results, Conclusion, References, etc.)
"""

import re
from typing import List, Dict

from src.parsing.layout_detector import TextBlock
from src.utils.logger import get_logger, log_stage

logger = get_logger("section_detector")

KNOWN_SECTIONS = [
    "abstract",
    "introduction",
    "related work",
    "background",
    "methodology",
    "method",
    "approach",
    "experiments",
    "experimental setup",
    "results",
    "discussion",
    "conclusion",
    "future work",
    "references",
    "acknowledgements",
]

SECTION_HEADING_PATTERN = re.compile(
    r"^\s*(\d+\.?\s*)?(" + "|".join(KNOWN_SECTIONS) + r")\s*$",
    re.IGNORECASE,
)


@log_stage("section_detection")
def detect_sections(blocks: List[TextBlock], paper_id: str = "") -> Dict[str, str]:
    """
    Walks blocks in order. Whenever a heading/title block's text matches
    a known section name, subsequent body text is accumulated under that
    section key until the next matching heading appears.
    Anything before the first recognized heading goes into 'preamble'.
    """
    sections: Dict[str, str] = {"preamble": ""}
    current_section = "preamble"

    for block in blocks:
        candidate = block.text.strip()
        match = SECTION_HEADING_PATTERN.match(candidate)

        if block.block_type in ("heading", "title") and match:
            current_section = match.group(2).lower()
            sections.setdefault(current_section, "")
            logger.info(f"Found section heading: '{current_section}'")
        else:
            sections[current_section] = sections.get(current_section, "") + block.text + " "

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
