"""
Layout detection: walks the PDF's raw text spans (via PyMuPDF's dict
output) and classifies each one as title / heading / body / caption
using font size and weight as a lightweight stand-in for a full layout
model like LayoutParser. This keeps Phase 1 dependency-light; swapping
in LayoutParser later is a drop-in replacement for this module.
"""

import fitz
from dataclasses import dataclass
from typing import List

from src.utils.logger import get_logger, log_stage

logger = get_logger("layout_detector")

HEADING_FONT_SIZE_THRESHOLD = 12.5
TITLE_FONT_SIZE_THRESHOLD = 16.0


@dataclass
class TextBlock:
    page_number: int
    text: str
    font_size: float
    is_bold: bool
    block_type: str  # 'title' | 'heading' | 'body' | 'caption'


@log_stage("layout_detection")
def detect_layout(pdf_path: str, paper_id: str = "") -> List[TextBlock]:
    """
    Returns a flat list of classified text blocks in reading order.
    Heuristics:
      - Large font on page 1                -> title
      - Bold + font size above threshold    -> heading
      - Starts with 'figure'/'table'        -> caption
      - Everything else                     -> body
    """
    doc = fitz.open(pdf_path)
    blocks: List[TextBlock] = []

    for page_num, page in enumerate(doc):
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue

                    font_size = span.get("size", 0)
                    is_bold = "Bold" in span.get("font", "")

                    if page_num == 0 and font_size >= TITLE_FONT_SIZE_THRESHOLD:
                        block_type = "title"
                    elif font_size >= HEADING_FONT_SIZE_THRESHOLD and is_bold:
                        block_type = "heading"
                    elif text.lower().startswith(("figure", "table")):
                        block_type = "caption"
                    else:
                        block_type = "body"

                    blocks.append(
                        TextBlock(
                            page_number=page_num + 1,
                            text=text,
                            font_size=font_size,
                            is_bold=is_bold,
                            block_type=block_type,
                        )
                    )

    doc.close()
    logger.info(f"Detected {len(blocks)} text blocks")
    return blocks


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.parsing.layout_detector <path_to_pdf>")
        sys.exit(1)

    blocks = detect_layout(sys.argv[1])
    for b in blocks:
        if b.block_type != "body":
            print(f"[{b.block_type.upper()}] p{b.page_number}: {b.text}")
