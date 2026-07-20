"""
Layout detection: walks the PDF's text lines (via PyMuPDF's dict output)
and classifies each one as title / heading / body / caption.

IMPORTANT: this does NOT use fixed font-size thresholds. Different
papers/templates use wildly different body/heading font sizes (e.g. one
PDF's headings might be 12.5pt bold, another's might be 11.96pt medium-
weight with no "Bold" in the font name at all). Instead we compute the
document's own dominant body-text size first, then classify anything
meaningfully larger than that baseline as a heading. This generalizes
across templates without retuning constants per paper.
"""

import fitz
from collections import Counter
from dataclasses import dataclass
from typing import List

from src.utils.logger import get_logger, log_stage

logger = get_logger("layout_detector")

# A line needs to be at least this many times larger than the document's
# dominant body-text size to count as a heading. Loose enough to catch
# "medium" weight headings that are only slightly larger than body text.
HEADING_SIZE_RATIO = 1.15

# A line this many times larger than body text (on page 1) is treated as
# the paper title rather than just a heading.
TITLE_SIZE_RATIO = 1.6

# Real section headings are short ("Introduction", "Model Architecture").
# Long wrapped paragraphs (e.g. a copyright/footer notice) can coincidentally
# share the heading font size, so cap heading length to filter those out.
MAX_HEADING_CHARS = 60


@dataclass
class TextBlock:
    page_number: int
    text: str
    font_size: float
    is_bold: bool
    block_type: str  # 'title' | 'heading' | 'body' | 'caption'


def _get_body_font_size(doc: "fitz.Document") -> float:
    """
    Finds the most common font size across the whole document, weighted
    by character count. This is our proxy for 'body text size' — the
    baseline everything else gets compared against.
    """
    size_counts: Counter = Counter()
    for page in doc:
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if text.strip():
                        size_counts[round(span.get("size", 0), 1)] += len(text)

    if not size_counts:
        return 10.0  # sane fallback
    return size_counts.most_common(1)[0][0]


@log_stage("layout_detection")
def detect_layout(pdf_path: str, paper_id: str = "") -> List[TextBlock]:
    """
    Returns a flat list of classified text blocks in reading order.
    Blocks are built per PDF *line* (all spans on that line joined into
    one string) rather than per span, so a numbered heading like
    "1" + "Introduction" (which PDFs often store as two separate spans)
    becomes a single "1 Introduction" block instead of being split apart.
    """
    doc = fitz.open(pdf_path)
    body_size = _get_body_font_size(doc)
    logger.info(f"Detected body text baseline size: {body_size}pt")

    heading_min_size = body_size * HEADING_SIZE_RATIO
    title_min_size = body_size * TITLE_SIZE_RATIO

    blocks: List[TextBlock] = []

    for page_num, page in enumerate(doc):
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            for line in block.get("lines", []):
                spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                if not spans:
                    continue

                # Join all spans on this line into one logical block of text
                text = "".join(s["text"] for s in spans).strip()
                # Use the max font size on the line (numbers/labels are
                # sometimes a hair smaller than the accompanying text)
                font_size = max(s.get("size", 0) for s in spans)
                is_bold = any("Bold" in s.get("font", "") for s in spans)

                if page_num == 0 and font_size >= title_min_size:
                    block_type = "title"
                elif font_size >= heading_min_size and len(text) <= MAX_HEADING_CHARS:
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
            print(f"[{b.block_type.upper()}] p{b.page_number} ({b.font_size:.1f}pt): {b.text}")