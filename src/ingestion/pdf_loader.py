"""
Ingestion stage: opens a PDF, pulls native text per page via PyMuPDF,
and detects pages that look scanned (little/no extractable text) so
those can be rendered to images and handed to the OCR stage.
"""

import os
import fitz  # PyMuPDF
from dataclasses import dataclass, field
from typing import List

from src.utils.logger import get_logger, log_stage

logger = get_logger("pdf_loader")

MIN_CHARS_PER_PAGE_FOR_NATIVE_TEXT = 20


@dataclass
class PageContent:
    page_number: int
    text: str
    has_extractable_text: bool
    image_path: str = ""


@dataclass
class LoadedDocument:
    paper_id: str
    pdf_path: str
    pages: List[PageContent] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n".join(p.text for p in self.pages)

    @property
    def needs_ocr(self) -> bool:
        return any(not p.has_extractable_text for p in self.pages)


@log_stage("ingestion")
def load_pdf(
    pdf_path: str,
    paper_id: str,
    image_dir: str = "data/processed/page_images",
) -> LoadedDocument:
    """
    Opens the PDF and extracts text page by page. Any page with fewer than
    MIN_CHARS_PER_PAGE_FOR_NATIVE_TEXT characters of native text is assumed
    to be a scanned image and gets rendered to a PNG at 300 DPI for the OCR
    stage to pick up.
    """
    os.makedirs(image_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    pages: List[PageContent] = []

    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        has_text = len(text) >= MIN_CHARS_PER_PAGE_FOR_NATIVE_TEXT
        image_path = ""

        if not has_text:
            pix = page.get_pixmap(dpi=300)
            image_path = os.path.join(image_dir, f"{paper_id}_page_{i + 1}.png")
            pix.save(image_path)
            logger.info(f"Page {i + 1} looks scanned -> rendered to {image_path}")

        pages.append(
            PageContent(
                page_number=i + 1,
                text=text,
                has_extractable_text=has_text,
                image_path=image_path,
            )
        )

    doc.close()
    return LoadedDocument(paper_id=paper_id, pdf_path=pdf_path, pages=pages)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.ingestion.pdf_loader <path_to_pdf>")
        sys.exit(1)

    result = load_pdf(sys.argv[1], paper_id="test_paper")
    print(f"Loaded {len(result.pages)} pages. Needs OCR: {result.needs_ocr}")
    print(result.full_text[:500])
