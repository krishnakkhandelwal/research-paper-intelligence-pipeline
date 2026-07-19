"""
OCR stage: fills in text for pages that pdf_loader flagged as scanned
(i.e. had no usable native text layer), using Tesseract on the rendered
page image.
"""

import pytesseract
from PIL import Image

from src.utils.logger import get_logger, log_stage
from src.ingestion.pdf_loader import LoadedDocument

logger = get_logger("ocr")


@log_stage("ocr")
def run_ocr_on_document(document: LoadedDocument, paper_id: str = "") -> LoadedDocument:
    """
    Mutates and returns the document: any page with has_extractable_text=False
    gets its .text field filled in via OCR on the page's rendered image.
    """
    for page in document.pages:
        if not page.has_extractable_text and page.image_path:
            try:
                image = Image.open(page.image_path)
                ocr_text = pytesseract.image_to_string(image)
                page.text = ocr_text.strip()
                page.has_extractable_text = len(page.text) > 0
                logger.info(
                    f"OCR extracted {len(page.text)} chars from page {page.page_number}"
                )
            except Exception as e:
                logger.error(f"OCR failed on page {page.page_number}: {e}")

    return document


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.ingestion.ocr <path_to_pdf>")
        sys.exit(1)

    from src.ingestion.pdf_loader import load_pdf

    doc = load_pdf(sys.argv[1], paper_id="test_paper")
    doc = run_ocr_on_document(doc, paper_id="test_paper")
    print(doc.full_text[:1000])
