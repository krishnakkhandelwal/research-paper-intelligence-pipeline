"""
Figure extraction: pulls every embedded raster image out of the PDF
using PyMuPDF and saves it as figure1.png, figure2.png, ... Images
below a minimum pixel size are skipped since those are almost always
logos/icons rather than real figures.
"""

import os
import fitz
from dataclasses import dataclass
from typing import List

from src.utils.logger import get_logger, log_stage

logger = get_logger("figure_extractor")

MIN_WIDTH_PX = 100
MIN_HEIGHT_PX = 100


@dataclass
class ExtractedFigure:
    figure_id: str
    page_number: int
    image_path: str


@log_stage("figure_extraction")
def extract_figures(
    pdf_path: str,
    paper_id: str,
    output_dir: str = "data/figures",
) -> List[ExtractedFigure]:
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    extracted: List[ExtractedFigure] = []
    figure_count = 0

    for page_num, page in enumerate(doc):
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            width = base_image.get("width", 0)
            height = base_image.get("height", 0)

            if width < MIN_WIDTH_PX or height < MIN_HEIGHT_PX:
                continue

            figure_count += 1
            figure_id = f"figure{figure_count}"
            ext = base_image.get("ext", "png")
            image_path = os.path.join(output_dir, f"{paper_id}_{figure_id}.{ext}")

            with open(image_path, "wb") as f:
                f.write(base_image["image"])

            extracted.append(
                ExtractedFigure(
                    figure_id=figure_id,
                    page_number=page_num + 1,
                    image_path=image_path,
                )
            )
            logger.info(f"Saved {figure_id} (page {page_num + 1}) -> {image_path}")

    doc.close()
    return extracted


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.parsing.figure_extractor <path_to_pdf>")
        sys.exit(1)

    result = extract_figures(sys.argv[1], paper_id="test_paper")
    print(f"Extracted {len(result)} figures")
