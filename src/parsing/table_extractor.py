"""
Table extraction: uses Camelot to detect and pull out every table in
the PDF, saving each one as a standalone CSV (table_1.csv, table_2.csv, ...).
Tries 'lattice' mode first (tables with visible grid lines), and falls
back to 'stream' mode for borderless tables if nothing is found.
"""

import os
import camelot
from dataclasses import dataclass
from typing import List

from src.utils.logger import get_logger, log_stage

logger = get_logger("table_extractor")


@dataclass
class ExtractedTable:
    table_id: str
    page_number: int
    csv_path: str


@log_stage("table_extraction")
def extract_tables(
    pdf_path: str,
    paper_id: str,
    output_dir: str = "data/tables",
) -> List[ExtractedTable]:
    os.makedirs(output_dir, exist_ok=True)
    extracted: List[ExtractedTable] = []

    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
        if tables.n == 0:
            tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
    except Exception as e:
        logger.error(f"Camelot failed on {pdf_path}: {e}")
        return extracted

    for i, table in enumerate(tables, start=1):
        table_id = f"table_{i}"
        csv_path = os.path.join(output_dir, f"{paper_id}_{table_id}.csv")
        table.df.to_csv(csv_path, index=False, header=False)

        extracted.append(
            ExtractedTable(
                table_id=table_id,
                page_number=int(table.page),
                csv_path=csv_path,
            )
        )
        logger.info(f"Saved {table_id} (page {table.page}) -> {csv_path}")

    return extracted


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.parsing.table_extractor <path_to_pdf>")
        sys.exit(1)

    result = extract_tables(sys.argv[1], paper_id="test_paper")
    print(f"Extracted {len(result)} tables")
