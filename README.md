# Research Intelligence Pipeline

An end-to-end data/ML engineering pipeline that turns a scientific paper (PDF) into structured, searchable, LLM-enriched data — sections, tables, figures, references, extracted datasets/models/metrics, embeddings, and a queryable API.

## What this project aims to do

Most "AI portfolio projects" are movie recommenders or churn predictors. This one is different: it's a **document-processing data pipeline** that mirrors what a real Data/ML Engineer builds in industry — ingest unstructured input, extract structured signal from it, enrich it with an LLM, index it for retrieval, and serve it through an API.

Concretely, given any research paper PDF, the pipeline:
1. Ingests the PDF and OCRs any scanned pages
2. Detects layout (title/heading/body/caption) and groups text into sections (Introduction, Methodology, Results, etc.)
3. Extracts every table to its own CSV and every figure to its own image
4. Parses the bibliography into individual reference entries
5. Uses an LLM to pull out the problem statement, novel contribution, datasets used, models, metrics, and limitations
6. Chunks and embeds the paper's content, storing vectors in FAISS and metadata in SQLite
7. Exposes all of this through a FastAPI service (upload / semantic search across papers / RAG-based Q&A on a paper) and a Streamlit dashboard

The point isn't summarization — it's **turning a folder of PDFs into a searchable, structured research database**.

## Why this maps to a Data/ML Engineer role

| Skill the JD asks for | Where it shows up here |
|---|---|
| Data pipeline development | The ingestion → parsing → enrichment → embedding → storage chain, orchestrated as one pipeline |
| ML model development | LLM-based information extraction, embedding models for semantic search |
| Data integration (structured/semi/unstructured) | Unstructured text + semi-structured tables + structured JSON, unified into one schema |
| Performance optimization | Chunking strategy, FAISS indexing, caching processed results in SQLite instead of reprocessing |
| Monitoring & maintenance | Structured, stage-by-stage logging with timing and failure tracking on every pipeline step |
| Cloud platform | Dockerized; deployable to AWS/Azure/GCP |

## Project status

Being built and pushed in 3 phases:
- ✅ **Phase 1 — Document Processing Core**: ingestion, OCR, layout detection, section detection, table/figure extraction, reference parsing
- ⏳ **Phase 2 — Intelligence Layer**: LLM enrichment, chunking, embeddings, SQLite + FAISS storage, pipeline orchestrator
- ⏳ **Phase 3 — Serving Layer**: FastAPI endpoints, Streamlit dashboard, Docker, tests

This README documents Phase 1, which is complete.

---

## File Structure & Role of Each File

```
research-intelligence-pipeline/
├── README.md                    This file
├── requirements.txt             All dependencies across all 3 phases (installed once)
├── .env.example                 Template for API keys / config — copy to .env and fill in
├── .gitignore                   Excludes venvs, caches, generated data, DBs
├── phase1_demo.py                Runnable script: PDF in -> structured JSON out
│
├── config/
│   └── settings.py              Central config (paths, model names, chunk size) read from .env.
│                                 Nothing else in the codebase hardcodes a path or parameter —
│                                 change a value once here instead of hunting through every file.
│
├── data/                        Generated at runtime, gitignored except folder placeholders
│   ├── raw/                     Where you drop input PDFs
│   ├── processed/page_images/   Rendered images of scanned pages (for OCR)
│   ├── tables/                  table_1.csv, table_2.csv, ... per paper
│   └── figures/                 figure1.png, figure2.png, ... per paper
│
├── src/
│   ├── ingestion/
│   │   ├── pdf_loader.py        Opens a PDF, extracts native text per page (PyMuPDF), and
│   │   │                        flags any page with too little text as "scanned" for OCR
│   │   └── ocr.py               Runs Tesseract OCR on pages pdf_loader flagged as scanned
│   │
│   ├── parsing/
│   │   ├── layout_detector.py   Classifies every text line as title / heading / body / caption.
│   │   │                        Computes the document's own body-text font size as a baseline
│   │   │                        (rather than a fixed threshold) so it generalizes across
│   │   │                        different paper templates/conferences.
│   │   ├── section_detector.py  Groups body text under the nearest preceding heading
│   │   │                        (Introduction, Methodology, Results, etc.) using whatever
│   │   │                        headings layout_detector actually found in *this* paper —
│   │   │                        not a fixed list of expected section names.
│   │   ├── table_extractor.py   Uses Camelot to detect and extract every table to its own CSV
│   │   ├── figure_extractor.py  Uses PyMuPDF to extract every embedded image to its own file
│   │   └── reference_extractor.py  Splits the References section into individual entries,
│   │                            pulling out a best-effort publication year from each
│   │
│   └── utils/
│       ├── logger.py            Shared structured logging + a @log_stage decorator that times
│       │                        every pipeline stage and logs start/success/failure consistently
│       └── schema.py             The canonical PaperSchema dataclass — every module reads/writes
│                                 this same shape, so parsing, enrichment, storage, and the API
│                                 all agree on one data contract instead of inventing their own dicts
│
└── (api/, dashboard/, tests/, Dockerfile — added in Phases 2 and 3)
```

### How a PDF flows through Phase 1 (see `phase1_demo.py`)
```
PDF
 ├─▶ pdf_loader.load_pdf()          -> per-page text + scanned-page flags
 │      └─▶ ocr.run_ocr_on_document() (only if any page needs it)
 │
 ├─▶ layout_detector.detect_layout() -> classified text blocks (title/heading/body/caption)
 │      └─▶ section_detector.detect_sections() -> {section_name: text}
 │             └─▶ reference_extractor.extract_references() -> parsed bibliography
 │
 ├─▶ table_extractor.extract_tables()   -> CSVs
 ├─▶ figure_extractor.extract_figures() -> PNGs
 │
 └─▶ assembled into a PaperSchema and printed as JSON
```

---

## Tech Stack & Libraries

**Ingestion & Parsing (Phase 1 — implemented)**
- [`PyMuPDF`](https://pymupdf.readthedocs.io/) (`fitz`) — PDF text/image extraction, font metadata
- [`pdfplumber`](https://github.com/jsvine/pdfplumber) — supplementary text/layout extraction
- [`Camelot`](https://camelot-py.readthedocs.io/) — table detection and extraction (requires Ghostscript)
- [`pytesseract`](https://github.com/madmaze/pytesseract) + Tesseract OCR — text extraction from scanned pages
- [`Pillow`](https://python-pillow.org/) — image handling
- [`python-dotenv`](https://github.com/theskumar/python-dotenv) — loads `.env` config

**Intelligence Layer (Phase 2 — planned)**
- `sentence-transformers` — embedding generation for semantic search
- `faiss-cpu` — vector similarity search
- `SQLAlchemy` + SQLite — structured metadata storage
- `openai` (or Gemini API) — LLM-based information extraction (problem statement, contributions, datasets, metrics)

**Serving Layer (Phase 3 — planned)**
- `FastAPI` + `uvicorn` — REST API (upload / search / Q&A endpoints)
- `Streamlit` — interactive dashboard for demoing the pipeline
- `Docker` — containerized deployment

---

## Setup

```bash
pip install -r requirements.txt

# System dependencies:
sudo apt-get install ghostscript tesseract-ocr   # Camelot + OCR need these

cp .env.example .env   # fill in LLM API keys once Phase 2 lands
```

## Try Phase 1

```bash
python phase1_demo.py data/raw/your_paper.pdf
```

Outputs a `PaperSchema` JSON with title, abstract, sections, extracted table/figure file paths, and parsed references. (LLM-derived fields — `problem_statement`, `datasets`, `metrics`, etc. — populate starting in Phase 2.)