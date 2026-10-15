# Context ingestion pipeline

Phase 1 of a context-only RAG chatbot. The pipeline watches `Context-fetcher/`, extracts PDFs, scanned documents, images, and Excel workbooks, creates citation-aware chunks, and stores them in SQLite for later embedding.

## Setup

Python 3.11 is recommended. Docling downloads its layout/OCR models on first use; later runs can work offline with the cached models.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
Copy-Item .env.example .env
```

Place supported files in `Context-fetcher/`, then run:

```powershell
python -m app ingest
python -m app status
python -m app inspect "example.pdf"
python -m app watch
```

Supported extensions: `.pdf`, `.xlsx`, `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`, and `.webp`.

The database is `data/context.db`; lossless normalized extraction artifacts are stored by content hash in `data/extracted/`. Both locations are intentionally ignored by Git.

## Behavior

- Startup and every watcher event batch trigger a complete reconciliation.
- SHA-256 hashes prevent redundant OCR and allow rename detection.
- New versions replace their chunks transactionally; a failed update preserves the last successful chunks.
- Unsupported or corrupt files are recorded as failures without stopping other files.
- Excel citations include sheet, row, and cell ranges. PDF and image citations retain Docling page provenance and bounding boxes when available.

Run tests with:

```powershell
python -m pytest
```

