# Contexted RAG

A local-first, context-only chatbot for PDFs, scanned documents, images, and Excel workbooks. Drop files into `Context-fetcher/`; the application extracts and watches them, builds a hybrid search index, and answers with source citations.

## What it includes

- Docling layout extraction and RapidOCR for scans and images
- Sheet, row, and cell provenance for Excel workbooks
- Incremental SHA-256 file monitoring with safe update, rename, and delete handling
- Structure-aware chunks stored in SQLite
- Local MiniLM embeddings stored in Qdrant local mode
- Dense semantic search plus SQLite FTS5/BM25, fused with reciprocal rank fusion
- Strict refusal when indexed evidence is missing
- Validated file/page/sheet citations on every supported answer
- Provider-neutral OpenAI-compatible generation, configured for Groq by default
- Responsive local web chat, source inventory, and indexing status

Files and embeddings stay on the computer. Only the small set of retrieved excerpts needed for a question is sent to the configured answer provider.

## Setup

Python 3.11 is recommended. Docling and the embedding model download their model files on first use and reuse the local cache afterward.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
Copy-Item .env.example .env
```

Open `.env` and add an API key:

```dotenv
CONTEXT_LLM_API_KEY=your_key_here
```

The default endpoint is Groq. Any OpenAI-compatible provider can be used by changing `CONTEXT_LLM_BASE_URL` and `CONTEXT_LLM_MODEL`. A local Ollama or llama.cpp endpoint can be configured through the same settings.

## Run the application

Place supported files in `Context-fetcher/`, then start the local app:

```powershell
python -m app serve
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The server reconciles the folder at startup and continuously watches for changes.

Useful CLI commands:

```powershell
python -m app ingest                  # extract changed documents and embed new chunks
python -m app index                   # embed any pending chunks
python -m app search "invoice total" # inspect hybrid retrieval
python -m app status                  # inspect file state
python -m app inspect "example.pdf"   # inspect chunks and citations
python -m app watch                   # extraction-only watcher
```

Supported extensions: `.pdf`, `.xlsx`, `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`, and `.webp`.

## Data and safety

- SQLite state and cached extraction artifacts live under `data/`.
- Dense vectors live in `data/qdrant/`.
- `data/`, `.env`, and source documents are excluded from Git.
- Document text is treated as untrusted data and cannot override the system instruction.
- Generated source IDs are checked against the retrieved set before an answer is returned.
- An updated document replaces its old chunks transactionally. Failed updates preserve the last known-good version.

## Tests

```powershell
python -m pytest
```

The suite covers hashing, chunking, table-header repetition, database transactions, lifecycle reconciliation, stale-vector removal, hybrid exact-term retrieval, grounding refusal, and citation validation.
