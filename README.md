# AskMyDocs

AskMyDocs is a local Retrieval-Augmented Generation (RAG) application for
asking questions about PDF documents. Upload a PDF, select it from the document
list, and receive a Gemini-generated answer grounded in the most relevant
passages from that document.

The FastAPI backend serves the browser UI itself, so there is no separate
frontend or backend directory.

## Features

- PDF upload, validation, selectable-text extraction, and overlapping chunking
- Semantic retrieval with Sentence Transformers and per-document FAISS indexes
- Source-grounded Gemini answers with retrieved chunk references and previews
- Exact fallback response — `Answer not found from the document.` — when no
  relevant chunks are available
- Document list, selection, and confirmed deletion of the PDF and its local
  retrieval artifacts
- API validation and clear errors for invalid files, unknown documents, missing
  API keys, unavailable embedding models, and Gemini provider failures
- Optional offline hashing embeddings for demos without the MiniLM model
- Local evaluation script reporting keyword hit rate and latency statistics
- Docker support for local containerized runs

## Tech stack

| Area | Technology |
| --- | --- |
| API and UI hosting | FastAPI, Uvicorn |
| UI | Vanilla HTML, CSS, and JavaScript |
| PDF extraction | pypdf |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector search | FAISS |
| Answer generation | Gemini via `google-genai`, with an HTTPS fallback |
| Evaluation | Python, Requests |

## Architecture

```text
PDF upload
  → text extraction
  → overlapping text chunks
  → embeddings
  → FAISS index

Question
  → query embedding
  → top matching chunks
  → Gemini with a source-grounded prompt
  → answer, sources, and context previews
```

Each document receives a separate FAISS index and metadata file. The metadata
records the embedding provider used, preventing query embeddings from being
mixed with an incompatible index.

## Project structure

```text
askmydocs/
├── main.py                       # FastAPI routes and UI entry point
├── config.py                     # Absolute runtime paths
├── ingestion.py                  # PDF extraction, chunks, embeddings, FAISS persistence
├── embeddings.py                 # Lazy MiniLM loading and offline fallback
├── retrieval.py                  # Provider-safe similarity search
├── llm.py                        # Gemini client, fallback, and grounded prompt
├── evaluate.py                   # Optional API evaluation tool
├── index.html                    # Single-page user interface
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container build and startup configuration
├── .env.example                  # Safe environment-variable template
└── test_questions.example.json   # Evaluation input template
```

The `uploads/` and `index/` folders are created automatically at runtime and
are intentionally excluded from Git.

## Installation

Requirements: Python 3.10+ and a Gemini API key.

```bash
git clone https://github.com/YOUR_USERNAME/askmydocs.git
cd askmydocs
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
```

## Environment setup

Edit `.env` and add your own key. Never commit this file.

```dotenv
GEMINI_API_KEY=your_api_key_here
```

Optional configuration:

```dotenv
# Defaults to gemini-3.5-flash-lite
GEMINI_MODEL=gemini-3.5-flash-lite

# Uses deterministic local hashing embeddings instead of downloading MiniLM.
# Re-upload documents after changing this mode.
# ASKMYDOCS_EMBEDDING_MODE=offline
```

## Run the application

```bash
uvicorn main:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000). The interactive API docs
are available at [http://localhost:8000/docs](http://localhost:8000/docs).
Do not open `index.html` directly from the filesystem; it must be served by the
backend.

## Usage

1. Start the server and open the application in a browser.
2. Choose or drag a PDF into the upload panel.
3. Select **Process Document** and wait for indexing to complete.
4. Select the document from **Your Documents**.
5. Ask a question in the chat field.
6. Review the answer, retrieved source chunks, and context preview.
7. Use the **✕** control beside a document to delete it. Confirming deletion
   removes the uploaded PDF, FAISS index, chunks, metadata, and registry entry.

## API

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | Serves the web UI |
| `GET` | `/health` | Application and configuration status |
| `GET` | `/documents` | Lists processed documents |
| `POST` | `/upload` | Uploads and indexes a PDF (`multipart/form-data`, `file`) |
| `POST` | `/query` | Answers a question (`question`, `doc_id`) |
| `DELETE` | `/documents/{doc_id}` | Permanently deletes a document and all local artifacts |

## Evaluation

Start the application, then copy and adapt the template:

```bash
cp test_questions.example.json test_questions.json
python evaluate.py --doc_id YOUR_DOCUMENT_ID --questions test_questions.json
```

The script writes `evaluation_report.json` with per-question answers, source
chunks, latency, keyword hit checks, average latency, and p95 latency. Both
generated files are ignored by Git.

## Docker

```bash
docker build -t askmydocs .
docker run --rm -p 8080:8080 --env-file .env askmydocs
```

Open [http://localhost:8080](http://localhost:8080). Container storage is
ephemeral; mount persistent storage or use managed storage/vector services for
a production deployment.

## Limitations

- Image-only/scanned PDFs require OCR before this application can extract text.
- The first semantic upload downloads the MiniLM embedding model when it is not
  already cached.
- Retrieval quality depends on PDF text quality and the chosen chunking scheme.
- Gemini availability, quotas, and response quality depend on the configured
  Gemini project and model.
- Runtime documents and indexes are local to the running machine/container.

## Future improvements

- OCR support for scanned PDFs
- Persistent cloud object/vector storage
- User authentication and per-user document isolation
- Streaming answers and richer citations with page-level source locations
- Automated retrieval and answer-quality test suites
