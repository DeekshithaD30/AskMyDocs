"""AskMyDocs FastAPI application."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config import FRONTEND_FILE, INDEX_DIR, UPLOAD_DIR, ensure_runtime_directories
from embeddings import EmbeddingUnavailableError
from ingestion import ingest_document
from llm import LLMServiceError, generate_answer
from retrieval import retrieve_relevant_chunks

ensure_runtime_directories()

app = FastAPI(
    title="AskMyDocs — RAG Document Intelligence System",
    description="Upload PDFs and ask source-grounded questions.",
    version="1.1.0",
)

# Useful when the static page is served separately during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(max_length=10_000)
    doc_id: str = Field(
        min_length=8,
        max_length=32,
        pattern=r"^[a-f0-9]+$",
        description="The hexadecimal document identifier returned by /upload.",
    )


def _load_registry() -> dict:
    registry_path = INDEX_DIR / "registry.json"
    if not registry_path.exists():
        return {}
    try:
        with open(registry_path, encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail="The document registry is invalid. Re-upload the affected document.",
        ) from exc


def _save_registry(registry: dict) -> None:
    """Atomically persist the document registry."""
    registry_path = INDEX_DIR / "registry.json"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=INDEX_DIR, delete=False
    ) as file:
        json.dump(registry, file, indent=2)
        temp_path = file.name
    os.replace(temp_path, registry_path)


def _is_valid_doc_id(doc_id: str) -> bool:
    """Accept legacy short IDs and current UUID-derived hexadecimal IDs."""
    return bool(re.fullmatch(r"[a-f0-9]{8,32}", doc_id))


@app.get("/", include_in_schema=False)
def frontend():
    """Serve the bundled UI from the same origin as the API."""
    if FRONTEND_FILE.exists():
        return FileResponse(FRONTEND_FILE)
    return {"status": "running", "message": "AskMyDocs API is live. Open /docs for the API."}


@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "running",
        "llm_configured": bool(os.getenv("GEMINI_API_KEY")),
        "frontend_available": FRONTEND_FILE.exists(),
    }


@app.delete("/documents/{doc_id}", tags=["Documents"])
def delete_document(doc_id: str):
    """Delete a document and every local artefact created for it."""
    if not _is_valid_doc_id(doc_id):
        raise HTTPException(status_code=422, detail="Invalid document ID.")

    registry = _load_registry()
    metadata = registry.get(doc_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Every path is built from a validated identifier and a fixed suffix.
    for path in (
        UPLOAD_DIR / f"{doc_id}.pdf",
        INDEX_DIR / f"{doc_id}.faiss",
        INDEX_DIR / f"{doc_id}_chunks.pkl",
        INDEX_DIR / f"{doc_id}_meta.json",
    ):
        path.unlink(missing_ok=True)

    del registry[doc_id]
    _save_registry(registry)
    return {"message": f"Deleted {metadata.get('filename', 'document')}", "doc_id": doc_id}


@app.post("/upload", tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """Save a PDF and run extraction, chunking, embedding, and indexing."""
    filename = Path(file.filename or "document.pdf").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    doc_id = uuid.uuid4().hex
    file_path = UPLOAD_DIR / f"{doc_id}.pdf"
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        result = await run_in_threadpool(ingest_document, str(file_path), doc_id, filename)
    except ValueError as exc:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="The document could not be processed. Check the server logs and try again.",
        ) from exc
    finally:
        await file.close()

    return {
        "doc_id": doc_id,
        "filename": filename,
        "chunks_created": result["chunks"],
        "message": f"Document processed successfully. {result['chunks']} chunks indexed.",
    }


@app.post("/query", tags=["Query"])
async def query_document(request: QueryRequest):
    """Retrieve document context and generate a grounded answer."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    registry = _load_registry()
    metadata = registry.get(request.doc_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Document not found. Upload the PDF first.")
    doc_name = metadata.get("filename", "the document")

    try:
        chunks = await run_in_threadpool(
            retrieve_relevant_chunks, question, request.doc_id, 4
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except EmbeddingUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        result = await run_in_threadpool(generate_answer, question, chunks, doc_name)
    except LLMServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "question": question,
        "answer": result["answer"],
        "sources": result["sources"],
        "doc_id": request.doc_id,
        "doc_name": doc_name,
        "context_preview": result["context_preview"],
    }


@app.get("/documents", tags=["Documents"])
def list_documents():
    registry = _load_registry()
    documents = [
        {
            "doc_id": doc_id,
            "filename": metadata.get("filename", "Untitled PDF"),
            "chunks": metadata.get("chunk_count", 0),
        }
        for doc_id, metadata in registry.items()
    ]
    return {"documents": documents, "count": len(documents)}
