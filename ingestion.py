"""PDF ingestion: extract text, chunk it, embed it, and persist a FAISS index."""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path

import faiss
import numpy as np
import pypdf

from config import INDEX_DIR, ensure_runtime_directories
from embeddings import EMBEDDING_DIMENSION, embed_texts

ensure_runtime_directories()

CHUNK_SIZE = 400
OVERLAP = 50


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract selectable text from every page of a PDF."""
    try:
        reader = pypdf.PdfReader(pdf_path)
        full_text = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                full_text.append(f"[Page {page_number}]\n{page_text}")
        return "\n".join(full_text).strip()
    except Exception as exc:
        raise ValueError("The uploaded file is not a readable PDF.") from exc


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP
) -> list[str]:
    """Split document text into overlapping, word-based chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be zero or greater and smaller than chunk_size.")

    words = text.split()
    return [
        " ".join(words[start : start + chunk_size])
        for start in range(0, len(words), chunk_size - overlap)
        if words[start : start + chunk_size]
    ]


def generate_embeddings(chunks: list[str]) -> np.ndarray:
    """Compatibility helper that returns embeddings for callers using this API."""
    embeddings, _ = embed_texts(chunks)
    return embeddings


def build_and_save_faiss_index(embeddings: np.ndarray, doc_id: str) -> faiss.IndexFlatL2:
    """Build and persist an exact L2 FAISS index for one document."""
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("Cannot build an index without at least one embedding.")
    if embeddings.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError("Embedding dimension does not match the configured index dimension.")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
    faiss.write_index(index, str(INDEX_DIR / f"{doc_id}.faiss"))
    return index


def _atomic_json_write(path: Path, data: dict) -> None:
    """Avoid leaving a corrupt registry if the process stops during a write."""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        json.dump(data, tmp, indent=2)
        temp_name = tmp.name
    os.replace(temp_name, path)


def ingest_document(pdf_path: str, doc_id: str, original_filename: str) -> dict:
    """Run the full ingestion pipeline and save all retrieval artefacts."""
    text = extract_text_from_pdf(pdf_path)
    if not text:
        raise ValueError(
            "No selectable text was found. This may be a scanned PDF; use an OCR-enabled PDF instead."
        )

    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No text chunks could be created from this PDF.")

    embeddings, backend = embed_texts(chunks)
    build_and_save_faiss_index(embeddings, doc_id)

    with open(INDEX_DIR / f"{doc_id}_chunks.pkl", "wb") as file:
        pickle.dump(chunks, file)

    _atomic_json_write(
        INDEX_DIR / f"{doc_id}_meta.json",
        {"embedding_backend": backend, "embedding_dimension": int(embeddings.shape[1])},
    )

    registry_path = INDEX_DIR / "registry.json"
    try:
        with open(registry_path, encoding="utf-8") as file:
            registry = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        registry = {}

    registry[doc_id] = {
        "filename": original_filename,
        "chunk_count": len(chunks),
        "char_count": len(text),
        "embedding_backend": backend,
    }
    _atomic_json_write(registry_path, registry)

    return {"doc_id": doc_id, "chunks": len(chunks), "filename": original_filename}
