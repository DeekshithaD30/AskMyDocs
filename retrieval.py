"""Retrieve the most relevant indexed chunks for a document question."""

from __future__ import annotations

import json
import pickle

import faiss

from config import INDEX_DIR
from embeddings import EMBEDDING_DIMENSION, SEMANTIC_BACKEND, embed_texts


def _load_metadata(doc_id: str) -> dict:
    metadata_path = INDEX_DIR / f"{doc_id}_meta.json"
    if not metadata_path.exists():
        # Indexes created before metadata existed used MiniLM.
        return {
            "embedding_backend": SEMANTIC_BACKEND,
            "embedding_dimension": EMBEDDING_DIMENSION,
        }
    with open(metadata_path, encoding="utf-8") as file:
        return json.load(file)


def retrieve_relevant_chunks(query: str, doc_id: str, top_k: int = 4) -> list[dict]:
    """Embed a question and return up to ``top_k`` closest document chunks."""
    if not query or not query.strip():
        return []
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    index_path = INDEX_DIR / f"{doc_id}.faiss"
    chunks_path = INDEX_DIR / f"{doc_id}_chunks.pkl"
    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            f"No index found for document '{doc_id}'. Upload the PDF again and retry."
        )

    index = faiss.read_index(str(index_path))
    with open(chunks_path, "rb") as file:
        chunks: list[str] = pickle.load(file)

    metadata = _load_metadata(doc_id)
    backend = metadata.get("embedding_backend", SEMANTIC_BACKEND)
    query_embedding, _ = embed_texts([query], preferred_backend=backend)
    if index.d != query_embedding.shape[1]:
        raise ValueError(
            "This index is incompatible with its embedding configuration. Re-upload the PDF."
        )

    distances, indices = index.search(query_embedding, min(top_k, len(chunks)))
    return [
        {
            "chunk": chunks[idx],
            "chunk_index": int(idx),
            "distance": float(distance),
            "rank": rank,
        }
        for rank, (idx, distance) in enumerate(zip(indices[0], distances[0]), start=1)
        if 0 <= idx < len(chunks)
    ]
