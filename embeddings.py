"""Embedding providers used by ingestion and retrieval.

The preferred provider is all-MiniLM-L6-v2. Loading it is deliberately lazy:
the API can start and show useful errors even when the model has not yet been
downloaded. A deterministic hashing fallback keeps the local demo usable in
offline environments; re-upload a document when switching providers.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Literal

import numpy as np

EMBEDDING_DIMENSION = 384
SEMANTIC_BACKEND = "sentence-transformers:all-MiniLM-L6-v2"
HASHING_BACKEND = "hashing-v1"

EmbeddingBackend = Literal[
    "sentence-transformers:all-MiniLM-L6-v2", "hashing-v1"
]

_model = None
_model_error: Exception | None = None
_TOKEN_PATTERN = re.compile(r"[\w']+", re.UNICODE)


class EmbeddingUnavailableError(RuntimeError):
    """Raised when a document requires an embedding provider that is unavailable."""


def _load_semantic_model():
    """Load and cache the optional SentenceTransformer model only when needed."""
    global _model, _model_error

    if _model is not None:
        return _model
    if _model_error is not None:
        raise EmbeddingUnavailableError(
            "The semantic embedding model is unavailable. Check network access, "
            "then restart the server, or re-upload the document using offline mode."
        ) from _model_error

    try:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("all-MiniLM-L6-v2")
        return _model
    except Exception as exc:  # Download/cache failures must not crash API startup.
        _model_error = exc
        raise EmbeddingUnavailableError(
            "Could not load the semantic embedding model. Check network access, "
            "or set ASKMYDOCS_EMBEDDING_MODE=offline and upload the PDF again."
        ) from exc


def _hashing_embeddings(texts: list[str]) -> np.ndarray:
    """Create normalized, fixed-size lexical vectors without any external model."""
    vectors = np.zeros((len(texts), EMBEDDING_DIMENSION), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in _TOKEN_PATTERN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIMENSION
            vectors[row, index] += 1.0 if digest[4] & 1 else -1.0

        magnitude = np.linalg.norm(vectors[row])
        if magnitude:
            vectors[row] /= magnitude
    return vectors


def embed_texts(
    texts: list[str], preferred_backend: str | None = None
) -> tuple[np.ndarray, EmbeddingBackend]:
    """Embed texts with the requested or configured provider.

    ``auto`` (the default) prefers semantic embeddings and falls back to the
    built-in hashing embedder. A persisted index records its provider, so query
    embeddings cannot silently use an incompatible vector space.
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32), HASHING_BACKEND

    configured_mode = os.getenv("ASKMYDOCS_EMBEDDING_MODE", "auto").lower()
    backend = preferred_backend

    if backend is None:
        if configured_mode in {"offline", "hashing"}:
            backend = HASHING_BACKEND
        else:
            backend = SEMANTIC_BACKEND

    if backend == HASHING_BACKEND:
        return _hashing_embeddings(texts), HASHING_BACKEND

    if backend != SEMANTIC_BACKEND:
        raise EmbeddingUnavailableError(f"Unsupported embedding backend: {backend}")

    try:
        model = _load_semantic_model()
        embeddings = model.encode(
            texts,
            show_progress_bar=False,
            batch_size=32,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32), SEMANTIC_BACKEND
    except EmbeddingUnavailableError:
        # Only a new, auto-mode ingestion may fall back. Existing indexes must
        # retain the provider with which they were created.
        if preferred_backend is None and configured_mode == "auto":
            return _hashing_embeddings(texts), HASHING_BACKEND
        raise
