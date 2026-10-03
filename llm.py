"""Grounded Gemini answer generation using Google's supported GenAI SDK."""

from __future__ import annotations

import logging
import os

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
DEFAULT_MODEL = "gemini-3.5-flash-lite"
_client = None


class LLMServiceError(RuntimeError):
    """An expected configuration or provider error suitable for an API response."""


def _get_client():
    """Create the GenAI client only when a question is actually submitted."""
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise LLMServiceError(
            "GEMINI_API_KEY is not configured. Add it to .env and restart the server."
        )

    try:
        from google import genai
    except ImportError:
        # Existing projects can keep working until dependencies are refreshed.
        # New installations use the supported SDK declared in requirements.txt.
        return None

    try:
        _client = genai.Client(api_key=api_key)
        return _client
    except Exception as exc:
        logger.exception("Unable to initialize the Gemini client")
        raise LLMServiceError(
            "Gemini could not be initialized. Install the project dependencies and verify GEMINI_API_KEY."
        ) from exc


def _provider_error_message(exc: Exception, model: str) -> str:
    """Turn common provider failures into safe, actionable UI messages."""
    response = getattr(exc, "response", None)
    status = (
        getattr(exc, "status_code", None)
        or getattr(response, "status_code", None)
        or getattr(exc, "code", None)
    )
    try:
        status = int(status)
    except (TypeError, ValueError):
        status = None

    if status in {401, 403}:
        return "Gemini rejected the API key. Create or enable a Gemini API key, then restart the server."
    if status == 404:
        return f"Gemini model '{model}' is unavailable for this API key. Set GEMINI_MODEL to an available Flash model and restart."
    if status == 429:
        return "Gemini request limit reached. Wait a moment or check your API quota, then try again."
    return "Gemini could not generate an answer. Check the server log for the provider error and try again."


def _generate_with_rest(api_key: str, model: str, prompt: str) -> str:
    """Use the public Generate Content endpoint when the new SDK is absent."""
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": api_key},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1024},
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    parts = payload.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    answer = "".join(part.get("text", "") for part in parts).strip()
    if not answer:
        raise LLMServiceError("Gemini returned no usable answer. Please try again.")
    return answer


def build_prompt(query: str, chunks: list[dict], doc_name: str) -> str:
    """Create a strict, source-grounded RAG prompt."""
    context = "\n\n".join(
        f"[Source: Chunk {chunk['chunk_index'] + 1}]\n{chunk['chunk']}"
        for chunk in chunks
    )
    return f'''You are a precise document assistant for the document titled "{doc_name}".

STRICT RULES:
- Answer only from the supplied context.
- Do not use outside knowledge or assumptions.
- If the answer is absent, respond exactly: "Answer not found from the document."
- Be concise and cite the chunk numbers you used.

CONTEXT FROM DOCUMENT:
{context}

USER QUESTION: {query}

ANSWER:'''


def generate_answer(query: str, chunks: list[dict], doc_name: str) -> dict:
    """Generate an answer through the current, supported Google GenAI SDK."""
    if not chunks:
        return {
            "answer": "Answer not found from the document.",
            "sources": [],
            "context_preview": [],
        }

    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    try:
        prompt = build_prompt(query, chunks, doc_name)
        client = _get_client()
        if client is None:
            answer = _generate_with_rest(os.environ["GEMINI_API_KEY"], model, prompt)
        else:
            from google.genai import types

            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=1024,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                ),
            )
            answer = (response.text or "").strip()
            if not answer:
                raise LLMServiceError("Gemini returned no usable answer. Please try again.")
    except LLMServiceError:
        raise
    except Exception as exc:
        logger.exception("Gemini generation failed for model %s", model)
        raise LLMServiceError(_provider_error_message(exc, model)) from exc

    return {
        "answer": answer,
        "sources": [f"Chunk {chunk['chunk_index'] + 1}" for chunk in chunks],
        "context_preview": [f"{chunk['chunk'][:200]}..." for chunk in chunks],
    }
