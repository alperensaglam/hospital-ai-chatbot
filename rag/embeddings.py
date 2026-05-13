"""
Embedding generation module.

Uses LiteLLM to call any OpenAI-compatible embedding API.
The model is configured via the EMBEDDING_MODEL env var.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def get_embeddings(texts: list[str], model: str | None = None) -> list[list[float]]:
    """
    Generate embeddings for a list of texts using LiteLLM.

    Returns a list of float vectors, one per input text.
    """
    import litellm

    model = model or EMBEDDING_MODEL
    response = litellm.embedding(model=model, input=texts)
    return [item["embedding"] for item in response.data]


def get_single_embedding(text: str, model: str | None = None) -> list[float]:
    """Convenience wrapper for a single text."""
    return get_embeddings([text], model=model)[0]
