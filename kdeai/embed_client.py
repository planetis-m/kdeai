from __future__ import annotations

from typing import Iterable

import dspy
from dotenv import load_dotenv

load_dotenv()

_MODEL_ID = "openrouter/google/gemini-embedding-001"
_EMBEDDER: dspy.Embedder | None = None


def _get_embedder() -> dspy.Embedder:
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    _EMBEDDER = dspy.Embedder(_MODEL_ID)
    return _EMBEDDER


def compute_embedding(text: str) -> list[float]:
    embedding = _get_embedder()(text)
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()
    if isinstance(embedding, Iterable):
        return list(embedding)
    raise TypeError("embedder returned unexpected embedding type")
