from __future__ import annotations

import os
from typing import Iterable

import dspy
import litellm
from dotenv import load_dotenv

load_dotenv()


_MODEL_ID = "openrouter/google/gemini-embedding-001"
_EMBEDDER: dspy.Embedder | None = None


def _openrouter_embedder(texts: list[str], **_kwargs: object) -> list[list[float]]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    api_base = os.getenv("OPENROUTER_API_BASE") or "https://openrouter.ai/api/v1"
    model = _MODEL_ID.split("/", 1)[1] if _MODEL_ID.startswith("openrouter/") else _MODEL_ID
    response = litellm.embedding(
        model=model,
        input=texts,
        api_key=api_key,
        api_base=api_base,
        custom_llm_provider="openai",
        encoding_format="float",
    )
    return [data["embedding"] for data in response.data]


def _get_embedder() -> dspy.Embedder:
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    _EMBEDDER = dspy.Embedder(_openrouter_embedder)
    return _EMBEDDER


def compute_embedding(text: str) -> list[float]:
    embedding = _get_embedder()(text)
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()
    if isinstance(embedding, Iterable):
        return list(embedding)
    raise TypeError("embedder returned unexpected embedding type")
