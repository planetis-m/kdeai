from __future__ import annotations

from typing import Iterable, Mapping, Sequence
import math
import os

import dspy
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_DEFAULT_MODEL_ID = "google/gemini-embedding-001"
_CLIENT: OpenAI | None = None


def _embedding_response_data(response: object) -> list[object]:
    if isinstance(response, dict):
        data = response.get("data")
    else:
        data = getattr(response, "data", None)
    if not isinstance(data, list):
        raise TypeError("embedding response missing data list")
    return data


def _extract_embedding(item: object) -> list[float]:
    if isinstance(item, dict):
        embedding = item.get("embedding")
    else:
        embedding = getattr(item, "embedding", None)
    if not isinstance(embedding, Iterable):
        raise TypeError("embedding item missing embedding list")
    return [float(value) for value in embedding]


def _get_client() -> OpenAI:
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY must be set for embeddings")
    api_base = os.getenv("OPENROUTER_API_BASE") or "https://openrouter.ai/api/v1"
    _CLIENT = OpenAI(api_key=api_key, base_url=api_base)
    return _CLIENT


def _normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return values
    return [value / norm for value in values]


def _openai_embed(
    texts: Sequence[str],
    *,
    policy: Mapping[str, object] | None,
) -> list[list[float]]:
    client = _get_client()
    model_id = os.getenv("KDEAI_EMBED_MODEL", _DEFAULT_MODEL_ID)
    policy = dict(policy or {})
    normalization = str(policy.get("normalization", "none"))
    target_dim = policy.get("dim")
    target_dim = int(target_dim) if target_dim is not None else None
    request_kwargs = {}
    if target_dim is not None:
        request_kwargs["dimensions"] = target_dim
    response = client.embeddings.create(
        model=model_id,
        input=list(texts),
        **request_kwargs,
    )
    data = _embedding_response_data(response)
    embeddings: list[list[float]] = []
    for item in data:
        values = _extract_embedding(item)
        if target_dim is not None and len(values) != target_dim:
            if len(values) < target_dim:
                raise ValueError(
                    f"embedding dim mismatch: expected {target_dim}, got {len(values)}"
                )
            values = values[:target_dim]
        if normalization == "l2_normalize":
            values = _normalize(values)
        embeddings.append(values)
    return embeddings


def _get_embedder(policy: Mapping[str, object] | None) -> dspy.Embedder:
    return dspy.Embedder(lambda texts: _openai_embed(texts, policy=policy))


def compute_embeddings(
    texts: Sequence[str],
    *,
    policy: Mapping[str, object] | None = None,
) -> list[list[float]]:
    embedder = _get_embedder(policy)
    embeddings = embedder(list(texts))
    if hasattr(embeddings, "tolist"):
        embeddings = embeddings.tolist()
    if isinstance(embeddings, Iterable):
        return [list(row) for row in embeddings]
    raise TypeError("embedder returned unexpected embedding type")


def compute_embedding(
    text: str,
    *,
    policy: Mapping[str, object] | None = None,
) -> list[float]:
    embeddings = compute_embeddings([text], policy=policy)
    if not embeddings:
        raise RuntimeError("embedding response empty")
    return embeddings[0]
