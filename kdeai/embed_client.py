from __future__ import annotations

from typing import Iterable, Sequence
import os
from openai import OpenAI

from kdeai.config import EmbeddingPolicy

_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
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
    _CLIENT = OpenAI(api_key=api_key, base_url=_OPENROUTER_API_BASE)
    return _CLIENT


def _openai_embed(
    texts: Sequence[str],
    *,
    policy: EmbeddingPolicy,
) -> list[list[float]]:
    client = _get_client()
    model_id = policy.model_id
    target_dim = policy.dim
    if not isinstance(target_dim, int) or target_dim <= 0:
        raise ValueError("embedding policy dim must be a positive int")
    request_kwargs = {"dimensions": target_dim}
    response = client.embeddings.create(
        model=model_id,
        input=list(texts),
        **request_kwargs,
    )
    data = _embedding_response_data(response)
    if len(data) != len(texts):
        raise RuntimeError(
            "embedding response length mismatch for "
            f"model={model_id} dim={target_dim}: expected {len(texts)} got {len(data)}"
        )
    embeddings: list[list[float]] = []
    for item in data:
        values = _extract_embedding(item)
        if len(values) != target_dim:
            if len(values) < target_dim:
                raise ValueError(
                    "embedding dim mismatch for "
                    f"model={model_id} dim={target_dim}: expected {target_dim} got {len(values)}"
                )
            values = values[:target_dim]
        embeddings.append(values)
    return embeddings


def _coerce_texts(texts: Sequence[str]) -> list[str]:
    if isinstance(texts, (str, bytes, bytearray)):
        raise TypeError("texts must be a sequence of strings, not a string")
    try:
        values = list(texts)
    except TypeError as exc:
        raise TypeError("texts must be a sequence of strings") from exc
    for value in values:
        if not isinstance(value, str):
            raise TypeError("texts must contain only strings")
    return values


def compute_embeddings(
    texts: Sequence[str],
    *,
    policy: EmbeddingPolicy,
) -> list[list[float]]:
    coerced = _coerce_texts(texts)
    if not coerced:
        return []
    return _openai_embed(coerced, policy=policy)


def compute_embedding(
    text: str,
    *,
    policy: EmbeddingPolicy,
) -> list[float]:
    embeddings = compute_embeddings([text], policy=policy)
    if not embeddings:
        raise RuntimeError("embedding response empty")
    return embeddings[0]
