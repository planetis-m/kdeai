from __future__ import annotations

from typing import Iterable, Sequence
import os

from dotenv import load_dotenv
from openai import OpenAI

from kdeai.config import EmbeddingPolicy

load_dotenv()

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
    request_kwargs = {}
    if target_dim is not None:
        request_kwargs["dimensions"] = target_dim
    response = client.embeddings.create(
        model=model_id,
        input=list(texts),
        **request_kwargs,
    )
    data = _embedding_response_data(response)
    if len(data) != len(texts):
        raise RuntimeError("embedding response length mismatch")
    embeddings: list[list[float]] = []
    for item in data:
        values = _extract_embedding(item)
        if target_dim is not None and len(values) != target_dim:
            if len(values) < target_dim:
                raise ValueError(
                    f"embedding dim mismatch: expected {target_dim}, got {len(values)}"
                )
            values = values[:target_dim]
        embeddings.append(values)
    return embeddings


def compute_embeddings(
    texts: Sequence[str],
    *,
    policy: EmbeddingPolicy,
) -> list[list[float]]:
    if not isinstance(texts, Sequence):
        raise TypeError("texts must be a sequence")
    if not texts:
        return []
    embeddings = _openai_embed(texts, policy=policy)
    if not isinstance(embeddings, Iterable):
        raise TypeError("embedding response is not iterable")
    rows = [list(row) for row in embeddings]
    if len(rows) != len(texts):
        raise RuntimeError("embedder returned unexpected number of embeddings")
    target_dim = policy.dim
    if target_dim is not None:
        trimmed_rows: list[list[float]] = []
        for row in rows:
            values = [float(value) for value in row]
            if len(values) != target_dim:
                if len(values) < target_dim:
                    raise ValueError(
                        f"embedding dim mismatch: expected {target_dim}, got {len(values)}"
                    )
                values = values[:target_dim]
            trimmed_rows.append(values)
        return trimmed_rows
    return [[float(value) for value in row] for row in rows]


def compute_embedding(
    text: str,
    *,
    policy: EmbeddingPolicy,
) -> list[float]:
    embeddings = compute_embeddings([text], policy=policy)
    if not embeddings:
        raise RuntimeError("embedding response empty")
    return embeddings[0]
