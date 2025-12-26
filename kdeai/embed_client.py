from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import Client

load_dotenv()


def _get_client() -> Client:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return Client(base_url="https://openrouter.ai/api/v1", api_key=api_key)


_CLIENT = _get_client()


def compute_embedding(text: str) -> list[float]:
    response = _CLIENT.embeddings.create(
        model="google/gemini-embedding-001",
        input=text,
    )
    return list(response.data[0].embedding)
