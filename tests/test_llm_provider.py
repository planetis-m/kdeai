from __future__ import annotations

import os
from pathlib import Path

import dspy

from kdeai.config import Config
from kdeai.llm_provider import configure_dspy


def _load_env_if_missing(keys: list[str]) -> None:
    if any(os.getenv(key) for key in keys):
        return
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        name = name.strip()
        if name and name not in os.environ:
            os.environ[name] = value.strip()


def test_configure_dspy_uses_generation_model_id() -> None:
    _load_env_if_missing(["OPENROUTER_API_KEY"])
    assert os.getenv("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY must be set for DSPy usage"

    config = Config(
        data={
            "prompt": {
                "generation_model_id": "openrouter/x-ai/grok-4-fast",
                "examples": {
                    "embedding_policy": {
                        "model_id": "openrouter/google/gemini-embedding-001",
                    }
                },
            }
        },
        config_hash="test",
        embed_policy_hash="test",
    )

    configure_dspy(config)

    lm = dspy.settings.lm
    assert lm is not None
    assert getattr(lm, "model", None) == "openrouter/x-ai/grok-4-fast"


def test_configure_dspy_falls_back_to_examples_embedding_model() -> None:
    _load_env_if_missing(["OPENROUTER_API_KEY"])
    assert os.getenv("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY must be set for DSPy usage"

    config = Config(
        data={
            "prompt": {
                "examples": {
                    "embedding_policy": {
                        "model_id": "openrouter/google/gemini-embedding-001",
                    }
                }
            }
        },
        config_hash="test",
        embed_policy_hash="test",
    )

    configure_dspy(config)

    lm = dspy.settings.lm
    assert lm is not None
    assert getattr(lm, "model", None) == "openrouter/google/gemini-embedding-001"
