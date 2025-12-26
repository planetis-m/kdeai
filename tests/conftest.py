from __future__ import annotations

import asyncio
import copy
import sys
from pathlib import Path

from kdeai.config import Config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_EVENT_LOOP: asyncio.AbstractEventLoop | None = None


def pytest_sessionstart() -> None:
    global _EVENT_LOOP
    try:
        asyncio.get_running_loop()
        return
    except RuntimeError:
        pass
    _EVENT_LOOP = asyncio.new_event_loop()
    asyncio.set_event_loop(_EVENT_LOOP)


def pytest_sessionfinish() -> None:
    global _EVENT_LOOP
    if _EVENT_LOOP is None:
        return
    _EVENT_LOOP.close()
    _EVENT_LOOP = None
    asyncio.set_event_loop(None)


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def build_config_dict(overrides: dict | None = None) -> dict:
    base = {
        "format": 2,
        "languages": {"source": "en", "targets": ["de"]},
        "markers": {
            "ai_flag": "kdeai-ai",
            "comment_prefixes": {
                "tool": "KDEAI:",
                "ai": "KDEAI-AI:",
                "tm": "KDEAI-TM:",
                "review": "KDEAI-REVIEW:",
            },
        },
        "tm": {
            "lookup_scopes": ["session", "workspace", "reference"],
            "selection": {
                "review_status_order": ["reviewed", "draft", "needs_review", "unreviewed"],
                "prefer_human": True,
            },
        },
        "prompt": {
            "generation_model_id": "test-generation-model",
            "examples": {
                "mode_default": "auto",
                "lookup_scopes": ["workspace", "reference"],
                "top_n": 6,
                "embedding_policy": {
                    "model_id": "provider/model@version",
                    "dim": 384,
                    "distance": "cosine",
                    "encoding": "float32_le",
                    "normalization": "none",
                    "input_canonicalization": "source_text_v1",
                    "require_finite": True,
                },
                "eligibility": {
                    "min_review_status": "reviewed",
                    "allow_ai_generated": False,
                },
            },
            "glossary": {
                "mode_default": "auto",
                "lookup_scopes": ["reference"],
                "spacy_model": "en_core_web_sm",
                "normalization_id": "kdeai_glossary_norm_v1",
                "max_terms": 10,
            },
        },
        "apply": {
            "mode_default": "strict",
            "overwrite_default": "conservative",
            "tagging": {
                "tm_copy": {
                    "add_flags": ["fuzzy"],
                    "add_ai_flag": False,
                    "comment_prefix_key": "tm",
                },
                "llm": {
                    "add_flags": ["fuzzy"],
                    "add_ai_flag": True,
                    "comment_prefix_key": "ai",
                },
            },
        },
        "sqlite": {
            "workspace_tm": {
                "synchronous": "normal",
                "busy_timeout_ms": {"read": 5000, "write": 50},
            }
        },
    }
    if overrides:
        return _deep_merge(base, overrides)
    return base


def build_config(overrides: dict | None = None) -> Config:
    return Config.model_validate(build_config_dict(overrides))
