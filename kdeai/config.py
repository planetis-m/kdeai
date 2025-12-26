from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from kdeai import hash as kdehash


@dataclass(frozen=True)
class Config:
    data: dict
    config_hash: str
    embed_policy_hash: str


def _require_mapping(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _normalize_embed_policy(config: dict) -> dict:
    prompt = _require_mapping(config.get("prompt"), "prompt")
    examples = _require_mapping(prompt.get("examples"), "prompt.examples")
    policy = _require_mapping(examples.get("embedding_policy"), "prompt.examples.embedding_policy")

    required_keys = [
        "model_id",
        "dim",
        "distance",
        "encoding",
        "input_canonicalization",
        "normalization",
    ]
    missing = [key for key in required_keys if key not in policy]
    if missing:
        raise ValueError(f"embedding_policy missing required keys: {', '.join(missing)}")

    model_id = str(policy["model_id"])
    dim = int(policy["dim"])
    distance = str(policy["distance"]).lower()
    encoding = str(policy["encoding"])
    input_canonicalization = str(policy["input_canonicalization"])
    normalization = str(policy["normalization"])
    require_finite = policy.get("require_finite", True)

    if encoding != "float32_le":
        raise ValueError("embedding_policy.encoding must be 'float32_le'")
    if input_canonicalization != "source_text_v1":
        raise ValueError("embedding_policy.input_canonicalization must be 'source_text_v1'")

    return {
        "model_id": model_id,
        "dim": dim,
        "distance": distance,
        "encoding": encoding,
        "input_canonicalization": input_canonicalization,
        "normalization": normalization,
        "require_finite": bool(require_finite),
    }


def compute_config_hash(config: dict) -> str:
    return kdehash.sha256_hex_text(kdehash.canonical_json(config))


def compute_embed_policy_hash(config: dict) -> str:
    policy = _normalize_embed_policy(config)
    return kdehash.sha256_hex_text(kdehash.canonical_json(policy))


def load_config(config_path: Path) -> Config:
    config = _require_mapping(
        json.loads(config_path.read_text(encoding="utf-8")),
        "config",
    )
    return Config(
        data=config,
        config_hash=compute_config_hash(config),
        embed_policy_hash=compute_embed_policy_hash(config),
    )


def load_config_from_root(root: Path) -> Config:
    return load_config(root / ".kdeai" / "config.json")
