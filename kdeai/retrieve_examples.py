from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence
import logging

from kdeai.config import ExamplesEligibility
from kdeai import examples as kdeexamples
from kdeai import po_utils

logger = logging.getLogger(__name__)

EmbeddingFunc = Callable[[Sequence[str]], Sequence[Sequence[float]]]


def _examples_pointer_path(
    project_root: Path,
    *,
    scope: str,
    lang: str,
) -> Path:
    return (
        project_root
        / ".kdeai"
        / "cache"
        / "examples"
        / scope
        / f"examples.{scope}.{lang}.current.json"
    )


def _read_pointer(path: Path, label: str) -> dict:
    try:
        return po_utils.read_json(path, label)
    except Exception as exc:
        logger.debug("Examples pointer read failed: %s", exc)
        raise


def open_examples_best_effort(
    project_root: Path,
    *,
    scope: str,
    lang: str,
    project_id: str,
    config_hash: str,
    embed_policy_hash: str,
    sqlite_vector_path: str | None,
    required: bool,
) -> kdeexamples.ExamplesDb | None:
    pointer_path = _examples_pointer_path(project_root, scope=scope, lang=lang)
    if not pointer_path.exists():
        return None
    try:
        pointer = _read_pointer(pointer_path, f"examples {scope} pointer")
    except Exception as exc:
        if required:
            raise RuntimeError("examples required but pointer read failed") from exc
        return None
    db_file = pointer.get("db_file")
    if not db_file:
        return None
    db_path = pointer_path.parent / str(db_file)
    if not db_path.exists():
        return None
    try:
        return kdeexamples.open_examples_db(
            db_path,
            project_id=project_id,
            config_hash=config_hash,
            embed_policy_hash=embed_policy_hash,
            sqlite_vector_path=sqlite_vector_path,
        )
    except Exception as exc:
        message = (
            "examples required but DB open/validation failed for "
            f"scope={scope} lang={lang}"
        )
        if required:
            raise RuntimeError(message) from exc
        logger.debug("Examples DB open failed: %s", exc)
        return None


def collect_examples(
    *,
    examples_db: kdeexamples.ExamplesDb | None,
    embedder: EmbeddingFunc | None,
    source_text: str,
    top_n: int,
    lang: str,
    eligibility: ExamplesEligibility,
    review_status_order: Sequence[str],
    required: bool,
) -> list[kdeexamples.ExampleMatch]:
    if examples_db is None or embedder is None:
        return []
    try:
        embeddings = embedder([source_text])
        if len(embeddings) != 1:
            raise ValueError("expected single embedding")
        embedding = embeddings[0]
    except Exception as exc:
        if required:
            raise RuntimeError("examples required but embedding failed") from exc
        return []
    try:
        return kdeexamples.query_examples(
            examples_db,
            query_embedding=embedding,
            top_n=top_n,
            lang=lang,
            eligibility=eligibility,
            review_status_order=review_status_order,
        )
    except Exception as exc:
        if required:
            raise RuntimeError("examples required but query failed") from exc
        return []
