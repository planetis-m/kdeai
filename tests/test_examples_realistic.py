from __future__ import annotations

import math
import os
from pathlib import Path
import time
import sqlite3

import polib
import pytest
import httpx
import openai

from conftest import build_config
from kdeai import db as kdedb
from kdeai import examples
from kdeai import hash as kdehash
from kdeai import po_model
from kdeai import prompt
from kdeai import workspace_tm
from kdeai.config import Config
from kdeai.embed_client import compute_embedding


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


def _write_trimmed_po(src_path: Path, dest_path: Path, *, max_entries: int) -> list[str]:
    po = polib.pofile(str(src_path))
    new_po = polib.POFile()
    new_po.metadata = dict(po.metadata)

    selected: list[str] = []
    for entry in po:
        if entry.obsolete:
            continue
        has_singular = bool(entry.msgstr.strip())
        has_plural = any(value.strip() for value in entry.msgstr_plural.values())
        if not (has_singular or has_plural):
            continue
        new_entry = polib.POEntry(
            msgid=entry.msgid,
            msgstr=entry.msgstr,
            msgid_plural=entry.msgid_plural,
            msgctxt=entry.msgctxt,
            msgstr_plural=dict(entry.msgstr_plural),
        )
        new_entry.flags = list(entry.flags)
        new_po.append(new_entry)
        selected.append(entry.msgid)
        if len(selected) >= max_entries:
            break

    if len(selected) < max_entries:
        raise AssertionError("not enough translated entries in playground file")

    new_po.save(str(dest_path))
    return selected


def test_compute_embedding_returns_real_values() -> None:
    _load_env_if_missing(["OPENROUTER_API_KEY"])
    assert os.getenv("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY must be set"

    config = build_config(
        {
            "prompt": {
                "examples": {
                    "embedding_policy": {
                        "model_id": "google/gemini-embedding-001",
                        "dim": 768,
                    }
                }
            }
        }
    )
    try:
        embedding = compute_embedding("File", policy=config.prompt.examples.embedding_policy)
    except Exception as exc:
        if isinstance(exc, (openai.APIConnectionError, httpx.ConnectError)):
            pytest.skip("embedding provider unreachable in test environment")
        raise
    assert isinstance(embedding, list)
    assert embedding, "embedding must be non-empty"
    assert all(math.isfinite(value) for value in embedding)


def _build_examples_db_from_playground(tmp_path: Path) -> tuple[Path, po_model.PoUnit, Config]:
    project_root = Path(__file__).resolve().parents[1]

    source_po = project_root / "tests" / "playground" / "dolphin.po"
    trimmed_po = tmp_path / "sample.po"
    _write_trimmed_po(source_po, trimmed_po, max_entries=2)

    data = trimmed_po.read_bytes()
    units = po_model.parse_po_bytes(data)
    if not units:
        raise AssertionError("expected parsed units from trimmed playground file")
    unit = units[0]

    db_path = tmp_path / "workspace.tm.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(kdedb.WORKSPACE_TM_SCHEMA)
    meta = {
        "schema_version": "1",
        "kind": "workspace_tm",
        "project_id": "test-project",
        "config_hash": "test-config",
        "created_at": "2024-01-01T00:00:00Z",
    }
    conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta.items())
    conn.commit()

    config = build_config(
        {
            "tm": {"selection": {"review_status_order": ["reviewed", "draft"]}},
            "prompt": {
                "examples": {
                    "embedding_policy": {
                        "model_id": "google/gemini-embedding-001",
                        "dim": 768,
                        "normalization": "none",
                    },
                    "eligibility": {
                        "min_review_status": "draft",
                        "allow_ai_generated": False,
                    },
                }
            },
        }
    )

    workspace_tm.index_file_snapshot_tm(
        conn,
        file_path=str(trimmed_po),
        lang="de",
        bytes=data,
        sha256=kdehash.sha256_hex_bytes(data),
        mtime_ns=time.time_ns(),
        size=len(data),
        config=config,
    )

    def embedder(texts: list[str]) -> list[list[float]]:
        policy = config.prompt.examples.embedding_policy
        return [compute_embedding(text, policy=policy) for text in texts]

    output_path = tmp_path / "examples.sqlite"
    examples.build_examples_db_from_workspace(
        conn,
        output_path=output_path,
        lang="de",
        config=config,
        project_id="test-project",
        config_hash="test-config",
        embed_policy_hash="test-embed-policy",
        embedder=embedder,
    )
    conn.close()
    return output_path, unit, config


def test_build_examples_db_from_playground(tmp_path: Path, monkeypatch) -> None:
    _load_env_if_missing(["OPENROUTER_API_KEY"])
    assert os.getenv("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY must be set"

    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(project_root)

    try:
        output_path, unit, config = _build_examples_db_from_playground(tmp_path)
    except Exception as exc:
        if isinstance(exc, (openai.APIConnectionError, httpx.ConnectError)):
            pytest.skip("embedding provider unreachable in test environment")
        raise

    db = examples.open_examples_db(
        output_path,
        project_id="test-project",
        config_hash="test-config",
        embed_policy_hash="test-embed-policy",
    )
    kdedb.enable_sqlite_vector(db.conn, extension_path="./vector.so")
    try:
        matches = examples.query_examples(
            db,
            query_embedding=compute_embedding(
                unit.source_text, policy=config.prompt.examples.embedding_policy
            ),
            top_n=2,
            lang="de",
        )
        assert matches, "expected at least one example match"
    finally:
        db.conn.close()


def test_retrieve_few_shot_examples_from_db(tmp_path: Path, monkeypatch) -> None:
    _load_env_if_missing(["OPENROUTER_API_KEY"])
    assert os.getenv("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY must be set"

    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(project_root)

    try:
        output_path, unit, config = _build_examples_db_from_playground(tmp_path)
    except Exception as exc:
        if isinstance(exc, (openai.APIConnectionError, httpx.ConnectError)):
            pytest.skip("embedding provider unreachable in test environment")
        raise

    db = examples.open_examples_db(
        output_path,
        project_id="test-project",
        config_hash="test-config",
        embed_policy_hash="test-embed-policy",
    )
    kdedb.enable_sqlite_vector(db.conn, extension_path="./vector.so")
    try:
        matches = examples.query_examples(
            db,
            query_embedding=compute_embedding(
                unit.source_text, policy=config.prompt.examples.embedding_policy
            ),
            top_n=2,
            lang="de",
        )
        assert matches, "expected example matches"
        payload = prompt.build_prompt_payload(
            config=build_config({"languages": {"source": "en"}}),
            msgctxt=unit.msgctxt,
            msgid=unit.msgid,
            msgid_plural=unit.msgid_plural,
            target_lang="de",
            examples=matches,
            glossary=[],
        )
        assert "1. Source:" in payload["few_shot_examples"]
        assert "Translation:" in payload["few_shot_examples"]
    finally:
        db.conn.close()
