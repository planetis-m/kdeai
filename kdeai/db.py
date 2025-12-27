from __future__ import annotations

from pathlib import Path
from typing import Mapping
import sqlite3

WORKSPACE_TM_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE files (
  file_path TEXT NOT NULL,
  lang TEXT NOT NULL,
  indexed_sha256 TEXT NOT NULL,
  indexed_mtime_ns INTEGER NOT NULL,
  indexed_size INTEGER NOT NULL,
  indexed_at TEXT NOT NULL,
  PRIMARY KEY (file_path, lang)
);

CREATE TABLE sources (
  source_key TEXT PRIMARY KEY,
  msgctxt TEXT NOT NULL DEFAULT '',
  msgid TEXT NOT NULL,
  msgid_plural TEXT NOT NULL DEFAULT '',
  source_text TEXT NOT NULL
);

CREATE TABLE translations (
  file_path TEXT NOT NULL,
  lang TEXT NOT NULL,
  source_key TEXT NOT NULL,
  msgstr TEXT NOT NULL DEFAULT '',
  msgstr_plural TEXT NOT NULL DEFAULT '{}',
  review_status TEXT NOT NULL,
  is_ai_generated INTEGER NOT NULL DEFAULT 0,
  translation_hash TEXT NOT NULL,
  PRIMARY KEY (file_path, lang, source_key),
  FOREIGN KEY (file_path, lang)
    REFERENCES files(file_path, lang)
    ON DELETE CASCADE
);

CREATE TABLE best_translations (
  source_key TEXT NOT NULL,
  lang TEXT NOT NULL,
  file_path TEXT NOT NULL,
  msgstr TEXT NOT NULL DEFAULT '',
  msgstr_plural TEXT NOT NULL DEFAULT '{}',
  review_status TEXT NOT NULL,
  is_ai_generated INTEGER NOT NULL DEFAULT 0,
  translation_hash TEXT NOT NULL,
  PRIMARY KEY (source_key, lang)
);

CREATE INDEX idx_trans_lookup ON translations(source_key, lang);
CREATE INDEX idx_best_lang ON best_translations(lang);
"""

SUPPORTED_SCHEMA_VERSION = "1"

REFERENCE_TM_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE sources (
  source_key TEXT PRIMARY KEY,
  msgctxt TEXT NOT NULL DEFAULT '',
  msgid TEXT NOT NULL,
  msgid_plural TEXT NOT NULL DEFAULT '',
  source_text TEXT NOT NULL
);

CREATE TABLE translations (
  source_key TEXT NOT NULL,
  lang TEXT NOT NULL,
  file_path TEXT NOT NULL,
  file_sha256 TEXT NOT NULL,
  msgstr TEXT NOT NULL DEFAULT '',
  msgstr_plural TEXT NOT NULL DEFAULT '{}',
  review_status TEXT NOT NULL,
  is_ai_generated INTEGER NOT NULL DEFAULT 0,
  translation_hash TEXT NOT NULL,
  PRIMARY KEY (source_key, lang, file_path)
);

CREATE TABLE best_translations (
  source_key TEXT NOT NULL,
  lang TEXT NOT NULL,
  file_path TEXT NOT NULL,
  file_sha256 TEXT NOT NULL,
  msgstr TEXT NOT NULL DEFAULT '',
  msgstr_plural TEXT NOT NULL DEFAULT '{}',
  review_status TEXT NOT NULL,
  is_ai_generated INTEGER NOT NULL DEFAULT 0,
  translation_hash TEXT NOT NULL,
  PRIMARY KEY (source_key, lang)
);

CREATE INDEX idx_ref_lookup ON translations(source_key, lang);
CREATE INDEX idx_ref_best_lang ON best_translations(lang);
"""

EXAMPLES_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE examples (
  id INTEGER PRIMARY KEY,
  source_key TEXT NOT NULL UNIQUE,
  source_text TEXT NOT NULL,
  lang TEXT NOT NULL,
  msgstr TEXT NOT NULL DEFAULT '',
  msgstr_plural TEXT NOT NULL DEFAULT '{}',
  review_status TEXT NOT NULL,
  is_ai_generated INTEGER NOT NULL DEFAULT 0,
  translation_hash TEXT NOT NULL,
  file_path TEXT NOT NULL DEFAULT '',
  file_sha256 TEXT NOT NULL DEFAULT '',
  embedding BLOB NOT NULL
);

CREATE INDEX idx_examples_lang ON examples(lang);
CREATE INDEX idx_examples_quality ON examples(review_status, is_ai_generated);
"""

GLOSSARY_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE terms (
  term_key TEXT NOT NULL,
  src_lang TEXT NOT NULL,
  tgt_lang TEXT NOT NULL,

  src_surface TEXT NOT NULL,
  src_lemma_seq_json TEXT NOT NULL,
  token_count INTEGER NOT NULL,

  tgt_primary TEXT NOT NULL,
  tgt_alternates_json TEXT NOT NULL DEFAULT '[]',

  freq INTEGER NOT NULL,
  score REAL NOT NULL,

  evidence_msgid TEXT NOT NULL DEFAULT '',
  evidence_msgstr TEXT NOT NULL DEFAULT '',

  file_path TEXT NOT NULL DEFAULT '',
  source_key TEXT NOT NULL DEFAULT '',
  file_sha256 TEXT NOT NULL DEFAULT '',

  PRIMARY KEY (src_lang, tgt_lang, term_key)
);

CREATE INDEX idx_terms_lang_pair ON terms(src_lang, tgt_lang);
CREATE INDEX idx_terms_token_count ON terms(src_lang, tgt_lang, token_count);
"""


def _apply_pragma(conn: sqlite3.Connection, name: str, value: object) -> None:
    if isinstance(value, bool):
        rendered = "ON" if value else "OFF"
    else:
        rendered = str(value)
    conn.execute(f"PRAGMA {name}={rendered}")


def configure_workspace_tm(
    conn: sqlite3.Connection,
    *,
    busy_timeout_ms: int,
    synchronous: str = "NORMAL",
) -> None:
    _apply_pragma(conn, "journal_mode", "WAL")
    _apply_pragma(conn, "busy_timeout", int(busy_timeout_ms))
    _apply_pragma(conn, "foreign_keys", True)
    _apply_pragma(conn, "synchronous", str(synchronous).upper())


def connect_workspace_tm(
    path: Path,
    *,
    busy_timeout_ms: int,
    synchronous: str = "NORMAL",
) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    configure_workspace_tm(conn, busy_timeout_ms=busy_timeout_ms, synchronous=synchronous)
    return conn


def connect_readonly(path: Path, *, busy_timeout_ms: int | None = None) -> sqlite3.Connection:
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    if busy_timeout_ms is not None:
        _apply_pragma(conn, "busy_timeout", int(busy_timeout_ms))
    _apply_pragma(conn, "query_only", True)
    return conn


def connect_writable(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(str(path))


def try_enable_sqlite_vector(conn: sqlite3.Connection, *, extension_path: str) -> bool:
    """Attempt to load sqlite-vector without leaving extension loading enabled."""
    conn.enable_load_extension(True)
    try:
        conn.load_extension(extension_path)
    except sqlite3.Error:
        return False
    finally:
        conn.enable_load_extension(False)
    return True


def read_meta(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute("SELECT key, value FROM meta").fetchall()
    return {str(key): str(value) for key, value in rows}


def _require_keys(meta: Mapping[str, str], keys: list[str]) -> None:
    missing = [key for key in keys if key not in meta or meta[key] == ""]
    if missing:
        raise ValueError(f"meta missing required keys: {', '.join(missing)}")


def _require_int(meta: Mapping[str, str], key: str) -> int:
    _require_keys(meta, [key])
    try:
        return int(meta[key])
    except ValueError as exc:
        raise ValueError(f"meta key {key} must be an int") from exc


def _parse_bool_text(value: str, key: str) -> None:
    normalized = value.strip().lower()
    if normalized in {"0", "1", "false", "true"}:
        return
    raise ValueError(f"meta key {key} must be a boolean-like string")


def validate_meta(
    meta: Mapping[str, str],
    *,
    expected_project_id: str,
    expected_config_hash: str,
    expected_kind: str | None = None,
    expected_embed_policy_hash: str | None = None,
    expected_normalization_id: str | None = None,
) -> None:
    _require_keys(meta, ["schema_version", "kind", "project_id", "config_hash", "created_at"])
    if meta["schema_version"] != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"meta schema_version unsupported: expected {SUPPORTED_SCHEMA_VERSION}, got {meta['schema_version']}"
        )

    kind = meta["kind"]
    if expected_kind is not None and kind != expected_kind:
        raise ValueError(f"meta kind mismatch: expected {expected_kind}, got {kind}")

    if meta["project_id"] != expected_project_id:
        raise ValueError("meta project_id mismatch")
    if meta["config_hash"] != expected_config_hash:
        raise ValueError("meta config_hash mismatch")

    if kind == "workspace_tm":
        return
    if kind == "reference_tm":
        _require_int(meta, "snapshot_id")
        return
    if kind == "examples":
        _require_keys(
            meta,
            [
                "embed_policy_hash",
                "embedding_model_id",
                "embedding_dim",
                "embedding_distance",
                "vector_encoding",
                "embedding_normalization",
                "require_finite",
                "examples_scope",
                "examples_lang",
                "source_snapshot_kind",
            ],
        )
        if expected_embed_policy_hash is not None and meta["embed_policy_hash"] != expected_embed_policy_hash:
            raise ValueError("meta embed_policy_hash mismatch")
        _require_int(meta, "embedding_dim")
        _parse_bool_text(meta["require_finite"], "require_finite")
        source_kind = meta["source_snapshot_kind"]
        if source_kind not in {"workspace_tm", "reference_tm"}:
            raise ValueError("meta source_snapshot_kind must be workspace_tm or reference_tm")
        if source_kind == "reference_tm":
            _require_int(meta, "source_snapshot_id")
        elif "source_snapshot_id" in meta and meta["source_snapshot_id"] != "":
            _require_int(meta, "source_snapshot_id")
        return
    if kind == "glossary":
        _require_keys(
            meta,
            [
                "snapshot_id",
                "source_snapshot_kind",
                "source_snapshot_id",
                "glossary_src_lang",
                "tokenizer_id",
                "normalization_id",
                "spacy_version",
                "spacy_model",
                "spacy_model_version",
            ],
        )
        _require_int(meta, "snapshot_id")
        if meta["source_snapshot_kind"] != "reference_tm":
            raise ValueError("meta source_snapshot_kind must be reference_tm")
        _require_int(meta, "source_snapshot_id")
        if expected_normalization_id is not None and meta["normalization_id"] != expected_normalization_id:
            raise ValueError("meta normalization_id mismatch")
        return

    raise ValueError(f"meta kind unsupported: {kind}")


def validate_meta_table(
    conn: sqlite3.Connection,
    *,
    expected_project_id: str,
    expected_config_hash: str,
    expected_kind: str | None = None,
    expected_embed_policy_hash: str | None = None,
    expected_normalization_id: str | None = None,
) -> dict[str, str]:
    meta = read_meta(conn)
    validate_meta(
        meta,
        expected_project_id=expected_project_id,
        expected_config_hash=expected_config_hash,
        expected_kind=expected_kind,
        expected_embed_policy_hash=expected_embed_policy_hash,
        expected_normalization_id=expected_normalization_id,
    )
    return meta
