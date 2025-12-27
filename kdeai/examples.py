from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence
import array
import math
import sqlite3
import sys

from kdeai import db as kdedb
from kdeai.config import Config, ExamplesEligibility
from kdeai.po_utils import parse_msgstr_plural


DEFAULT_MIN_REVIEW_STATUS = "reviewed"


@dataclass(frozen=True)
class ExampleRow:
    source_key: str
    source_text: str
    lang: str
    msgstr: str
    msgstr_plural: str
    review_status: str
    is_ai_generated: int
    translation_hash: str
    file_path: str
    file_sha256: str
    embedding: bytes


@dataclass(frozen=True)
class ExampleMatch:
    source_key: str
    source_text: str
    lang: str
    msgstr: str
    msgstr_plural: dict[str, str]
    review_status: str
    is_ai_generated: int
    translation_hash: str
    file_path: str
    file_sha256: str
    distance: float


@dataclass(frozen=True)
class ExamplesDb:
    conn: sqlite3.Connection
    meta: dict[str, str]
    embedding_dim: int
    embedding_distance: str
    vector_encoding: str
    embedding_normalization: str
    require_finite: bool


EmbeddingFunc = Callable[[Sequence[str]], Sequence[Sequence[float]]]


@dataclass(frozen=True)
class PendingExample:
    source_key: str
    source_text: str
    lang: str
    msgstr: str
    msgstr_plural: str
    review_status: str
    is_ai_generated: int
    translation_hash: str
    file_path: str
    file_sha256: str


def _examples_settings(config: Config) -> tuple[list[str], str, bool]:
    examples = config.prompt.examples
    eligibility = examples.eligibility
    review_status_order = list(config.tm.selection.review_status_order)
    min_review_status = str(eligibility.min_review_status or DEFAULT_MIN_REVIEW_STATUS)
    allow_ai_generated = bool(eligibility.allow_ai_generated)
    return review_status_order, min_review_status, allow_ai_generated


def _is_non_empty(msgstr: str, msgstr_plural: Mapping[str, str], has_plural: bool) -> bool:
    if has_plural:
        return any(str(value).strip() for value in msgstr_plural.values())
    return msgstr.strip() != ""


def _review_rank(order: Sequence[str]) -> dict[str, int]:
    return {status: idx for idx, status in enumerate(order)}


def _review_eligible(
    review_status: str,
    *,
    min_review_status: str,
    review_order: Sequence[str],
) -> bool:
    order_rank = _review_rank(review_order)
    if min_review_status not in order_rank:
        raise ValueError("min_review_status not present in tm.selection.review_status_order")
    candidate_rank = order_rank.get(review_status, len(review_order))
    return candidate_rank <= order_rank[min_review_status]


def _iter_workspace_rows(conn: sqlite3.Connection, lang: str):
    query = (
        "SELECT s.source_key, s.source_text, s.msgid_plural, "
        "t.lang, t.msgstr, t.msgstr_plural, t.review_status, t.is_ai_generated, "
        "t.translation_hash, t.file_path "
        "FROM sources s "
        "JOIN best_translations t ON t.source_key = s.source_key "
        "WHERE t.lang = ? "
        "ORDER BY t.source_key"
    )
    return conn.execute(query, (lang,))


def _iter_reference_rows(conn: sqlite3.Connection, lang: str):
    query = (
        "SELECT s.source_key, s.source_text, s.msgid_plural, "
        "t.lang, t.msgstr, t.msgstr_plural, t.review_status, t.is_ai_generated, "
        "t.translation_hash, t.file_path, t.file_sha256 "
        "FROM sources s "
        "JOIN best_translations t ON t.source_key = s.source_key "
        "WHERE t.lang = ? "
        "ORDER BY t.source_key"
    )
    return conn.execute(query, (lang,))


def _normalize_embedding(values: Sequence[float], normalization: str) -> list[float]:
    if normalization == "none":
        return [float(value) for value in values]
    if normalization == "l2_normalize":
        floats = [float(value) for value in values]
        norm = math.sqrt(sum(value * value for value in floats))
        if norm == 0.0:
            return floats
        return [value / norm for value in floats]
    raise ValueError(f"unsupported embedding normalization: {normalization}")


def _pack_embedding(
    values: Sequence[float],
    *,
    embedding_dim: int,
    require_finite: bool,
) -> bytes:
    if len(values) != embedding_dim:
        raise ValueError(f"embedding dim mismatch: expected {embedding_dim}, got {len(values)}")
    floats = [float(value) for value in values]
    if require_finite and any(not math.isfinite(value) for value in floats):
        raise ValueError("embedding contains non-finite values")
    data = array.array("f", floats)
    if data.itemsize != 4:
        raise ValueError("embedding must be float32")
    if sys.byteorder != "little":
        data.byteswap()
    blob = data.tobytes()
    if len(blob) != 4 * embedding_dim:
        raise ValueError("embedding blob length mismatch")
    return blob


def _build_examples_rows(
    rows: Iterable[Sequence[object]],
    *,
    lang: str,
    config: Config,
    embedder: EmbeddingFunc,
    embedding_dim: int,
    embedding_normalization: str,
    require_finite: bool,
    include_file_sha256: bool,
) -> list[ExampleRow]:
    review_order, min_review_status, allow_ai_generated = _examples_settings(config)
    candidates: list[PendingExample] = []
    for row in rows:
        source_key = str(row[0])
        source_text = str(row[1])
        msgid_plural = str(row[2])
        msgstr = str(row[4])
        msgstr_plural = str(row[5])
        review_status = str(row[6])
        is_ai_generated = int(row[7])
        translation_hash = str(row[8])
        file_path = str(row[9])
        file_sha256 = str(row[10]) if include_file_sha256 else ""

        if not source_text.strip():
            continue
        if not _review_eligible(
            review_status,
            min_review_status=min_review_status,
            review_order=review_order,
        ):
            continue
        if not allow_ai_generated and is_ai_generated:
            continue
        plural_map = parse_msgstr_plural(msgstr_plural)
        has_plural = msgid_plural != ""
        if not _is_non_empty(msgstr, plural_map, has_plural):
            continue
        candidates.append(
            PendingExample(
                source_key=source_key,
                source_text=source_text,
                lang=lang,
                msgstr=msgstr,
                msgstr_plural=msgstr_plural,
                review_status=review_status,
                is_ai_generated=is_ai_generated,
                translation_hash=translation_hash,
                file_path=file_path,
                file_sha256=file_sha256,
            )
        )

    if not candidates:
        return []

    embeddings = embedder([row.source_text for row in candidates])
    if len(embeddings) != len(candidates):
        raise ValueError("embedder returned unexpected number of embeddings")

    payload: list[ExampleRow] = []
    for row, embedding in zip(candidates, embeddings):
        normalized = _normalize_embedding(embedding, embedding_normalization)
        blob = _pack_embedding(
            normalized,
            embedding_dim=embedding_dim,
            require_finite=require_finite,
        )
        payload.append(
            ExampleRow(
                source_key=row.source_key,
                source_text=row.source_text,
                lang=row.lang,
                msgstr=row.msgstr,
                msgstr_plural=row.msgstr_plural,
                review_status=row.review_status,
                is_ai_generated=int(row.is_ai_generated),
                translation_hash=row.translation_hash,
                file_path=row.file_path,
                file_sha256=row.file_sha256,
                embedding=blob,
            )
        )
    return payload


def _create_vector_index(
    conn: sqlite3.Connection,
    *,
    embedding_dim: int,
    embedding_distance: str,
) -> None:
    distance = embedding_distance.upper()
    conn.execute(
        "SELECT vector_init('examples', 'embedding', ?)",
        (f"type=FLOAT32,dimension={embedding_dim},distance={distance}",),
    )
    conn.execute("SELECT vector_quantize('examples', 'embedding')")


def _build_examples_db(
    rows: Iterable[Sequence[object]],
    *,
    output_path: Path,
    scope: str,
    source_snapshot_kind: str,
    source_snapshot_id: str | None,
    lang: str,
    config: Config,
    project_id: str,
    config_hash: str,
    embed_policy_hash: str,
    embedder: EmbeddingFunc,
    sqlite_vector_path: str | None,
) -> Path:
    policy = config.prompt.examples.embedding_policy
    embedding_dim = int(policy.dim)
    embedding_distance = str(policy.distance)
    vector_encoding = str(policy.encoding)
    embedding_normalization = str(policy.normalization)
    require_finite = bool(policy.require_finite)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)

    conn = kdedb.connect_writable(output_path)
    conn.executescript(kdedb.EXAMPLES_SCHEMA)

    include_file_sha256 = source_snapshot_kind == "reference_tm"
    payload = _build_examples_rows(
        rows,
        lang=lang,
        config=config,
        embedder=embedder,
        embedding_dim=embedding_dim,
        embedding_normalization=embedding_normalization,
        require_finite=require_finite,
        include_file_sha256=include_file_sha256,
    )

    if payload:
        conn.executemany(
            "INSERT INTO examples ("
            "source_key, source_text, lang, msgstr, msgstr_plural, review_status, "
            "is_ai_generated, translation_hash, file_path, file_sha256, embedding"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row.source_key,
                    row.source_text,
                    row.lang,
                    row.msgstr,
                    row.msgstr_plural,
                    row.review_status,
                    int(row.is_ai_generated),
                    row.translation_hash,
                    row.file_path,
                    row.file_sha256,
                    row.embedding,
                )
                for row in payload
            ],
        )

    if not sqlite_vector_path:
        conn.close()
        raise ValueError("sqlite-vector extension path is required to build examples")
    try:
        kdedb.enable_sqlite_vector(conn, extension_path=sqlite_vector_path)
    except Exception as exc:
        conn.close()
        raise RuntimeError(
            f"failed to load sqlite-vector extension at {sqlite_vector_path}: {exc}"
        ) from exc

    _create_vector_index(
        conn,
        embedding_dim=embedding_dim,
        embedding_distance=embedding_distance,
    )

    meta_payload = {
        "schema_version": "1",
        "kind": "examples",
        "project_id": project_id,
        "config_hash": config_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embed_policy_hash": embed_policy_hash,
        "embedding_model_id": str(policy.model_id),
        "embedding_dim": str(embedding_dim),
        "embedding_distance": embedding_distance,
        "vector_encoding": vector_encoding,
        "embedding_normalization": embedding_normalization,
        "require_finite": "1" if require_finite else "0",
        "examples_scope": scope,
        "examples_lang": lang,
        "source_snapshot_kind": source_snapshot_kind,
        "source_snapshot_id": source_snapshot_id or "",
    }
    conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta_payload.items())
    conn.commit()
    kdedb.validate_meta_table(
        conn,
        expected_project_id=project_id,
        expected_config_hash=config_hash,
        expected_kind="examples",
        expected_embed_policy_hash=embed_policy_hash,
    )
    conn.close()
    return output_path


def build_examples_db_from_workspace(
    workspace_conn: sqlite3.Connection,
    *,
    output_path: Path,
    lang: str,
    config: Config,
    project_id: str,
    config_hash: str,
    embed_policy_hash: str,
    embedder: EmbeddingFunc,
    sqlite_vector_path: str | None,
) -> Path:
    meta = kdedb.read_meta(workspace_conn)
    kdedb.validate_meta(
        meta,
        expected_project_id=project_id,
        expected_config_hash=config_hash,
        expected_kind="workspace_tm",
    )
    rows = _iter_workspace_rows(workspace_conn, lang)
    return _build_examples_db(
        rows,
        output_path=output_path,
        scope="workspace",
        source_snapshot_kind="workspace_tm",
        source_snapshot_id=None,
        lang=lang,
        config=config,
        project_id=project_id,
        config_hash=config_hash,
        embed_policy_hash=embed_policy_hash,
        embedder=embedder,
        sqlite_vector_path=sqlite_vector_path,
    )


def build_examples_db_from_reference(
    reference_conn: sqlite3.Connection,
    *,
    output_path: Path,
    lang: str,
    config: Config,
    project_id: str,
    config_hash: str,
    embed_policy_hash: str,
    embedder: EmbeddingFunc,
    sqlite_vector_path: str | None,
) -> Path:
    meta = kdedb.read_meta(reference_conn)
    kdedb.validate_meta(
        meta,
        expected_project_id=project_id,
        expected_config_hash=config_hash,
        expected_kind="reference_tm",
    )
    snapshot_id = str(meta.get("snapshot_id", ""))
    if snapshot_id == "":
        raise ValueError("reference snapshot_id missing")
    rows = _iter_reference_rows(reference_conn, lang)
    return _build_examples_db(
        rows,
        output_path=output_path,
        scope="reference",
        source_snapshot_kind="reference_tm",
        source_snapshot_id=snapshot_id,
        lang=lang,
        config=config,
        project_id=project_id,
        config_hash=config_hash,
        embed_policy_hash=embed_policy_hash,
        embedder=embedder,
        sqlite_vector_path=sqlite_vector_path,
    )


def open_examples_db(
    path: Path,
    *,
    project_id: str,
    config_hash: str,
    embed_policy_hash: str,
    sqlite_vector_path: str | None,
) -> ExamplesDb:
    conn = kdedb.connect_readonly(path)
    try:
        meta = kdedb.validate_meta_table(
            conn,
            expected_project_id=project_id,
            expected_config_hash=config_hash,
            expected_kind="examples",
            expected_embed_policy_hash=embed_policy_hash,
        )
        _validate_examples_meta(meta)
        if sqlite_vector_path is None:
            raise RuntimeError("sqlite-vector unavailable")
        try:
            kdedb.enable_sqlite_vector(conn, extension_path=sqlite_vector_path)
        except Exception as exc:
            raise RuntimeError(
                f"failed to load sqlite-vector extension at {sqlite_vector_path}: {exc}"
            ) from exc
    except Exception:
        conn.close()
        raise
    return ExamplesDb(
        conn=conn,
        meta=meta,
        embedding_dim=int(meta["embedding_dim"]),
        embedding_distance=str(meta["embedding_distance"]),
        vector_encoding=str(meta["vector_encoding"]),
        embedding_normalization=str(meta["embedding_normalization"]),
        require_finite=str(meta["require_finite"]).strip().lower() in {"1", "true"},
    )


def _validate_examples_meta(meta: Mapping[str, str]) -> None:
    required_keys = [
        "schema_version",
        "kind",
        "project_id",
        "config_hash",
        "created_at",
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
        "source_snapshot_id",
    ]
    missing = [key for key in required_keys if key not in meta]
    if missing:
        raise ValueError(f"invalid examples DB meta: missing {', '.join(missing)}")
    blank = [
        key
        for key in required_keys
        if meta.get(key, "") == "" and key != "source_snapshot_id"
    ]
    if blank:
        raise ValueError(f"invalid examples DB meta: empty {', '.join(blank)}")
    if meta.get("vector_encoding") != "float32_le":
        raise ValueError("invalid examples DB meta: vector_encoding must be float32_le")
    try:
        embedding_dim = int(meta.get("embedding_dim", ""))
    except ValueError as exc:
        raise ValueError("invalid examples DB meta: embedding_dim must be an int") from exc
    if embedding_dim <= 0:
        raise ValueError("invalid examples DB meta: embedding_dim must be > 0")
    source_kind = meta.get("source_snapshot_kind", "")
    if source_kind == "reference_tm" and meta.get("source_snapshot_id", "") == "":
        raise ValueError(
            "invalid examples DB meta: source_snapshot_id required for reference_tm"
        )


def _vector_query_sql(
    *,
    with_lang_filter: bool,
    review_statuses: Sequence[str] | None,
    allow_ai_generated: bool | None,
) -> str:
    where_parts: list[str] = []
    if with_lang_filter:
        where_parts.append("e.lang = ?")
    if review_statuses:
        placeholders = ",".join(["?"] * len(review_statuses))
        where_parts.append(f"e.review_status IN ({placeholders})")
    if allow_ai_generated is False:
        where_parts.append("e.is_ai_generated = 0")
    where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
    return (
        "SELECT e.source_key, e.source_text, e.lang, e.msgstr, e.msgstr_plural, "
        "e.review_status, e.is_ai_generated, e.translation_hash, e.file_path, "
        "e.file_sha256, v.distance "
        "FROM examples e "
        "JOIN vector_quantize_scan('examples', 'embedding', ?, ?) v "
        "ON e.id = v.rowid "
        f"{where_clause} "
        "ORDER BY v.distance, e.source_key, e.translation_hash, e.file_path, e.file_sha256"
    )


def _eligible_review_statuses(
    eligibility: ExamplesEligibility,
    review_status_order: Sequence[str],
) -> list[str]:
    min_review_status = str(eligibility.min_review_status or DEFAULT_MIN_REVIEW_STATUS)
    order = list(review_status_order)
    if min_review_status not in order:
        raise ValueError("min_review_status not present in tm.selection.review_status_order")
    cutoff = order.index(min_review_status)
    return order[: cutoff + 1]


def query_examples(
    db: ExamplesDb,
    *,
    query_embedding: Sequence[float] | bytes,
    top_n: int,
    lang: str | None = None,
    eligibility: ExamplesEligibility | None = None,
    review_status_order: Sequence[str] | None = None,
) -> list[ExampleMatch]:
    if isinstance(query_embedding, (bytes, bytearray)):
        blob = bytes(query_embedding)
        if len(blob) != 4 * db.embedding_dim:
            raise ValueError("query embedding blob length mismatch")
    else:
        normalized = _normalize_embedding(
            query_embedding, db.embedding_normalization
        )
        blob = _pack_embedding(
            normalized,
            embedding_dim=db.embedding_dim,
            require_finite=db.require_finite,
        )

    review_statuses: list[str] | None = None
    allow_ai_generated: bool | None = None
    if eligibility is not None:
        if review_status_order is None:
            raise ValueError("review_status_order required when eligibility is provided")
        review_statuses = _eligible_review_statuses(eligibility, review_status_order)
        allow_ai_generated = bool(eligibility.allow_ai_generated)

    sql = _vector_query_sql(
        with_lang_filter=lang is not None,
        review_statuses=review_statuses,
        allow_ai_generated=allow_ai_generated,
    )
    has_filters = lang is not None or bool(review_statuses) or allow_ai_generated is False
    scan_limit = int(top_n)
    if has_filters:
        scan_limit = min(max(scan_limit * 10, scan_limit), 10000)

    params: list[object] = [blob, int(scan_limit)]
    if lang is not None:
        params.append(lang)
    if review_statuses:
        params.extend(review_statuses)

    try:
        rows = db.conn.execute(sql, params).fetchall()[:top_n]
    except Exception as exc:
        raise RuntimeError("sqlite-vector unavailable for examples") from exc
    matches: list[ExampleMatch] = []
    for row in rows:
        matches.append(
            ExampleMatch(
                source_key=str(row[0]),
                source_text=str(row[1]),
                lang=str(row[2]),
                msgstr=str(row[3]),
                msgstr_plural=parse_msgstr_plural(row[4]),
                review_status=str(row[5]),
                is_ai_generated=int(row[6]),
                translation_hash=str(row[7]),
                file_path=str(row[8]),
                file_sha256=str(row[9]),
                distance=float(row[10]),
            )
        )
    return matches
