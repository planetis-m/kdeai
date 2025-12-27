from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Mapping, Sequence
import sqlite3

from kdeai import hash as kdehash
from kdeai.config import Config
from kdeai import po_model

DEFAULT_REVIEW_STATUS_ORDER = ["reviewed", "draft", "needs_review", "unreviewed"]
DEFAULT_PREFER_HUMAN = True
DEFAULT_REVIEW_PREFIX = "KDEAI-REVIEW:"
DEFAULT_AI_PREFIX = "KDEAI-AI:"
DEFAULT_AI_FLAG = "kdeai-ai"


def _selection_settings(config: Config | None) -> tuple[list[str], bool]:
    if config is None:
        return DEFAULT_REVIEW_STATUS_ORDER, DEFAULT_PREFER_HUMAN
    selection = config.tm.selection
    return list(selection.review_status_order), bool(selection.prefer_human)


def _marker_settings_from_config(config: Config | None) -> tuple[str, str, str]:
    if config is None:
        return DEFAULT_REVIEW_PREFIX, DEFAULT_AI_PREFIX, DEFAULT_AI_FLAG
    prefixes = config.markers.comment_prefixes
    return prefixes.review, prefixes.ai, config.markers.ai_flag


def _tool_comment_lines(text: str | None, prefixes: Iterable[str]) -> list[str]:
    if not text:
        return []
    lines = [line.rstrip("\n") for line in text.replace("\r\n", "\n").split("\n")]
    selected = []
    for line in lines:
        for prefix in prefixes:
            if line.startswith(prefix):
                selected.append(line)
                break
    return selected


def _derive_review_status(
    msgstr: str,
    msgstr_plural: Mapping[str, str],
    has_plural: bool,
    flags: Sequence[str],
    tcomment: str,
    review_prefix: str,
) -> str:
    if has_plural:
        if any(str(value).strip() for value in msgstr_plural.values()):
            non_empty = True
        else:
            non_empty = False
    else:
        non_empty = msgstr.strip() != ""
    if not non_empty:
        return "unreviewed"
    if "fuzzy" in flags:
        return "needs_review"
    if _tool_comment_lines(tcomment, [review_prefix]):
        return "reviewed"
    return "draft"


def _derive_is_ai_generated(flags: Sequence[str], tcomment: str, ai_flag: str, ai_prefix: str) -> int:
    if ai_flag in flags:
        return 1
    if _tool_comment_lines(tcomment, [ai_prefix]):
        return 1
    return 0


def _selection_key(
    row: Mapping[str, object],
    review_rank: Mapping[str, int],
    prefer_human: bool,
    default_rank: int,
) -> tuple:
    review_status = str(row["review_status"])
    rank = review_rank.get(review_status, default_rank)
    is_ai_generated = int(row["is_ai_generated"])
    ai_rank = is_ai_generated if prefer_human else 0
    translation_hash = str(row["translation_hash"])
    file_path = str(row["file_path"])
    indexed_sha256 = str(row["indexed_sha256"])
    return (rank, ai_rank, translation_hash, file_path, indexed_sha256)


def _recompute_best_translations(
    cursor: sqlite3.Cursor,
    *,
    source_keys: Iterable[str],
    lang: str,
    review_status_order: list[str],
    prefer_human: bool,
) -> None:
    keys = list(source_keys)
    if not keys:
        return

    placeholders = ",".join("?" for _ in keys)
    query = (
        "SELECT t.source_key, t.lang, t.file_path, t.msgstr, t.msgstr_plural, "
        "t.review_status, t.is_ai_generated, t.translation_hash, f.indexed_sha256 "
        "FROM translations t "
        "JOIN files f ON t.file_path = f.file_path AND t.lang = f.lang "
        f"WHERE t.lang = ? AND t.source_key IN ({placeholders})"
    )
    rows = cursor.execute(query, [lang, *keys]).fetchall()

    grouped: dict[str, list[dict[str, object]]] = {key: [] for key in keys}
    for row in rows:
        grouped[str(row[0])].append(
            {
                "source_key": row[0],
                "lang": row[1],
                "file_path": row[2],
                "msgstr": row[3],
                "msgstr_plural": row[4],
                "review_status": row[5],
                "is_ai_generated": row[6],
                "translation_hash": row[7],
                "indexed_sha256": row[8],
            }
        )

    review_rank = {status: idx for idx, status in enumerate(review_status_order)}
    default_rank = len(review_status_order)

    for source_key in keys:
        candidates = grouped.get(source_key) or []
        if not candidates:
            cursor.execute(
                "DELETE FROM best_translations WHERE source_key = ? AND lang = ?",
                (source_key, lang),
            )
            continue

        best = min(
            candidates,
            key=lambda row: _selection_key(row, review_rank, prefer_human, default_rank),
        )
        cursor.execute(
            "INSERT INTO best_translations ("
            "source_key, lang, file_path, msgstr, msgstr_plural, "
            "review_status, is_ai_generated, translation_hash"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(source_key, lang) DO UPDATE SET "
            "file_path=excluded.file_path, "
            "msgstr=excluded.msgstr, "
            "msgstr_plural=excluded.msgstr_plural, "
            "review_status=excluded.review_status, "
            "is_ai_generated=excluded.is_ai_generated, "
            "translation_hash=excluded.translation_hash",
            (
                best["source_key"],
                best["lang"],
                best["file_path"],
                best["msgstr"],
                best["msgstr_plural"],
                best["review_status"],
                int(best["is_ai_generated"]),
                best["translation_hash"],
            ),
        )


def index_file_snapshot_tm(
    conn: sqlite3.Connection,
    *,
    file_path: str,
    lang: str,
    bytes: bytes,
    sha256: str,
    mtime_ns: int,
    size: int,
    config: Config | None = None,
) -> None:
    """Index one PO snapshot into the workspace TM in a single transaction."""
    units = po_model.parse_po_bytes(bytes)
    review_status_order, prefer_human = _selection_settings(config)
    review_prefix, ai_prefix, ai_flag = _marker_settings_from_config(config)

    now = datetime.now(timezone.utc).isoformat()

    cursor = conn.cursor()
    cursor.execute("BEGIN")
    try:
        cursor.execute(
            "INSERT INTO files (file_path, lang, indexed_sha256, indexed_mtime_ns, "
            "indexed_size, indexed_at) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(file_path, lang) DO UPDATE SET "
            "indexed_sha256=excluded.indexed_sha256, "
            "indexed_mtime_ns=excluded.indexed_mtime_ns, "
            "indexed_size=excluded.indexed_size, "
            "indexed_at=excluded.indexed_at",
            (file_path, lang, sha256, int(mtime_ns), int(size), now),
        )

        prior_keys = cursor.execute(
            "SELECT source_key FROM translations WHERE file_path = ? AND lang = ?",
            (file_path, lang),
        ).fetchall()
        prior_source_keys = {str(row[0]) for row in prior_keys}

        cursor.execute(
            "DELETE FROM translations WHERE file_path = ? AND lang = ?",
            (file_path, lang),
        )

        sources_payload = []
        translations_payload = []
        for unit in units:
            sources_payload.append(
                (
                    unit.source_key,
                    unit.msgctxt,
                    unit.msgid,
                    unit.msgid_plural,
                    unit.source_text,
                )
            )

            msgstr_plural_json = kdehash.canonical_msgstr_plural(unit.msgstr_plural)
            has_plural = unit.msgid_plural != ""
            review_status = _derive_review_status(
                unit.msgstr,
                unit.msgstr_plural,
                has_plural,
                unit.flags,
                unit.tcomment,
                review_prefix,
            )
            is_ai_generated = _derive_is_ai_generated(unit.flags, unit.tcomment, ai_flag, ai_prefix)
            translation_hash = kdehash.translation_hash(
                unit.source_key,
                lang,
                unit.msgstr,
                unit.msgstr_plural,
            )
            translations_payload.append(
                (
                    file_path,
                    lang,
                    unit.source_key,
                    unit.msgstr,
                    msgstr_plural_json,
                    review_status,
                    is_ai_generated,
                    translation_hash,
                )
            )

        if sources_payload:
            cursor.executemany(
                "INSERT INTO sources (source_key, msgctxt, msgid, msgid_plural, source_text) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(source_key) DO UPDATE SET "
                "msgctxt=excluded.msgctxt, "
                "msgid=excluded.msgid, "
                "msgid_plural=excluded.msgid_plural, "
                "source_text=excluded.source_text",
                sources_payload,
            )

        if translations_payload:
            cursor.executemany(
                "INSERT INTO translations (file_path, lang, source_key, msgstr, msgstr_plural, "
                "review_status, is_ai_generated, translation_hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                translations_payload,
            )

        new_source_keys = {unit.source_key for unit in units}
        affected_keys = prior_source_keys | new_source_keys

        _recompute_best_translations(
            cursor,
            source_keys=affected_keys,
            lang=lang,
            review_status_order=review_status_order,
            prefer_human=prefer_human,
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
