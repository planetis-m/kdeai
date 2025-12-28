from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import json
import sqlite3

from kdeai import apply as kdeapply
from kdeai.config import Config
from kdeai import db as kdedb
from kdeai import hash as kdehash
from kdeai import locks
from kdeai import po_model
from kdeai import po_utils
from kdeai import snapshot
from kdeai.constants import DbKind, REFERENCE_TM_SCHEMA_VERSION, ReviewStatus


DEFAULT_REVIEW_STATUS_ORDER = [
    ReviewStatus.REVIEWED,
    ReviewStatus.DRAFT,
    ReviewStatus.NEEDS_REVIEW,
    ReviewStatus.UNREVIEWED,
]


@dataclass(frozen=True)
class ReferenceSnapshot:
    snapshot_id: int
    db_path: Path
    created_at: str


def _marker_settings_from_config(config: Config) -> tuple[str, str, str]:
    markers = config.markers
    return (
        markers.ai_flag,
        markers.comment_prefixes.ai,
        markers.comment_prefixes.review,
    )


def _selection_settings(config: Config) -> tuple[list[str], bool]:
    selection = config.tm.selection
    return list(selection.review_status_order), bool(selection.prefer_human)


def _normalize_relpath(project_root: Path, path: Path) -> str:
    resolved = path.resolve()
    relpath = resolved.relative_to(project_root.resolve())
    return relpath.as_posix()


def _next_snapshot_id(reference_dir: Path) -> int:
    pointer_path = reference_dir / "reference.current.json"
    if not pointer_path.exists():
        return 1
    try:
        payload = json.loads(pointer_path.read_text(encoding="utf-8"))
        snapshot_id = int(payload.get("snapshot_id", 0))
    except (OSError, json.JSONDecodeError, ValueError):
        snapshot_id = 0
    return snapshot_id + 1


def build_reference_snapshot(
    project_root: Path,
    *,
    project_id: str,
    path_casefold: bool,
    config: Config,
    config_hash: str,
    paths: Optional[list[Path]] = None,
    label: str | None = None,
) -> ReferenceSnapshot:
    reference_dir = project_root / ".kdeai" / "cache" / "reference"
    snapshot_id = _next_snapshot_id(reference_dir)
    created_at = datetime.now(timezone.utc).isoformat()
    output_path = reference_dir / f"reference.{snapshot_id}.sqlite"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    ai_flag, ai_prefix, review_prefix = _marker_settings_from_config(config)
    review_status_order, prefer_human = _selection_settings(config)

    conn = sqlite3.connect(str(output_path))
    conn.executescript(kdedb.REFERENCE_TM_SCHEMA)

    sources_sql = (
        "INSERT INTO sources (source_key, msgctxt, msgid, msgid_plural, source_text) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(source_key) DO UPDATE SET "
        "msgctxt=excluded.msgctxt, "
        "msgid=excluded.msgid, "
        "msgid_plural=excluded.msgid_plural, "
        "source_text=excluded.source_text"
    )
    translations_sql = (
        "INSERT INTO translations ("
        "source_key, lang, file_path, file_sha256, msgstr, msgstr_plural, "
        "review_status, is_ai_generated, translation_hash"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(source_key, lang, file_path) DO UPDATE SET "
        "file_sha256=excluded.file_sha256, "
        "msgstr=excluded.msgstr, "
        "msgstr_plural=excluded.msgstr_plural, "
        "review_status=excluded.review_status, "
        "is_ai_generated=excluded.is_ai_generated, "
        "translation_hash=excluded.translation_hash"
    )

    for path in po_utils.iter_po_paths(project_root, paths):
        relpath = _normalize_relpath(project_root, path)
        relpath_key = relpath.casefold() if path_casefold else relpath
        lock_path = locks.per_file_lock_path(
            project_root,
            locks.lock_id(project_id, relpath_key),
        )
        locked = snapshot.locked_read_file(path, lock_path, relpath=relpath)
        po_file = po_model.load_po_from_bytes(locked.bytes)
        lang = po_utils.get_po_language(po_file, config)
        if not lang:
            raise ValueError(f"unable to infer language for {relpath}")

        sources_payload: list[tuple[str, str, str, str, str]] = []
        translations_payload: list[tuple[str, str, str, str, str, str, str, int, str]] = []
        for entry in po_model.iter_active_entries(po_file):
            msgctxt = entry.msgctxt or ""
            msgid = entry.msgid
            msgid_plural = entry.msgid_plural or ""
            msgstr = entry.msgstr or ""
            msgstr_plural = {str(k): str(v) for k, v in entry.msgstr_plural.items()}
            source_key = po_model.source_key_for(msgctxt, msgid, msgid_plural)
            source_text = po_model.source_text_v1(msgctxt, msgid, msgid_plural)
            msgstr_plural_json = kdehash.canonical_msgstr_plural(msgstr_plural)
            review_status = po_utils.derive_review_status_entry(entry, review_prefix)
            is_ai_generated = po_utils.derive_is_ai_generated_entry(entry, ai_flag, ai_prefix)
            translation_hash = kdehash.translation_hash(
                source_key, lang, msgstr, msgstr_plural
            )
            sources_payload.append(
                (source_key, msgctxt, msgid, msgid_plural, source_text)
            )
            translations_payload.append(
                (
                    source_key,
                    lang,
                    relpath,
                    locked.sha256,
                    msgstr,
                    msgstr_plural_json,
                    review_status,
                    int(is_ai_generated),
                    translation_hash,
                )
            )

        if sources_payload:
            conn.executemany(sources_sql, sources_payload)
        if translations_payload:
            conn.executemany(translations_sql, translations_payload)

    conn.execute("DELETE FROM best_translations")
    default_rank = len(review_status_order)
    case_clauses = []
    for idx, status in enumerate(review_status_order):
        escaped_status = str(status).replace("'", "''")
        case_clauses.append(f"WHEN '{escaped_status}' THEN {idx}")
    review_rank_case = (
        "CASE review_status "
        + " ".join(case_clauses)
        + f" ELSE {default_rank} END"
    )
    ai_order = "is_ai_generated" if prefer_human else "0"
    conn.execute(
        "INSERT INTO best_translations ("
        "source_key, lang, file_path, file_sha256, msgstr, msgstr_plural, "
        "review_status, is_ai_generated, translation_hash"
        ") "
        "SELECT source_key, lang, file_path, file_sha256, msgstr, msgstr_plural, "
        "review_status, is_ai_generated, translation_hash "
        "FROM ("
        "SELECT source_key, lang, file_path, file_sha256, msgstr, msgstr_plural, "
        "review_status, is_ai_generated, translation_hash, "
        "ROW_NUMBER() OVER (PARTITION BY source_key, lang ORDER BY "
        f"{review_rank_case}, {ai_order}, translation_hash, file_path, file_sha256"
        ") AS rn "
        "FROM translations"
        ") WHERE rn = 1"
    )

    meta_payload = {
        "schema_version": REFERENCE_TM_SCHEMA_VERSION,
        "kind": DbKind.REFERENCE_TM,
        "project_id": project_id,
        "config_hash": config_hash,
        "created_at": created_at,
        "snapshot_id": str(snapshot_id),
    }
    if label:
        meta_payload["source_label"] = str(label)
    conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta_payload.items())
    conn.commit()
    conn.close()

    return ReferenceSnapshot(
        snapshot_id=snapshot_id,
        db_path=output_path,
        created_at=created_at,
    )
