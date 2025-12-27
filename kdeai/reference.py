from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence
import json
import os
import sqlite3
import tempfile

import polib

from kdeai import apply as kdeapply
from kdeai.config import Config
from kdeai import db as kdedb
from kdeai import hash as kdehash
from kdeai import locks
from kdeai import po_model
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


def _load_po_from_bytes(data: bytes) -> polib.POFile:
    with tempfile.NamedTemporaryFile(suffix=".po", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return polib.pofile(tmp_path)
    finally:
        os.unlink(tmp_path)


def _iter_translation_entries(po_file: polib.POFile) -> Iterable[polib.POEntry]:
    for entry in po_file:
        if entry.obsolete:
            continue
        if entry.msgid == "":
            continue
        yield entry


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


def _derive_review_status(entry: polib.POEntry, review_prefix: str) -> str:
    has_plural = bool(entry.msgid_plural)
    if has_plural:
        non_empty = any(str(value).strip() for value in entry.msgstr_plural.values())
    else:
        non_empty = (entry.msgstr or "").strip() != ""
    if not non_empty:
        return ReviewStatus.UNREVIEWED
    if "fuzzy" in entry.flags:
        return ReviewStatus.NEEDS_REVIEW
    if _tool_comment_lines(entry.tcomment, [review_prefix]):
        return ReviewStatus.REVIEWED
    return ReviewStatus.DRAFT


def _derive_is_ai_generated(entry: polib.POEntry, ai_flag: str, ai_prefix: str) -> int:
    if ai_flag in entry.flags:
        return 1
    if _tool_comment_lines(entry.tcomment, [ai_prefix]):
        return 1
    return 0


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


def _selection_key(
    row: Sequence[object],
    review_rank: Mapping[str, int],
    prefer_human: bool,
    default_rank: int,
) -> tuple:
    review_status = str(row[6])
    rank = review_rank.get(review_status, default_rank)
    is_ai_generated = int(row[7])
    ai_rank = is_ai_generated if prefer_human else 0
    translation_hash = str(row[8])
    file_path = str(row[2])
    file_sha256 = str(row[3])
    return (rank, ai_rank, translation_hash, file_path, file_sha256)


def _normalize_relpath(project_root: Path, path: Path) -> str:
    resolved = path.resolve()
    relpath = resolved.relative_to(project_root.resolve())
    return relpath.as_posix()


def _iter_po_paths(project_root: Path, raw_paths: Optional[list[Path]]) -> list[Path]:
    roots = raw_paths if raw_paths else [project_root]
    seen: set[Path] = set()
    results: list[Path] = []

    for raw in roots:
        full = raw if raw.is_absolute() else project_root / raw
        if not full.exists():
            continue
        if full.is_file():
            candidates = [full]
        else:
            candidates = list(full.rglob("*.po"))
        for candidate in candidates:
            if candidate.suffix.lower() != ".po":
                continue
            if any(part in {".kdeai", ".git"} for part in candidate.parts):
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            results.append(resolved)
    return results


def _lang_from_po(po_file: polib.POFile, config: Config) -> str:
    language = po_file.metadata.get("Language") if po_file else None
    if language:
        return str(language).strip()
    targets = config.languages.targets
    if len(targets) == 1:
        return str(targets[0])
    return ""


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

    for path in _iter_po_paths(project_root, paths):
        relpath = _normalize_relpath(project_root, path)
        relpath_key = relpath.casefold() if path_casefold else relpath
        lock_path = locks.per_file_lock_path(
            project_root,
            locks.lock_id(project_id, relpath_key),
        )
        locked = snapshot.locked_read_file(path, lock_path, relpath=relpath)
        po_file = _load_po_from_bytes(locked.bytes)
        lang = _lang_from_po(po_file, config)
        if not lang:
            raise ValueError(f"unable to infer language for {relpath}")

        sources_payload: list[tuple[str, str, str, str, str]] = []
        translations_payload: list[tuple[str, str, str, str, str, str, str, int, str]] = []
        for entry in _iter_translation_entries(po_file):
            msgctxt = entry.msgctxt or ""
            msgid = entry.msgid
            msgid_plural = entry.msgid_plural or ""
            msgstr = entry.msgstr or ""
            msgstr_plural = {str(k): str(v) for k, v in entry.msgstr_plural.items()}
            source_key = po_model.source_key_for(msgctxt, msgid, msgid_plural)
            source_text = po_model.source_text_v1(msgctxt, msgid, msgid_plural)
            msgstr_plural_json = kdehash.canonical_msgstr_plural(msgstr_plural)
            review_status = _derive_review_status(entry, review_prefix)
            is_ai_generated = _derive_is_ai_generated(entry, ai_flag, ai_prefix)
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

    rows = conn.execute(
        "SELECT source_key, lang, file_path, file_sha256, msgstr, msgstr_plural, "
        "review_status, is_ai_generated, translation_hash FROM translations"
    ).fetchall()

    grouped: dict[tuple[str, str], list[Sequence[object]]] = {}
    for row in rows:
        grouped.setdefault((str(row[0]), str(row[1])), []).append(row)

    conn.execute("DELETE FROM best_translations")
    review_rank = {status: idx for idx, status in enumerate(review_status_order)}
    default_rank = len(review_status_order)
    best_payload: list[tuple[str, str, str, str, str, str, str, int, str]] = []
    for key in sorted(grouped.keys()):
        candidates = grouped[key]
        best = min(
            candidates,
            key=lambda row: _selection_key(row, review_rank, prefer_human, default_rank),
        )
        best_payload.append(
            (
                str(best[0]),
                str(best[1]),
                str(best[2]),
                str(best[3]),
                str(best[4]),
                str(best[5]),
                str(best[6]),
                int(best[7]),
                str(best[8]),
            )
        )

    if best_payload:
        conn.executemany(
            "INSERT INTO best_translations ("
            "source_key, lang, file_path, file_sha256, msgstr, msgstr_plural, "
            "review_status, is_ai_generated, translation_hash"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            best_payload,
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
