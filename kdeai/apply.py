from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping
import hashlib
import json
import os
import sqlite3
import tempfile

import polib

from kdeai import hash as kdehash
from kdeai import locks
from kdeai import po_model
from kdeai import po_utils
from kdeai.config import Config
from kdeai import snapshot
from kdeai import validate
from kdeai import workspace_tm
from kdeai.tm_types import SessionTm

DEFAULT_COMMENT_PREFIXES = po_utils.DEFAULT_COMMENT_PREFIXES
DEFAULT_MARKER_FLAGS = po_utils.DEFAULT_MARKER_FLAGS
DEFAULT_AI_FLAG = "kdeai-ai"
ALLOWED_OVERWRITE_POLICIES = {
    "conservative",
    "allow-nonempty",
    "allow-reviewed",
    "all",
}


@dataclass(frozen=True)
class ApplyResult:
    files_written: list[str]
    files_skipped: list[str]
    entries_applied: int
    errors: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class ApplyFileResult:
    file_path: str
    wrote: bool
    skipped: bool
    entries_applied: int
    errors: list[str]
    warnings: list[str]
    applied_entries: list[polib.POEntry]


def _load_po_from_bytes(data: bytes) -> polib.POFile:
    return po_utils.load_po_from_bytes(data)


def _serialize_po(po_file: polib.POFile) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".po", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        po_file.save(tmp_path)
        return Path(tmp_path).read_bytes()
    finally:
        if Path(tmp_path).exists():
            os.unlink(tmp_path)


def _entry_key(entry: polib.POEntry) -> tuple[str, str, str]:
    return (
        entry.msgctxt or "",
        entry.msgid,
        entry.msgid_plural or "",
    )


def _tool_comment_lines(text: str | None, prefixes: Iterable[str]) -> list[str]:
    return po_utils.tool_comment_lines(text, prefixes)


def _entry_state_hash(
    entry: polib.POEntry,
    lang: str,
    marker_flags: Iterable[str],
    comment_prefixes: Iterable[str],
) -> str:
    source_key = po_model.source_key_for(entry.msgctxt, entry.msgid, entry.msgid_plural)
    msgstr = entry.msgstr or ""
    msgstr_plural = {str(k): str(v) for k, v in entry.msgstr_plural.items()}
    marker_flags_present = [flag for flag in marker_flags if flag in entry.flags]
    tool_lines = _tool_comment_lines(entry.tcomment, comment_prefixes)
    return kdehash.state_hash(
        source_key,
        lang,
        msgstr,
        msgstr_plural,
        marker_flags_present,
        tool_lines,
    )


def _marker_settings_from_config(config: Config) -> tuple[list[str], list[str], str, str, str]:
    return po_utils.marker_settings_from_config(config)


def entry_state_hash(
    entry: polib.POEntry,
    *,
    lang: str,
    marker_flags: Iterable[str] | None = None,
    comment_prefixes: Iterable[str] | None = None,
) -> str:
    return _entry_state_hash(
        entry,
        lang,
        marker_flags or DEFAULT_MARKER_FLAGS,
        comment_prefixes or DEFAULT_COMMENT_PREFIXES,
    )


def _apply_flags(entry: polib.POEntry, add_flags: Iterable[str], remove_flags: Iterable[str]) -> None:
    remove_set = {str(flag) for flag in remove_flags}
    existing = [flag for flag in entry.flags if flag not in remove_set]
    for flag in add_flags:
        if flag not in existing:
            existing.append(str(flag))
    entry.flags = existing


def _apply_comments(
    entry: polib.POEntry,
    remove_prefixes: Iterable[str],
    ensure_lines: Iterable[str],
    append: str,
) -> None:
    original = entry.tcomment or ""
    normalized = original.replace("\r\n", "\n")
    lines = normalized.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]

    filtered: list[str] = []
    ensure_set = {str(line) for line in ensure_lines}
    for line in lines:
        if any(line.startswith(prefix) for prefix in remove_prefixes):
            continue
        if line in ensure_set:
            continue
        filtered.append(line)

    seen_ensure: set[str] = set()
    for line in ensure_lines:
        line_text = str(line)
        if line_text in seen_ensure:
            continue
        filtered.append(line_text)
        seen_ensure.add(line_text)

    if append:
        append_text = str(append)
        if not append_text.endswith("\n"):
            append_text += "\n"
        append_lines = append_text.split("\n")
        if append_lines and append_lines[-1] == "":
            append_lines = append_lines[:-1]
        filtered.extend(append_lines)

    entry.tcomment = "\n".join(filtered)


def _set_translation(
    entry: polib.POEntry,
    *,
    msgstr: str,
    msgstr_plural: Mapping[str, str],
) -> None:
    entry.msgstr = str(msgstr)
    entry.msgstr_plural = {str(k): str(v) for k, v in msgstr_plural.items()}


def _is_reviewed(entry: polib.POEntry, review_prefix: str) -> bool:
    return po_utils.is_reviewed(entry, review_prefix)


def _has_non_empty_translation(entry: polib.POEntry) -> bool:
    return po_utils.has_non_empty_translation(entry)


def _can_overwrite(current_non_empty: bool, reviewed: bool, overwrite: str) -> bool:
    return po_utils.can_overwrite(current_non_empty, reviewed, overwrite)


def _derive_review_status(entry: polib.POEntry, review_prefix: str) -> str:
    non_empty = po_utils.has_non_empty_translation(entry)
    if not non_empty:
        return "unreviewed"
    if "fuzzy" in entry.flags:
        return "needs_review"
    if po_utils.is_reviewed(entry, review_prefix):
        return "reviewed"
    return "draft"


def _derive_is_ai_generated(entry: polib.POEntry, ai_flag: str, ai_prefix: str) -> int:
    if ai_flag in entry.flags:
        return 1
    if po_utils.tool_comment_lines(entry.tcomment, [ai_prefix]):
        return 1
    return 0


def _update_session_tm(
    session_tm: SessionTm,
    *,
    entries: Iterable[polib.POEntry],
    lang: str,
    review_prefix: str,
    ai_prefix: str,
    ai_flag: str,
) -> None:
    for entry in entries:
        source_key = po_model.source_key_for(entry.msgctxt, entry.msgid, entry.msgid_plural)
        msgstr_plural = {str(k): str(v) for k, v in entry.msgstr_plural.items()}
        review_status = _derive_review_status(entry, review_prefix)
        is_ai_generated = _derive_is_ai_generated(entry, ai_flag, ai_prefix)
        translation_hash = kdehash.translation_hash(source_key, lang, entry.msgstr or "", msgstr_plural)
        session_tm[(source_key, lang)] = {
            "msgstr": entry.msgstr or "",
            "msgstr_plural": msgstr_plural,
            "review_status": review_status,
            "is_ai_generated": is_ai_generated,
            "translation_hash": translation_hash,
        }


def apply_plan_to_file(
    file_path: str,
    plan_items: list,
    mode: str,
    overwrite_policy: str,
    *,
    full_path: Path,
    lock_path: Path,
    base_sha256: str,
    lang: str,
    marker_flags: Iterable[str],
    comment_prefixes: Iterable[str],
    placeholder_patterns: Iterable[str],
    review_prefix: str = "KDEAI-REVIEW:",
) -> ApplyFileResult:
    if overwrite_policy not in ALLOWED_OVERWRITE_POLICIES:
        return ApplyFileResult(
            file_path,
            False,
            True,
            0,
            [f"unsupported overwrite mode: {overwrite_policy}"],
            [],
            [],
        )
    phase_a = snapshot.locked_read_file(full_path, lock_path)
    if mode == "strict" and (not base_sha256 or phase_a.sha256 != base_sha256):
        return ApplyFileResult(file_path, False, True, 0, [], [], [])

    po_file = _load_po_from_bytes(phase_a.bytes)
    entry_map = {_entry_key(entry): entry for entry in po_file if entry.msgid != "" and not entry.obsolete}

    to_apply: list[tuple[Mapping[str, object], polib.POEntry]] = []
    mismatch = False
    for entry_item in plan_items:
        if not isinstance(entry_item, Mapping):
            mismatch = True
            break
        key = (
            str(entry_item.get("msgctxt", "")),
            str(entry_item.get("msgid", "")),
            str(entry_item.get("msgid_plural", "")),
        )
        entry = entry_map.get(key)
        if entry is None:
            if mode == "strict":
                mismatch = True
                break
            continue
        current_hash = _entry_state_hash(entry, lang, marker_flags, comment_prefixes)
        base_state_hash = str(entry_item.get("base_state_hash", ""))
        if current_hash != base_state_hash:
            if mode == "strict":
                mismatch = True
                break
            continue
        to_apply.append((entry_item, entry))

    if mismatch:
        return ApplyFileResult(file_path, False, True, 0, [], [], [])

    file_errors: list[str] = []
    applied_entries: list[polib.POEntry] = []
    applied_in_file = 0
    for entry_item, entry in to_apply:
        current_non_empty = _has_non_empty_translation(entry)
        reviewed = _is_reviewed(entry, review_prefix)
        if not _can_overwrite(current_non_empty, reviewed, overwrite_policy):
            continue

        translation = entry_item.get("translation") if isinstance(entry_item, Mapping) else None
        if isinstance(translation, Mapping):
            msgstr = str(translation.get("msgstr", ""))
            msgstr_plural = translation.get("msgstr_plural", {})
            if not isinstance(msgstr_plural, Mapping):
                msgstr_plural = {}
            _set_translation(entry, msgstr=msgstr, msgstr_plural=msgstr_plural)

        flags = entry_item.get("flags") if isinstance(entry_item, Mapping) else None
        if isinstance(flags, Mapping):
            add_flags = flags.get("add", [])
            remove_flags = flags.get("remove", [])
            _apply_flags(entry, add_flags, remove_flags)

        comments = entry_item.get("comments") if isinstance(entry_item, Mapping) else None
        if isinstance(comments, Mapping):
            remove_prefixes = comments.get("remove_prefixes", [])
            ensure_lines = comments.get("ensure_lines", [])
            append = str(comments.get("append", ""))
            _apply_comments(entry, remove_prefixes, ensure_lines, append)

        plural_forms = po_file.metadata.get("Plural-Forms")
        validation_errors = validate.validate_entry(
            msgid=entry.msgid,
            msgid_plural=entry.msgid_plural or "",
            msgstr=entry.msgstr or "",
            msgstr_plural={str(k): str(v) for k, v in entry.msgstr_plural.items()},
            plural_forms=plural_forms,
            placeholder_patterns=placeholder_patterns,
        )
        if validation_errors:
            file_errors.extend(validation_errors)
            break

        applied_in_file += 1
        applied_entries.append(entry)

    if file_errors:
        return ApplyFileResult(file_path, False, True, 0, file_errors, [], [])

    if applied_in_file == 0:
        return ApplyFileResult(file_path, False, True, 0, [], [], [])

    new_bytes = _serialize_po(po_file)

    with locks.acquire_file_lock(lock_path):
        current_bytes = full_path.read_bytes()
        current_sha = hashlib.sha256(current_bytes).hexdigest()
        if current_sha != phase_a.sha256:
            return ApplyFileResult(file_path, False, True, 0, [], [], [])
        if mode == "strict" and current_sha != base_sha256:
            return ApplyFileResult(file_path, False, True, 0, [], [], [])
        tmp_handle = tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(full_path.parent),
            delete=False,
        )
        try:
            tmp_handle.write(new_bytes)
            tmp_handle.flush()
            os.fsync(tmp_handle.fileno())
            tmp_name = tmp_handle.name
        finally:
            tmp_handle.close()
        os.replace(tmp_name, full_path)

    return ApplyFileResult(file_path, True, False, applied_in_file, [], [], applied_entries)


def apply_plan(
    plan: Mapping[str, object],
    *,
    project_root: Path,
    config: Config,
    apply_mode: str | None = None,
    overwrite: str | None = None,
    post_index: bool | None = None,
    workspace_conn=None,
    session_tm: SessionTm | None = None,
) -> ApplyResult:
    project_meta = {}
    project_path = project_root / ".kdeai" / "project.json"
    if project_path.exists():
        try:
            project_meta = json.loads(project_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            project_meta = {}
    path_casefold = bool(project_meta.get("path_casefold", os.name == "nt"))

    defaults = plan.get("apply_defaults") if isinstance(plan, Mapping) else None
    defaults = defaults if isinstance(defaults, Mapping) else {}

    selected_mode = str(apply_mode or defaults.get("mode") or "strict")
    selected_overwrite = str(overwrite or defaults.get("overwrite") or "conservative")
    post_index_flag = bool(post_index) and isinstance(workspace_conn, sqlite3.Connection)

    project_id = str(plan.get("project_id", ""))
    lang = str(plan.get("lang", ""))
    marker_flags, comment_prefixes, review_prefix, ai_prefix, ai_flag = (
        _marker_settings_from_config(config)
    )
    placeholder_patterns = plan.get("placeholder_patterns") or []

    errors: list[str] = []
    plan_marker_flags = plan.get("marker_flags")
    if plan_marker_flags is not None:
        if not isinstance(plan_marker_flags, (list, tuple)):
            errors.append("plan marker_flags must be a list")
        else:
            normalized = [str(flag) for flag in plan_marker_flags]
            if normalized != list(marker_flags):
                errors.append("plan marker_flags do not match config")

    plan_comment_prefixes = plan.get("comment_prefixes")
    if plan_comment_prefixes is not None:
        if not isinstance(plan_comment_prefixes, (list, tuple)):
            errors.append("plan comment_prefixes must be a list")
        else:
            normalized = [str(prefix) for prefix in plan_comment_prefixes]
            if normalized != list(comment_prefixes):
                errors.append("plan comment_prefixes do not match config")

    plan_ai_flag = plan.get("ai_flag")
    if plan_ai_flag is not None and str(plan_ai_flag) != ai_flag:
        errors.append("plan ai_flag does not match config")

    files_written: list[str] = []
    files_skipped: list[str] = []
    warnings: list[str] = []
    entries_applied = 0

    files = plan.get("files") if isinstance(plan, Mapping) else None
    if not isinstance(files, Iterable):
        return ApplyResult(files_written, files_skipped, entries_applied, ["invalid plan files"], warnings)
    if errors:
        return ApplyResult(files_written, files_skipped, entries_applied, errors, warnings)

    for file_item in files:
        if not isinstance(file_item, Mapping):
            continue
        file_path = str(file_item.get("file_path", ""))
        base_sha256 = str(file_item.get("base_sha256", ""))
        entries = list(file_item.get("entries", []))

        relpath_key = file_path.casefold() if path_casefold else file_path
        lock_path = locks.per_file_lock_path(
            project_root,
            locks.lock_id(project_id, relpath_key),
        )
        full_path = project_root / file_path
        file_result = apply_plan_to_file(
            file_path,
            entries,
            selected_mode,
            selected_overwrite,
            full_path=full_path,
            lock_path=lock_path,
            base_sha256=base_sha256,
            lang=lang,
            marker_flags=marker_flags,
            comment_prefixes=comment_prefixes,
            placeholder_patterns=placeholder_patterns,
            review_prefix=review_prefix,
        )

        if file_result.warnings:
            warnings.extend(file_result.warnings)

        if file_result.errors:
            errors.extend(file_result.errors)
            files_skipped.append(file_path)
            continue
        if file_result.skipped:
            files_skipped.append(file_path)
            continue

        if file_result.wrote:
            files_written.append(file_path)
            entries_applied += file_result.entries_applied
            if session_tm is not None and file_result.applied_entries:
                _update_session_tm(
                    session_tm,
                    entries=file_result.applied_entries,
                    lang=lang,
                    review_prefix=review_prefix,
                    ai_prefix=ai_prefix,
                    ai_flag=ai_flag,
                )
            if post_index_flag:
                try:
                    full_path = project_root / file_path
                    data = full_path.read_bytes()
                    sha256_hex = hashlib.sha256(data).hexdigest()
                    stat = full_path.stat()
                    workspace_tm.index_file_snapshot_tm(
                        workspace_conn,
                        file_path=file_path,
                        lang=lang,
                        bytes=data,
                        sha256=sha256_hex,
                        mtime_ns=stat.st_mtime_ns,
                        size=stat.st_size,
                        config=config,
                    )
                except Exception as exc:
                    warnings.append(f"post-index failed for {file_path}: {exc}")

    return ApplyResult(files_written, files_skipped, entries_applied, errors, warnings)
