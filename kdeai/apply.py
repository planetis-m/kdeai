from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping
import hashlib
import os
import re
import sqlite3
import tempfile

import polib

from kdeai import hash as kdehash
from kdeai import locks
from kdeai import po_model
from kdeai import po_utils
from kdeai import state as kdestate
from kdeai.config import Config
from kdeai import snapshot
from kdeai import validate
from kdeai import workspace_tm
from kdeai.tm_types import SessionTm

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
class ApplyContext:
    lang: str
    mode: str
    overwrite_policy: str
    marker_flags: list[str]
    comment_prefixes: list[str]
    review_prefix: str
    ai_prefix: str
    ai_flag: str
    placeholder_patterns: list[str | re.Pattern[str]]


@dataclass(frozen=True)
class ApplyFileResult:
    file_path: str
    wrote: bool
    skipped: bool
    entries_applied: int
    errors: list[str]
    warnings: list[str]
    applied_entries: list[polib.POEntry]


def _validate_plan_header(
    plan: Mapping[str, object],
    *,
    project_id: str,
    config: Config,
    marker_flags: list[str],
    comment_prefixes: list[str],
    ai_flag: str,
    placeholder_patterns: list[str],
) -> list[str]:
    plan_project_id = str(plan.get("project_id", "")).strip()
    if not plan_project_id:
        return ["plan project_id missing"]
    if plan_project_id != project_id:
        return ["plan project_id does not match current project"]
    plan_config_hash = str(plan.get("config_hash", ""))
    if not plan_config_hash or plan_config_hash != config.config_hash:
        return ["plan config_hash does not match current config"]

    errors: list[str] = []
    plan_marker_flags = plan.get("marker_flags")
    if plan_marker_flags is not None:
        if not isinstance(plan_marker_flags, (list, tuple)):
            errors.append("plan marker_flags must be a list")
        else:
            normalized = [str(flag) for flag in plan_marker_flags]
            if sorted(normalized) != sorted(marker_flags):
                errors.append("plan marker_flags do not match config")

    plan_comment_prefixes = plan.get("comment_prefixes")
    if plan_comment_prefixes is not None:
        if not isinstance(plan_comment_prefixes, (list, tuple)):
            errors.append("plan comment_prefixes must be a list")
        else:
            normalized = [str(prefix) for prefix in plan_comment_prefixes]
            if sorted(normalized) != sorted(comment_prefixes):
                errors.append("plan comment_prefixes do not match config")

    plan_ai_flag = plan.get("ai_flag")
    if plan_ai_flag is not None and str(plan_ai_flag) != ai_flag:
        errors.append("plan ai_flag does not match config")

    plan_placeholder_patterns = plan.get("placeholder_patterns")
    if plan_placeholder_patterns is not None:
        if not isinstance(plan_placeholder_patterns, (list, tuple)):
            errors.append("plan placeholder_patterns must be a list")
        else:
            normalized = [str(pattern) for pattern in plan_placeholder_patterns]
            if normalized != placeholder_patterns:
                errors.append("plan placeholder_patterns do not match config")

    return errors


def _load_po_from_bytes(data: bytes) -> polib.POFile:
    return po_utils.load_po_from_bytes(data)


def _fsync_file(path: Path) -> None:
    with open(path, "rb") as f:
        os.fsync(f.fileno())


def _entry_key(entry: polib.POEntry) -> tuple[str, str, str]:
    return (
        entry.msgctxt or "",
        entry.msgid,
        entry.msgid_plural or "",
    )


def _marker_settings_from_config(config: Config) -> tuple[list[str], list[str], str, str, str]:
    return po_utils.marker_settings_from_config(config)


def entry_state_hash(
    entry: polib.POEntry,
    *,
    lang: str,
    marker_flags: Iterable[str],
    comment_prefixes: Iterable[str],
) -> str:
    return kdestate.entry_state_hash(
        entry,
        lang=lang,
        marker_flags=marker_flags,
        comment_prefixes=comment_prefixes,
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
    entry.msgstr_plural = kdestate.canonical_plural_map(msgstr_plural)


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
        msgstr_plural = kdestate.canonical_plural_map(entry.msgstr_plural)
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


def _is_tool_owned_prefix(prefix: str, *, tool_prefixes: Iterable[str]) -> bool:
    return any(prefix.startswith(tool_prefix) for tool_prefix in tool_prefixes)


def _validate_plan_entry(entry_item: object, *, tool_prefixes: list[str]) -> list[str]:
    if not isinstance(entry_item, Mapping):
        return ["plan entry must be an object"]
    errors: list[str] = []
    msgid = entry_item.get("msgid")
    if not isinstance(msgid, str):
        errors.append("msgid must be a string")
    if "msgctxt" in entry_item and not isinstance(entry_item.get("msgctxt"), str):
        errors.append("msgctxt must be a string when provided")
    if "msgid_plural" in entry_item and not isinstance(entry_item.get("msgid_plural"), str):
        errors.append("msgid_plural must be a string when provided")
    base_state_hash = entry_item.get("base_state_hash")
    if not isinstance(base_state_hash, str):
        errors.append("base_state_hash must be a string")
    action = str(entry_item.get("action", ""))
    if action in {"copy_tm", "llm"}:
        translation = entry_item.get("translation")
        if not isinstance(translation, Mapping):
            errors.append("translation must be an object for copy_tm/llm")
        else:
            if not isinstance(translation.get("msgstr"), str):
                errors.append("translation.msgstr must be a string")
            if not isinstance(translation.get("msgstr_plural"), Mapping):
                errors.append("translation.msgstr_plural must be an object")
    flags = entry_item.get("flags")
    if flags is not None:
        if not isinstance(flags, Mapping):
            errors.append("flags must be an object when provided")
        else:
            add_flags = flags.get("add", [])
            remove_flags = flags.get("remove", [])
            if not isinstance(add_flags, (list, tuple)):
                errors.append("flags.add must be a list")
            elif any(not isinstance(flag, str) for flag in add_flags):
                errors.append("flags.add must be a list of strings")
            if not isinstance(remove_flags, (list, tuple)):
                errors.append("flags.remove must be a list")
            elif any(not isinstance(flag, str) for flag in remove_flags):
                errors.append("flags.remove must be a list of strings")
    comments = entry_item.get("comments")
    if comments is not None:
        if not isinstance(comments, Mapping):
            errors.append("comments must be an object when provided")
        else:
            remove_prefixes = comments.get("remove_prefixes", [])
            ensure_lines = comments.get("ensure_lines", [])
            append = comments.get("append", "")
            if not isinstance(remove_prefixes, (list, tuple)):
                errors.append("plan comments remove_prefixes must be a list")
            elif any(not isinstance(prefix, str) for prefix in remove_prefixes):
                errors.append("plan comments remove_prefixes must be a list of strings")
            elif any(
                not _is_tool_owned_prefix(prefix, tool_prefixes=tool_prefixes)
                for prefix in remove_prefixes
            ):
                errors.append("plan comments remove_prefixes must use tool prefixes")
            if not isinstance(ensure_lines, (list, tuple)):
                errors.append("plan comments ensure_lines must be a list")
            elif any(not isinstance(line, str) for line in ensure_lines):
                errors.append("plan comments ensure_lines must be a list of strings")
            elif any(
                not any(line.startswith(prefix) for prefix in tool_prefixes)
                for line in ensure_lines
            ):
                errors.append("plan comments ensure_lines must use tool prefixes")
            if not isinstance(append, str):
                errors.append("plan comments append must be a string")
            elif append != "" and not append.endswith("\n"):
                errors.append("plan comments append must end with \\n")
    return errors


def apply_plan_to_file(
    file_path: str,
    plan_items: list,
    *,
    ctx: ApplyContext,
    full_path: Path,
    lock_path: Path,
    base_sha256: str,
) -> ApplyFileResult:
    tool_prefixes = [str(prefix) for prefix in ctx.comment_prefixes]
    marker_flags = list(ctx.marker_flags)
    if ctx.ai_flag not in marker_flags:
        marker_flags.append(ctx.ai_flag)
    comment_prefixes = list(ctx.comment_prefixes)

    file_warnings: list[str] = []
    phase_a = snapshot.locked_read_file(full_path, lock_path, relpath=file_path)
    if ctx.mode == "strict" and (not base_sha256 or phase_a.sha256 != base_sha256):
        file_warnings.append(f"{file_path}: skipped (strict): base_sha256 mismatch")
        return ApplyFileResult(file_path, False, True, 0, [], file_warnings, [])

    po_file = _load_po_from_bytes(phase_a.bytes)
    entry_map = {_entry_key(entry): entry for entry in po_file if entry.msgid != "" and not entry.obsolete}

    file_errors: list[str] = []
    to_apply: list[tuple[Mapping[str, object], polib.POEntry, str]] = []
    mismatch = False
    mismatch_reason: str | None = None
    for index, entry_item in enumerate(plan_items):
        entry_errors = _validate_plan_entry(entry_item, tool_prefixes=tool_prefixes)
        if entry_errors:
            for error in entry_errors:
                file_errors.append(f"{file_path}: invalid plan entry {index}: {error}")
            return ApplyFileResult(file_path, False, True, 0, file_errors, file_warnings, [])
        action = str(entry_item.get("action", ""))
        if action == "needs_llm":
            continue
        if action not in {"copy_tm", "llm"}:
            if action:
                file_errors.append(f"{file_path}: unsupported action: {action}")
            else:
                file_errors.append(f"{file_path}: unsupported action: missing")
            return ApplyFileResult(file_path, False, True, 0, file_errors, file_warnings, [])
        key = (
            str(entry_item.get("msgctxt", "")),
            str(entry_item.get("msgid", "")),
            str(entry_item.get("msgid_plural", "")),
        )
        entry = entry_map.get(key)
        if entry is None:
            if ctx.mode == "strict":
                mismatch = True
                mismatch_reason = "missing entry"
                break
            continue
        current_hash = kdestate.entry_state_hash(
            entry,
            lang=ctx.lang,
            marker_flags=marker_flags,
            comment_prefixes=comment_prefixes,
        )
        base_state_hash = str(entry_item.get("base_state_hash", ""))
        if current_hash != base_state_hash:
            if ctx.mode == "strict":
                mismatch = True
                mismatch_reason = "entry state mismatch"
                break
            continue
        to_apply.append((entry_item, entry, action))

    if mismatch:
        if ctx.mode == "strict":
            reason = mismatch_reason or "entry state mismatch"
            file_warnings.append(f"{file_path}: skipped (strict): {reason}")
        return ApplyFileResult(file_path, False, True, 0, [], file_warnings, [])

    applied_entries: list[polib.POEntry] = []
    applied_in_file = 0
    for entry_item, entry, action in to_apply:
        current_non_empty = _has_non_empty_translation(entry)
        reviewed = _is_reviewed(entry, ctx.review_prefix)
        if not _can_overwrite(current_non_empty, reviewed, ctx.overwrite_policy):
            continue

        translation = entry_item.get("translation")
        msgstr = translation.get("msgstr", "")
        msgstr_plural = translation.get("msgstr_plural", {})
        _set_translation(entry, msgstr=msgstr, msgstr_plural=msgstr_plural)

        flags = entry_item.get("flags")
        if flags is not None:
            add_flags = flags.get("add", [])
            remove_flags = flags.get("remove", [])
            _apply_flags(entry, add_flags, remove_flags)

        comments = entry_item.get("comments")
        if comments is not None:
            remove_prefixes = comments.get("remove_prefixes", [])
            ensure_lines = comments.get("ensure_lines", [])
            append = str(comments.get("append", ""))
            normalized_remove = [str(prefix) for prefix in remove_prefixes]
            normalized_ensure = [str(line) for line in ensure_lines]
            _apply_comments(entry, normalized_remove, normalized_ensure, append)

        plural_forms = po_file.metadata.get("Plural-Forms")
        validation_errors = validate.validate_entry(
            msgid=entry.msgid,
            msgid_plural=entry.msgid_plural or "",
            msgstr=entry.msgstr or "",
            msgstr_plural=kdestate.canonical_plural_map(entry.msgstr_plural),
            plural_forms=plural_forms,
            placeholder_patterns=ctx.placeholder_patterns,
        )
        if validation_errors:
            file_errors.extend(validation_errors)
            break

        applied_in_file += 1
        applied_entries.append(entry)

    if file_errors:
        return ApplyFileResult(file_path, False, True, 0, file_errors, file_warnings, [])

    if applied_in_file == 0:
        return ApplyFileResult(file_path, False, True, 0, [], file_warnings, [])

    tmp_handle = tempfile.NamedTemporaryFile(
        dir=str(full_path.parent),
        suffix=".po",
        delete=False,
    )
    tmp_name = tmp_handle.name
    tmp_handle.close()

    try:
        po_file.save(tmp_name)
        _fsync_file(Path(tmp_name))
    except Exception:
        if Path(tmp_name).exists():
            os.unlink(tmp_name)
        raise

    cleanup_path = tmp_name
    result: ApplyFileResult | None = None
    try:
        with locks.acquire_file_lock(lock_path):
            current_bytes = full_path.read_bytes()
            current_sha = hashlib.sha256(current_bytes).hexdigest()
            if current_sha != phase_a.sha256:
                file_warnings.append(f"{file_path}: skipped: file changed since phase A")
                result = ApplyFileResult(file_path, False, True, 0, [], file_warnings, [])
            else:
                os.replace(tmp_name, full_path)
                cleanup_path = None
        if result is not None:
            if cleanup_path and Path(cleanup_path).exists():
                os.unlink(cleanup_path)
            return result
    except Exception:
        if cleanup_path and Path(cleanup_path).exists():
            os.unlink(cleanup_path)
        raise

    return ApplyFileResult(file_path, True, False, applied_in_file, [], file_warnings, applied_entries)


def apply_plan(
    plan: Mapping[str, object],
    *,
    project_root: Path,
    project_id: str,
    path_casefold: bool,
    config: Config,
    apply_mode: str | None = None,
    overwrite: str | None = None,
    post_index: bool | None = None,
    workspace_conn: sqlite3.Connection | None = None,
    session_tm: SessionTm | None = None,
) -> ApplyResult:
    defaults = plan.get("apply_defaults") if isinstance(plan, Mapping) else None
    defaults = defaults if isinstance(defaults, Mapping) else {}

    selected_mode = str(apply_mode or defaults.get("mode") or "strict")
    selected_overwrite = str(overwrite or defaults.get("overwrite") or "conservative")
    if post_index is None:
        post_index_setting = str(defaults.get("post_index") or "off")
        effective_post_index = post_index_setting == "on"
    else:
        effective_post_index = bool(post_index)
    post_index_flag = effective_post_index and isinstance(workspace_conn, sqlite3.Connection)

    lang = str(plan.get("lang", ""))
    plan_format = plan.get("format") if isinstance(plan, Mapping) else None
    marker_flags, comment_prefixes, review_prefix, ai_prefix, ai_flag = (
        _marker_settings_from_config(config)
    )
    placeholder_patterns = list(config.apply.validation_patterns)
    compiled_placeholder_patterns = [re.compile(pattern) for pattern in placeholder_patterns]
    ctx = ApplyContext(
        lang=lang,
        mode=selected_mode,
        overwrite_policy=selected_overwrite,
        marker_flags=list(marker_flags),
        comment_prefixes=list(comment_prefixes),
        review_prefix=review_prefix,
        ai_prefix=ai_prefix,
        ai_flag=ai_flag,
        placeholder_patterns=compiled_placeholder_patterns,
    )

    errors = _validate_plan_header(
        plan,
        project_id=project_id,
        config=config,
        marker_flags=marker_flags,
        comment_prefixes=comment_prefixes,
        ai_flag=ai_flag,
        placeholder_patterns=placeholder_patterns,
    )
    if errors:
        return ApplyResult([], [], 0, errors, [])

    errors = []
    if str(plan_format) != "1":
        errors.append("unsupported plan format")
    if selected_mode not in {"strict", "rebase"}:
        errors.append(f"unsupported apply mode: {selected_mode}")
    if not lang.strip():
        errors.append("plan lang missing")
    if selected_overwrite not in ALLOWED_OVERWRITE_POLICIES:
        errors.append(f"unsupported overwrite mode: {selected_overwrite}")

    files_written: list[str] = []
    files_skipped: list[str] = []
    warnings: list[str] = []
    entries_applied = 0

    files = plan.get("files") if isinstance(plan, Mapping) else None
    if not isinstance(files, list):
        return ApplyResult(files_written, files_skipped, entries_applied, ["plan files must be a list"], warnings)
    if errors:
        return ApplyResult(files_written, files_skipped, entries_applied, errors, warnings)

    files_list = list(files)
    project_root_resolved = project_root.resolve()
    path_errors: list[str] = []
    entries_errors: list[str] = []
    for file_item in files_list:
        if not isinstance(file_item, Mapping):
            continue
        if "entries" in file_item and not isinstance(file_item.get("entries"), list):
            entry_file_path = str(file_item.get("file_path", ""))
            if entry_file_path:
                entries_errors.append(f"plan entries must be a list: {entry_file_path}")
            else:
                entries_errors.append("plan entries must be a list")
        file_path = str(file_item.get("file_path", ""))
        if not file_path:
            path_errors.append("plan file_path is invalid: empty")
            continue
        if "\\" in file_path:
            path_errors.append(f"plan file_path is invalid: {file_path}")
            continue
        if Path(file_path).is_absolute():
            path_errors.append(f"plan file_path is invalid: {file_path}")
            continue
        if PurePosixPath(file_path).as_posix() != file_path:
            path_errors.append(f"plan file_path is invalid: {file_path}")
            continue
        if "." in PurePosixPath(file_path).parts:
            path_errors.append(f"plan file_path is invalid: {file_path}")
            continue
        if ".." in PurePosixPath(file_path).parts:
            path_errors.append(f"plan file_path is invalid: {file_path}")
            continue
        full_path = (project_root / file_path).resolve()
        if not full_path.is_relative_to(project_root_resolved):
            path_errors.append(f"plan file_path is invalid: {file_path}")

    if path_errors:
        return ApplyResult(files_written, files_skipped, entries_applied, path_errors, warnings)
    if entries_errors:
        return ApplyResult(files_written, files_skipped, entries_applied, entries_errors, warnings)

    for file_item in files_list:
        if not isinstance(file_item, Mapping):
            continue
        file_path = str(file_item.get("file_path", ""))
        base_sha256 = str(file_item.get("base_sha256", ""))
        if "entries" in file_item:
            entries = list(file_item.get("entries"))
        else:
            entries = []

        relpath_key = file_path.casefold() if path_casefold else file_path
        lock_path = locks.per_file_lock_path(
            project_root,
            locks.lock_id(project_id, relpath_key),
        )
        full_path = (project_root / file_path).resolve()
        try:
            file_result = apply_plan_to_file(
                file_path,
                entries,
                ctx=ctx,
                full_path=full_path,
                lock_path=lock_path,
                base_sha256=base_sha256,
            )
        except Exception as exc:
            errors.append(f"{file_path}: {exc}")
            files_skipped.append(file_path)
            continue

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
                    snap = snapshot.locked_read_file(full_path, lock_path, relpath=file_path)
                    workspace_tm.index_file_snapshot_tm(
                        workspace_conn,
                        file_path=file_path,
                        lang=lang,
                        bytes=snap.bytes,
                        sha256=snap.sha256,
                        mtime_ns=snap.mtime_ns,
                        size=snap.size,
                        config=config,
                    )
                except Exception as exc:
                    warnings.append(f"post-index failed for {file_path}: {exc}")

    return ApplyResult(files_written, files_skipped, entries_applied, errors, warnings)
