from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional
import json
import re
import tempfile

import polib

from kdeai.config import Config
from kdeai.constants import ReviewStatus

DEFAULT_COMMENT_PREFIXES = [
    "KDEAI:",
    "KDEAI-AI:",
    "KDEAI-TM:",
    "KDEAI-REVIEW:",
]

DEFAULT_MARKER_FLAGS = ["fuzzy"]


def load_po_from_bytes(data: bytes) -> polib.POFile:
    with tempfile.NamedTemporaryFile(suffix=".po", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return polib.pofile(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def tool_comment_lines(text: str | None, prefixes: Iterable[str]) -> list[str]:
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


def is_reviewed(entry: polib.POEntry, review_prefix: str) -> bool:
    lines = tool_comment_lines(entry.tcomment, [review_prefix])
    return bool(lines)


def derive_review_status(
    msgstr: str,
    msgstr_plural: Mapping[str, str],
    has_plural: bool,
    flags: Iterable[str],
    tcomment: str | None,
    review_prefix: str,
) -> str:
    non_empty = is_translation_non_empty(msgstr, msgstr_plural, has_plural)
    if not non_empty:
        return ReviewStatus.UNREVIEWED
    if "fuzzy" in flags:
        return ReviewStatus.NEEDS_REVIEW
    if tool_comment_lines(tcomment, [review_prefix]):
        return ReviewStatus.REVIEWED
    return ReviewStatus.DRAFT


def derive_review_status_entry(entry: polib.POEntry, review_prefix: str) -> str:
    return derive_review_status(
        entry.msgstr or "",
        entry.msgstr_plural or {},
        bool(entry.msgid_plural),
        entry.flags,
        entry.tcomment,
        review_prefix,
    )


def derive_is_ai_generated(
    flags: Iterable[str],
    tcomment: str | None,
    ai_flag: str,
    ai_prefix: str,
) -> int:
    if ai_flag in flags:
        return 1
    if tool_comment_lines(tcomment, [ai_prefix]):
        return 1
    return 0


def derive_is_ai_generated_entry(entry: polib.POEntry, ai_flag: str, ai_prefix: str) -> int:
    return derive_is_ai_generated(entry.flags, entry.tcomment, ai_flag, ai_prefix)


def is_translation_non_empty(
    msgstr: str,
    msgstr_plural: Mapping[str, str],
    has_plural: bool,
) -> bool:
    if has_plural:
        return any(str(value).strip() for value in msgstr_plural.values())
    return msgstr.strip() != ""


def has_non_empty_translation(entry: polib.POEntry) -> bool:
    msgstr = entry.msgstr or ""
    msgstr_plural = entry.msgstr_plural or {}
    return is_translation_non_empty(msgstr, msgstr_plural, bool(entry.msgid_plural))


def can_overwrite(current_non_empty: bool, reviewed: bool, overwrite: str) -> bool:
    if overwrite == "conservative":
        return not current_non_empty and not reviewed
    if overwrite == "allow-nonempty":
        return not reviewed
    if overwrite == "allow-reviewed":
        return not reviewed or (reviewed and not current_non_empty)
    if overwrite == "all":
        return True
    raise ValueError(f"unsupported overwrite mode: {overwrite}")


def ensure_ai_flag_in_markers(
    marker_flags: Iterable[str],
    ai_flag: str,
) -> list[str]:
    """Return marker_flags with ai_flag included for state_hash computation."""
    flags = list(marker_flags)
    if ai_flag not in flags:
        flags.append(ai_flag)
    return flags


def marker_settings_from_config(config: Config) -> tuple[list[str], list[str], str, str, str]:
    prefixes = config.markers.comment_prefixes
    ordered = [prefixes.tool, prefixes.ai, prefixes.tm, prefixes.review]
    return DEFAULT_MARKER_FLAGS, ordered, prefixes.review, prefixes.ai, config.markers.ai_flag


def iter_po_paths(
    project_root: Path,
    raw_paths: Optional[list[Path]],
    *,
    excluded_dirnames: Iterable[str] | None = None,
) -> list[Path]:
    roots = raw_paths if raw_paths else [project_root]
    seen: set[Path] = set()
    results: list[Path] = []
    excluded = set(excluded_dirnames or {".kdeai", ".git"})

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
            if any(part in excluded for part in candidate.parts):
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            results.append(resolved)
    return sorted(results, key=lambda item: str(item))


def normalize_relpath(project_root: Path, path: Path) -> str:
    resolved = path.resolve()
    relpath = resolved.relative_to(project_root.resolve())
    return relpath.as_posix()


def relpath_key(relpath: str, path_casefold: bool) -> str:
    return relpath.casefold() if path_casefold else relpath


def read_json(path: Path, label: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def parse_nplurals(plural_forms: str | None) -> int | None:
    if not plural_forms:
        return None
    match = re.search(r"nplurals\s*=\s*(\d+)", plural_forms, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        value = int(match.group(1))
    except ValueError:
        return None
    return value if value >= 1 else None


def parse_msgstr_plural(value: object) -> dict[str, str]:
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    return {}
