from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping
import sqlite3

from kdeai import hash as kdehash
from kdeai.config import Config
from kdeai.po_utils import parse_msgstr_plural
from kdeai.tm_types import SessionTmView

DEFAULT_LOOKUP_SCOPES = ["session", "workspace", "reference"]


@dataclass(frozen=True)
class TmCandidate:
    source_key: str
    lang: str
    msgstr: str
    msgstr_plural: dict[str, str]
    review_status: str
    is_ai_generated: int
    translation_hash: str
    scope: str
    file_path: str = ""
    file_sha256: str = ""


def _lookup_scopes(
    config: Config | None,
    lookup_scopes: Iterable[str] | None,
) -> list[str]:
    if lookup_scopes is not None:
        if isinstance(lookup_scopes, (list, tuple)):
            return [str(scope) for scope in lookup_scopes]
        return [str(lookup_scopes)]
    if config is None:
        return DEFAULT_LOOKUP_SCOPES
    return list(config.tm.lookup_scopes)


def _is_write_eligible(msgstr: str, msgstr_plural: dict[str, str], has_plural: bool) -> bool:
    if has_plural:
        return any(value.strip() for value in msgstr_plural.values())
    return msgstr.strip() != ""


def _candidate_from_session(
    source_key: str,
    lang: str,
    entry: object,
) -> TmCandidate | None:
    if isinstance(entry, TmCandidate):
        return TmCandidate(
            source_key=source_key,
            lang=lang,
            msgstr=entry.msgstr,
            msgstr_plural=entry.msgstr_plural,
            review_status=entry.review_status,
            is_ai_generated=entry.is_ai_generated,
            translation_hash=entry.translation_hash,
            scope="session",
            file_path="",
            file_sha256="",
        )
    if not isinstance(entry, Mapping):
        return None

    msgstr = str(entry.get("msgstr", ""))
    msgstr_plural = parse_msgstr_plural(entry.get("msgstr_plural", {}))
    review_status = str(entry.get("review_status", "draft"))
    is_ai_generated = int(entry.get("is_ai_generated", 0))
    translation_hash = str(
        entry.get(
            "translation_hash",
            kdehash.translation_hash(source_key, lang, msgstr, msgstr_plural),
        )
    )

    return TmCandidate(
        source_key=source_key,
        lang=lang,
        msgstr=msgstr,
        msgstr_plural=msgstr_plural,
        review_status=review_status,
        is_ai_generated=is_ai_generated,
        translation_hash=translation_hash,
        scope="session",
    )


def _lookup_session(
    session_tm: SessionTmView | None,
    source_key: str,
    lang: str,
) -> TmCandidate | None:
    if not session_tm:
        return None
    entry = None
    if (source_key, lang) in session_tm:
        entry = session_tm[(source_key, lang)]
    if entry is None:
        return None
    return _candidate_from_session(source_key, lang, entry)


def _lookup_workspace(
    conn: sqlite3.Connection | None,
    source_key: str,
    lang: str,
) -> TmCandidate | None:
    if conn is None:
        return None
    try:
        row = conn.execute(
            "SELECT source_key, lang, file_path, msgstr, msgstr_plural, "
            "review_status, is_ai_generated, translation_hash "
            "FROM best_translations WHERE source_key = ? AND lang = ?",
            (source_key, lang),
        ).fetchone()
    except sqlite3.Error:
        return None
    if not row:
        return None

    return TmCandidate(
        source_key=str(row[0]),
        lang=str(row[1]),
        file_path=str(row[2]),
        msgstr=str(row[3]),
        msgstr_plural=parse_msgstr_plural(row[4]),
        review_status=str(row[5]),
        is_ai_generated=int(row[6]),
        translation_hash=str(row[7]),
        scope="workspace",
    )


def _lookup_reference(
    conn: sqlite3.Connection | None,
    source_key: str,
    lang: str,
) -> TmCandidate | None:
    if conn is None:
        return None
    try:
        row = conn.execute(
            "SELECT source_key, lang, file_path, file_sha256, msgstr, msgstr_plural, "
            "review_status, is_ai_generated, translation_hash "
            "FROM best_translations WHERE source_key = ? AND lang = ?",
            (source_key, lang),
        ).fetchone()
    except sqlite3.Error:
        return None
    if not row:
        return None

    return TmCandidate(
        source_key=str(row[0]),
        lang=str(row[1]),
        file_path=str(row[2]),
        file_sha256=str(row[3]),
        msgstr=str(row[4]),
        msgstr_plural=parse_msgstr_plural(row[5]),
        review_status=str(row[6]),
        is_ai_generated=int(row[7]),
        translation_hash=str(row[8]),
        scope="reference",
    )


def lookup_tm_exact(
    source_key: str,
    lang: str,
    *,
    has_plural: bool,
    config: Config | None = None,
    lookup_scopes: Iterable[str] | None = None,
    session_tm: SessionTmView | None = None,
    workspace_conn: sqlite3.Connection | None = None,
    reference_conn: sqlite3.Connection | None = None,
) -> TmCandidate | None:
    scopes = _lookup_scopes(config, lookup_scopes)

    for scope in scopes:
        if scope == "session":
            candidate = _lookup_session(session_tm, source_key, lang)
        elif scope == "workspace":
            candidate = _lookup_workspace(workspace_conn, source_key, lang)
        elif scope == "reference":
            candidate = _lookup_reference(reference_conn, source_key, lang)
        else:
            continue

        if candidate is None:
            continue
        if _is_write_eligible(candidate.msgstr, candidate.msgstr_plural, has_plural):
            return candidate

    return None
