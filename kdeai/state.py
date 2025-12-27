from __future__ import annotations

from typing import Iterable, Mapping

import polib

from kdeai import hash as kdehash
from kdeai import po_model
from kdeai import po_utils

DEFAULT_COMMENT_PREFIXES = po_utils.DEFAULT_COMMENT_PREFIXES
DEFAULT_MARKER_FLAGS = po_utils.DEFAULT_MARKER_FLAGS


def canonical_plural_map(msgstr_plural: Mapping[str, object]) -> dict[str, str]:
    canonical = {str(k): str(v) for k, v in msgstr_plural.items()}
    return dict(sorted(canonical.items(), key=lambda item: item[0]))


def entry_state_hash(
    entry: polib.POEntry,
    *,
    lang: str,
    marker_flags: Iterable[str] | None = None,
    comment_prefixes: Iterable[str] | None = None,
) -> str:
    marker_flags_list = list(marker_flags or DEFAULT_MARKER_FLAGS)
    comment_prefixes_list = list(comment_prefixes or DEFAULT_COMMENT_PREFIXES)
    source_key = po_model.source_key_for(entry.msgctxt, entry.msgid, entry.msgid_plural)
    msgstr = entry.msgstr or ""
    msgstr_plural = canonical_plural_map(entry.msgstr_plural)
    marker_flags_present = [flag for flag in marker_flags_list if flag in entry.flags]
    tool_lines = po_utils.tool_comment_lines(entry.tcomment, comment_prefixes_list)
    return kdehash.state_hash(
        source_key,
        lang,
        msgstr,
        msgstr_plural,
        marker_flags_present,
        tool_lines,
    )
