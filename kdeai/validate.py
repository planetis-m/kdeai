from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Mapping, Pattern

from kdeai.po_utils import parse_nplurals

NON_EMPTY_ERROR = "non-empty translation"
PLURAL_CONSISTENCY_ERROR = "plural key consistency"
TAG_INTEGRITY_ERROR = "tag/placeholder integrity"


def validate_non_empty(
    *,
    msgid: str,
    msgid_plural: str,
    msgstr: str,
    msgstr_plural: Mapping[str, str],
    plural_forms: str | None,
    placeholder_patterns: Iterable[str | Pattern[str]],
) -> str | None:
    _ = msgid, plural_forms, placeholder_patterns
    if msgid_plural:
        if any(str(value).strip() for value in msgstr_plural.values()):
            return None
        return NON_EMPTY_ERROR
    if msgstr.strip() == "":
        return NON_EMPTY_ERROR
    return None


def validate_plural_consistency(
    *,
    msgid: str,
    msgid_plural: str,
    msgstr: str,
    msgstr_plural: Mapping[str, str],
    plural_forms: str | None,
    placeholder_patterns: Iterable[str | Pattern[str]],
) -> str | None:
    _ = msgid, msgstr, placeholder_patterns
    if not msgid_plural:
        return None
    nplurals = parse_nplurals(plural_forms)
    keys = list(msgstr_plural.keys())
    if any(not str(key).isdigit() for key in keys):
        return PLURAL_CONSISTENCY_ERROR
    if nplurals is None:
        return None
    expected = {str(idx) for idx in range(nplurals)}
    if set(str(key) for key in keys) != expected:
        return PLURAL_CONSISTENCY_ERROR
    return None


def _extract_tokens(text: str, pattern: re.Pattern[str]) -> list[str]:
    return [match.group(0) for match in pattern.finditer(text)]


def _plural_key_sorter(value: str) -> tuple[int, str]:
    text = str(value)
    if text.isdigit():
        return (0, f"{int(text):09d}")
    return (1, text)


def validate_tag_integrity(
    *,
    msgid: str,
    msgid_plural: str,
    msgstr: str,
    msgstr_plural: Mapping[str, str],
    plural_forms: str | None,
    placeholder_patterns: Iterable[str | Pattern[str]],
) -> str | None:
    _ = plural_forms
    if not placeholder_patterns:
        return None
    plural = bool(msgid_plural)
    for raw_pattern in placeholder_patterns:
        if not isinstance(raw_pattern, re.Pattern):
            raw_pattern = re.compile(str(raw_pattern))
        compiled = raw_pattern
        if plural:
            for key in sorted(msgstr_plural.keys(), key=_plural_key_sorter):
                source_text = msgid if str(key) == "0" else msgid_plural
                source_tokens = _extract_tokens(source_text, compiled)
                target_tokens = _extract_tokens(str(msgstr_plural.get(key, "")), compiled)
                if Counter(target_tokens) != Counter(source_tokens):
                    return TAG_INTEGRITY_ERROR
        else:
            source_tokens = _extract_tokens(msgid, compiled)
            target_tokens = _extract_tokens(msgstr, compiled)
            if Counter(target_tokens) != Counter(source_tokens):
                return TAG_INTEGRITY_ERROR
    return None


VALIDATORS = [
    validate_non_empty,
    validate_plural_consistency,
    validate_tag_integrity,
]


def validate_entry(
    *,
    msgid: str,
    msgid_plural: str,
    msgstr: str,
    msgstr_plural: Mapping[str, str],
    plural_forms: str | None,
    placeholder_patterns: Iterable[str | Pattern[str]],
) -> list[str]:
    errors: list[str] = []
    for validator in VALIDATORS:
        error = validator(
            msgid=msgid,
            msgid_plural=msgid_plural,
            msgstr=msgstr,
            msgstr_plural=msgstr_plural,
            plural_forms=plural_forms,
            placeholder_patterns=placeholder_patterns,
        )
        if error:
            errors.append(error)
    return errors
