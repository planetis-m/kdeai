from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Pattern

from kdeai import po_utils

NON_EMPTY_ERROR = "non-empty translation"
PLURAL_CONSISTENCY_ERROR = "plural key consistency"
TAG_INTEGRITY_ERROR = "tag/placeholder integrity"


@dataclass(frozen=True)
class ValidationRequest:
    msgid: str
    msgid_plural: str
    msgstr: str
    msgstr_plural: Mapping[str, str]
    plural_forms: str | None
    placeholder_patterns: Iterable[str | Pattern[str]]


def validate_non_empty(request: ValidationRequest) -> str | None:
    if po_utils.is_translation_non_empty(
        request.msgstr,
        request.msgstr_plural,
        bool(request.msgid_plural),
    ):
        return None
    return NON_EMPTY_ERROR


def validate_plural_consistency(request: ValidationRequest) -> str | None:
    if not request.msgid_plural:
        return None
    nplurals = po_utils.parse_nplurals(request.plural_forms)
    keys = list(request.msgstr_plural.keys())
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


def validate_tag_integrity(request: ValidationRequest) -> str | None:
    if not request.placeholder_patterns:
        return None
    plural = bool(request.msgid_plural)
    for raw_pattern in request.placeholder_patterns:
        if not isinstance(raw_pattern, re.Pattern):
            raw_pattern = re.compile(str(raw_pattern))
        compiled = raw_pattern
        if plural:
            for key in sorted(request.msgstr_plural.keys(), key=_plural_key_sorter):
                source_text = request.msgid if str(key) == "0" else request.msgid_plural
                source_tokens = _extract_tokens(source_text, compiled)
                target_tokens = _extract_tokens(
                    str(request.msgstr_plural.get(key, "")),
                    compiled,
                )
                if Counter(target_tokens) != Counter(source_tokens):
                    return TAG_INTEGRITY_ERROR
        else:
            source_tokens = _extract_tokens(request.msgid, compiled)
            target_tokens = _extract_tokens(request.msgstr, compiled)
            if Counter(target_tokens) != Counter(source_tokens):
                return TAG_INTEGRITY_ERROR
    return None


VALIDATORS = [
    validate_non_empty,
    validate_plural_consistency,
    validate_tag_integrity,
]


def validate_entry(request: ValidationRequest) -> list[str]:
    errors: list[str] = []
    for validator in VALIDATORS:
        error = validator(request)
        if error:
            errors.append(error)
    return errors
