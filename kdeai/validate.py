from __future__ import annotations

import re
from typing import Iterable, Mapping, Pattern


class NonEmptyValidator:
    error_message = "non-empty translation"

    def validate(
        self,
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
            return self.error_message
        if msgstr.strip() == "":
            return self.error_message
        return None


def _parse_nplurals(plural_forms: str | None) -> int | None:
    if not plural_forms:
        return None
    match = re.search(r"nplurals\s*=\s*(\d+)", plural_forms)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


class PluralConsistencyValidator:
    error_message = "plural key consistency"

    def validate(
        self,
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
        nplurals = _parse_nplurals(plural_forms)
        keys = list(msgstr_plural.keys())
        if any(not str(key).isdigit() for key in keys):
            return self.error_message
        if nplurals is None:
            return None
        expected = {str(idx) for idx in range(nplurals)}
        if set(str(key) for key in keys) != expected:
            return self.error_message
        return None


def _extract_tokens(text: str, pattern: re.Pattern[str]) -> list[str]:
    return [match.group(0) for match in pattern.finditer(text)]


def _plural_key_sorter(value: str) -> tuple[int, str]:
    text = str(value)
    if text.isdigit():
        return (0, f"{int(text):09d}")
    return (1, text)


class TagIntegrityValidator:
    error_message = "tag/placeholder integrity"

    def validate(
        self,
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
                    if target_tokens != source_tokens:
                        return self.error_message
            else:
                source_tokens = _extract_tokens(msgid, compiled)
                target_tokens = _extract_tokens(msgstr, compiled)
                if target_tokens != source_tokens:
                    return self.error_message
        return None


def validate_entry(
    *,
    msgid: str,
    msgid_plural: str,
    msgstr: str,
    msgstr_plural: Mapping[str, str],
    plural_forms: str | None,
    placeholder_patterns: Iterable[str | Pattern[str]],
) -> list[str]:
    validators = [
        NonEmptyValidator(),
        PluralConsistencyValidator(),
        TagIntegrityValidator(),
    ]
    errors: list[str] = []
    for validator in validators:
        error = validator.validate(
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
