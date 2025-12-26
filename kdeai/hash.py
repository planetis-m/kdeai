from __future__ import annotations

from typing import Iterable, Mapping
import hashlib
import json


def sha256_hex_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_hex_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_json_bytes(obj: object) -> bytes:
    return canonical_json(obj).encode("utf-8")


def source_key(msgctxt: str, msgid: str, msgid_plural: str) -> str:
    return sha256_hex_text(f"{msgctxt}\u0004{msgid}\u0000{msgid_plural}")


def canonical_msgstr_plural(msgstr_plural: Mapping[str, str]) -> str:
    canonical = {str(k): str(v) for k, v in msgstr_plural.items()}
    return canonical_json(canonical)


def translation_hash(
    source_key_hex: str,
    lang: str,
    msgstr: str,
    msgstr_plural: Mapping[str, str],
) -> str:
    msgstr_plural_json = canonical_msgstr_plural(msgstr_plural)
    body = (
        "v1\n"
        f"source_key={source_key_hex}\n"
        f"lang={lang}\n"
        f"msgstr={msgstr}\n"
        f"msgstr_plural={msgstr_plural_json}\n"
    )
    return sha256_hex_text(body)


def _normalize_lines(lines: Iterable[str]) -> str:
    return "\n".join(line.rstrip("\n") for line in lines)


def state_hash(
    source_key_hex: str,
    lang: str,
    msgstr: str,
    msgstr_plural: Mapping[str, str],
    marker_flags: Iterable[str],
    tool_comment_lines: Iterable[str],
) -> str:
    msgstr_plural_json = canonical_msgstr_plural(msgstr_plural)
    marker_flags_str = ",".join(sorted(marker_flags))
    tool_lines = _normalize_lines(tool_comment_lines)
    body = (
        "v2\n"
        f"source_key={source_key_hex}\n"
        f"lang={lang}\n"
        f"msgstr={msgstr}\n"
        f"msgstr_plural={msgstr_plural_json}\n"
        f"marker_flags={marker_flags_str}\n"
        f"tool_comment_lines={tool_lines}\n"
    )
    return sha256_hex_text(body)


def term_key(lemma_tokens: Iterable[str]) -> str:
    canonical = "v1\n" + "\u001f".join(lemma_tokens)
    return sha256_hex_text(canonical)
