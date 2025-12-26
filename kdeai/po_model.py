from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping
import os
import tempfile

import polib

from kdeai import hash as kdehash


@dataclass(frozen=True)
class PoUnit:
    msgctxt: str
    msgid: str
    msgid_plural: str
    msgstr: str
    msgstr_plural: Mapping[str, str]
    source_key: str
    source_text: str


def _normalize_field(value: str | None) -> str:
    return value if value is not None else ""


def source_key_for(msgctxt: str | None, msgid: str, msgid_plural: str | None) -> str:
    norm_ctxt = _normalize_field(msgctxt)
    norm_plural = _normalize_field(msgid_plural)
    return kdehash.source_key(norm_ctxt, msgid, norm_plural)


def source_text_v1(msgctxt: str | None, msgid: str, msgid_plural: str | None) -> str:
    norm_ctxt = _normalize_field(msgctxt)
    norm_plural = _normalize_field(msgid_plural)
    return f"ctx:{norm_ctxt}\nid:{msgid}\npl:{norm_plural}"


def _iter_translation_entries(po_file: polib.POFile) -> Iterable[polib.POEntry]:
    for entry in po_file:
        if entry.obsolete:
            continue
        if entry.msgid == "":
            continue
        yield entry


def _po_entry_to_unit(entry: polib.POEntry) -> PoUnit:
    msgctxt = _normalize_field(entry.msgctxt)
    msgid = entry.msgid
    msgid_plural = _normalize_field(entry.msgid_plural)
    msgstr = entry.msgstr or ""
    msgstr_plural = {str(k): str(v) for k, v in entry.msgstr_plural.items()}
    return PoUnit(
        msgctxt=msgctxt,
        msgid=msgid,
        msgid_plural=msgid_plural,
        msgstr=msgstr,
        msgstr_plural=msgstr_plural,
        source_key=source_key_for(msgctxt, msgid, msgid_plural),
        source_text=source_text_v1(msgctxt, msgid, msgid_plural),
    )


def _load_po_from_bytes(data: bytes) -> polib.POFile:
    with tempfile.NamedTemporaryFile(suffix=".po", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return polib.pofile(tmp_path)
    finally:
        os.unlink(tmp_path)


def parse_po_bytes(data: bytes) -> list[PoUnit]:
    po_file = _load_po_from_bytes(data)
    return [_po_entry_to_unit(entry) for entry in _iter_translation_entries(po_file)]


def parse_po_path(path: Path) -> list[PoUnit]:
    po_file = polib.pofile(str(path))
    return [_po_entry_to_unit(entry) for entry in _iter_translation_entries(po_file)]
