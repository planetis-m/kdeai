from __future__ import annotations

from kdeai.po_utils import parse_msgstr_plural


def test_parse_msgstr_plural_with_dict() -> None:
    assert parse_msgstr_plural({"0": "eins", 1: "zwei"}) == {"0": "eins", "1": "zwei"}


def test_parse_msgstr_plural_with_json_dict() -> None:
    assert parse_msgstr_plural('{"0":"eins","1":"zwei"}') == {
        "0": "eins",
        "1": "zwei",
    }


def test_parse_msgstr_plural_with_invalid_json() -> None:
    assert parse_msgstr_plural("{") == {}


def test_parse_msgstr_plural_with_non_dict_json() -> None:
    assert parse_msgstr_plural('["eins"]') == {}
