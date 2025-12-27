from __future__ import annotations

from kdeai.po_utils import parse_msgstr_plural, parse_nplurals


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


def test_parse_nplurals_with_valid_header() -> None:
    assert parse_nplurals("nplurals=1; plural=0;") == 1
    assert parse_nplurals("nplurals=3; plural=...;") == 3


def test_parse_nplurals_with_invalid_header() -> None:
    assert parse_nplurals("plural=0;") is None
    assert parse_nplurals(None) is None
