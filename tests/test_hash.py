from __future__ import annotations

from kdeai import hash as kdehash


def test_plural_hash_deterministic_order() -> None:
    source_key = kdehash.source_key("ctx", "id", "plural")
    msgstr_plural_a = {"1": "one", "0": "zero"}
    msgstr_plural_b = {"0": "zero", "1": "one"}
    assert (
        kdehash.translation_hash(source_key, "de", "hello", msgstr_plural_a)
        == kdehash.translation_hash(source_key, "de", "hello", msgstr_plural_b)
    )
    assert (
        kdehash.state_hash(source_key, "de", "hello", msgstr_plural_a, [], [])
        == kdehash.state_hash(source_key, "de", "hello", msgstr_plural_b, [], [])
    )
