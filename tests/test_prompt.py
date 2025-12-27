from __future__ import annotations

from kdeai import prompt as kdeprompt
from kdeai.examples import ExampleMatch


def test_examples_context_omits_distance() -> None:
    examples_a = [
        ExampleMatch(
            source_key="sk",
            source_text="hello",
            lang="de",
            msgstr="",
            msgstr_plural={"1": "eins", "0": "null"},
            review_status="reviewed",
            is_ai_generated=0,
            translation_hash="th",
            file_path="f.po",
            file_sha256="fh",
            distance=0.123456789,
        )
    ]
    examples_b = [
        ExampleMatch(
            source_key="sk",
            source_text="hello",
            lang="de",
            msgstr="",
            msgstr_plural={"0": "null", "1": "eins"},
            review_status="reviewed",
            is_ai_generated=0,
            translation_hash="th",
            file_path="f.po",
            file_sha256="fh",
            distance=0.987654321,
        )
    ]
    assert kdeprompt.examples_context(examples_a) == kdeprompt.examples_context(examples_b)
    assert "0.123456789" not in kdeprompt.examples_context(examples_a)
    assert "0.987654321" not in kdeprompt.examples_context(examples_b)
