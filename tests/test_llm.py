from __future__ import annotations

import os
import warnings
from pathlib import Path
import types

from conftest import build_config
from kdeai.llm import (
    _translation_payload,
    batch_translate,
    build_prompt_payload as llm_build_prompt_payload,
)
from kdeai.prompt import build_prompt_payload as prompt_build_prompt_payload


class _Example:
    def __init__(self, source_text: str, msgstr: str) -> None:
        self.source_text = source_text
        self.msgstr = msgstr
        self.msgstr_plural = {}


class _Term:
    def __init__(self, src_surface: str, tgt_primary: str) -> None:
        self.src_surface = src_surface
        self.tgt_primary = tgt_primary
        self.tgt_alternates = []


class _GlossaryMatch:
    def __init__(self, term: _Term) -> None:
        self.term = term


def test_build_prompt_payload_includes_structured_fields() -> None:
    payload = prompt_build_prompt_payload(
        config=build_config({"languages": {"source": "en"}}),
        msgctxt=None,
        msgid="File",
        msgid_plural=None,
        target_lang="de",
        examples=[_Example("ctx:\nid:File\npl:", "Datei")],
        glossary=[_GlossaryMatch(_Term("File", "Datei"))],
    )

    assert payload["source_text_v1"] == "ctx:\nid:File\npl:"
    assert payload["target_lang"] == "de"
    assert payload["glossary_context"] == "File -> Datei"
    assert "1. Source:" in payload["few_shot_examples"]
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"


def test_llm_build_prompt_payload_uses_source_text_v1() -> None:
    payload = llm_build_prompt_payload(
        {
            "msgctxt": "",
            "msgid": "Save",
            "msgid_plural": "",
            "examples": [],
            "glossary_terms": [],
        },
        target_lang="de",
    )
    assert payload["source_text_v1"] == "ctx:\nid:Save\npl:"


def test_translation_payload_fills_plural_forms() -> None:
    payload = _translation_payload(
        msgid_plural="Files",
        translated_text="Datei",
        translated_plural="Dateien",
        plural_forms="nplurals=3; plural=...;",
    )

    assert payload["msgstr"] == ""
    assert payload["msgstr_plural"]["0"] == "Datei"
    assert payload["msgstr_plural"]["1"] == "Dateien"
    assert payload["msgstr_plural"]["2"] == "Dateien"


def _load_env_if_missing(keys: list[str]) -> None:
    if any(os.getenv(key) for key in keys):
        return
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        name = name.strip()
        if name and name not in os.environ:
            os.environ[name] = value.strip()


def test_batch_translate_updates_needs_llm_entries() -> None:
    _load_env_if_missing(["OPENROUTER_API_KEY"])
    if not os.getenv("OPENROUTER_API_KEY"):
        raise AssertionError("OPENROUTER_API_KEY must be set for DSPy usage")
    warnings.filterwarnings(
        "ignore",
        message="Pydantic serializer warnings:*",
        category=UserWarning,
        module="pydantic",
    )

    plan = {
        "lang": "de",
        "files": [
            {
                "file_path": "locale/de.po",
                "entries": [
                        {
                            "msgctxt": "",
                            "msgid": "File",
                            "msgid_plural": "",
                            "action": "llm",
                        },
                        {
                            "msgctxt": "",
                            "msgid": "Files",
                            "msgid_plural": "Files",
                            "action": "llm",
                        },
                    {
                        "msgctxt": "",
                        "msgid": "Skip",
                        "msgid_plural": "",
                        "action": "copy_tm",
                    },
                ],
            }
        ],
    }

    config = build_config(
        {
            "prompt": {"generation_model_id": "openrouter/x-ai/grok-4-fast"},
            "apply": {
                "tagging": {
                    "llm": {
                        "add_flags": ["fuzzy"],
                        "add_ai_flag": True,
                        "comment_prefix_key": "ai",
                    }
                }
            },
        }
    )

    from kdeai import llm as kdellm
    original_forward = kdellm.KDEAITranslator.forward
    try:
        def _fake_forward(self, _prompt):
            return types.SimpleNamespace(translated_text="Datei", translated_plural="Dateien")

        kdellm.KDEAITranslator.forward = _fake_forward
        batch_translate(plan["files"][0]["entries"], config, target_lang=plan["lang"])
    finally:
        kdellm.KDEAITranslator.forward = original_forward

    entries = plan["files"][0]["entries"]
    singular = entries[0]
    plural = entries[1]
    skipped = entries[2]

    assert singular["translation"]["msgstr"].strip()
    assert singular["translation"]["msgstr_plural"] == {}
    assert plural["translation"]["msgstr"] == ""
    assert plural["translation"]["msgstr_plural"]["0"].strip()
    assert plural["translation"]["msgstr_plural"]["1"].strip()
    assert "translation" not in skipped

    assert singular["tag_profile"] == "llm"
    assert plural["tag_profile"] == "llm"


def test_batch_translate_adds_tags_for_needs_llm() -> None:
    _load_env_if_missing(["OPENROUTER_API_KEY"])
    if not os.getenv("OPENROUTER_API_KEY"):
        raise AssertionError("OPENROUTER_API_KEY must be set for DSPy usage")
    warnings.filterwarnings(
        "ignore",
        message="Pydantic serializer warnings:*",
        category=UserWarning,
        module="pydantic",
    )

    plan = {
        "lang": "de",
        "files": [
            {
                "file_path": "locale/de.po",
                "entries": [
                        {
                            "msgctxt": "",
                    "msgid": "Save",
                    "msgid_plural": "",
                    "action": "llm",
                    "examples": [{"source_text": "ctx:\nid:Save\npl:", "msgstr": "Speichern", "msgstr_plural": {}}],
                    "glossary_terms": [{"src_surface": "Save", "tgt_primary": "Speichern", "tgt_alternates": []}],
                }
                    ],
                }
            ],
        }

    config = build_config(
        {
            "prompt": {"generation_model_id": "openrouter/x-ai/grok-4-fast"},
            "apply": {
                "tagging": {
                    "llm": {
                        "add_flags": ["fuzzy"],
                        "add_ai_flag": True,
                        "comment_prefix_key": "ai",
                    }
                }
            },
        }
    )

    from kdeai import llm as kdellm
    original_forward = kdellm.KDEAITranslator.forward
    try:
        def _fake_forward(self, _prompt):
            return types.SimpleNamespace(translated_text="Speichern", translated_plural="")

        kdellm.KDEAITranslator.forward = _fake_forward
        batch_translate(plan["files"][0]["entries"], config, target_lang=plan["lang"])
    finally:
        kdellm.KDEAITranslator.forward = original_forward

    entry = plan["files"][0]["entries"][0]
    assert entry["action"] == "llm"
    assert entry["translation"]["msgstr"].strip()
    assert entry["translation"]["msgstr_plural"] == {}
    assert entry["tag_profile"] == "llm"
    assert "prompt" not in entry
