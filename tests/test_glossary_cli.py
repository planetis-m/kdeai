from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import spacy
from typer.testing import CliRunner

from conftest import build_config_dict
from kdeai import glossary as kdeglo
from kdeai import llm as kdellm
from kdeai.cli import app


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = PROJECT_ROOT / "tests" / "playground2" / "katefiletree.po"


def _write_config(root: Path) -> None:
    config = build_config_dict(
        {
            "languages": {"source": "en", "targets": ["el"]},
            "prompt": {
                "examples": {
                    "embedding_policy": {"model_id": "test-model", "dim": 384},
                },
                "glossary": {
                    "spacy_model": "en_core_web_sm",
                    "normalization_id": kdeglo.NORMALIZATION_ID,
                },
            },
        }
    )
    config_path = root / ".kdeai" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")


def _copy_fixture(root: Path) -> Path:
    dest = root / "tests" / "playground2"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURE, dest / "katefiletree.po")
    return dest / "katefiletree.po"


def _load_glossary_terms(root: Path) -> list[kdeglo.GlossaryTerm]:
    pointer_path = root / ".kdeai" / "cache" / "glossary" / "glossary.current.json"
    payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    db_file = str(payload.get("db_file", ""))
    glossary_path = pointer_path.parent / db_file
    conn = sqlite3.connect(str(glossary_path))
    try:
        return kdeglo.load_terms(conn, src_lang="en", tgt_lang="el")
    finally:
        conn.close()


def _write_glossary_output(terms: list[kdeglo.GlossaryTerm], output_path: Path) -> None:
    lines = []
    for term in sorted(terms, key=lambda item: item.src_surface.casefold()):
        lines.append(f"{term.src_surface} -> {term.tgt_primary}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_glossary_build_from_playground2(monkeypatch, tmp_path: Path) -> None:
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        import pytest

        pytest.skip("en_core_web_sm not installed; run: python -m spacy download en_core_web_sm")

    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path)
    _copy_fixture(tmp_path)

    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0

    result = runner.invoke(app, ["reference", "build", "tests/playground2"])
    assert result.exit_code == 0

    result = runner.invoke(app, ["glossary", "build"])
    assert result.exit_code == 0

    terms = _load_glossary_terms(tmp_path)
    term_by_surface = {term.src_surface: term for term in terms}

    assert "Document Name" in term_by_surface
    assert term_by_surface["Document Name"].tgt_primary == "Όνομα εγγράφου"

    glossary_output = tmp_path / "glossary.txt"
    _write_glossary_output(terms, glossary_output)
    print("Glossary terms (src -> tgt):")
    print(glossary_output.read_text(encoding="utf-8"))


def test_translate_includes_glossary_in_prompt(monkeypatch, tmp_path: Path) -> None:
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        import pytest

        pytest.skip("en_core_web_sm not installed; run: python -m spacy download en_core_web_sm")

    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path)
    _copy_fixture(tmp_path)

    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0

    result = runner.invoke(app, ["reference", "build", "tests/playground2"])
    assert result.exit_code == 0

    result = runner.invoke(app, ["glossary", "build"])
    assert result.exit_code == 0

    captured: dict[str, object] = {}

    def _fake_translate(entries, _config, **_kwargs):
        for entry in entries:
            if entry.get("action") != "llm":
                continue
            prompt = entry.get("prompt") or {}
            if not prompt:
                prompt = {
                    "few_shot_examples": entry.get("examples", ""),
                    "glossary_context": entry.get("glossary_terms", ""),
                }
                glossary_context = str(prompt.get("glossary_context", ""))
                if glossary_context:
                    formatted_terms = [
                        f"- {term.strip()}"
                        for term in glossary_context.split(",")
                        if term.strip()
                    ]
                    glossary_block = "\n".join(formatted_terms)
                    prompt["messages"] = [
                        {"role": "user", "content": f"Glossary:\n{glossary_block}"}
                    ]
                else:
                    prompt["messages"] = []
                entry["prompt"] = prompt
            if entry.get("msgid") == "Middle Click To Close Documents":
                captured["prompt"] = prompt
            msgid_plural = str(entry.get("msgid_plural", ""))
            if msgid_plural:
                entry["translation"] = {
                    "msgstr": "",
                    "msgstr_plural": {"0": "el-0", "1": "el-1"},
                }
            else:
                entry["translation"] = {"msgstr": "el-text", "msgstr_plural": {}}
            entry["action"] = "llm"
        return entries

    monkeypatch.setattr(kdellm, "batch_translate", _fake_translate)

    result = runner.invoke(
        app,
        [
            "translate",
            "tests/playground2",
            "--lang",
            "el",
            "--cache",
            "on",
            "--examples",
            "off",
            "--glossary",
            "required",
        ],
    )
    assert result.exit_code == 0

    prompt = captured.get("prompt")
    assert isinstance(prompt, dict)
    glossary_context = str(prompt.get("glossary_context", ""))
    assert "Close -> Κλείσιμο" in glossary_context
    assert "Documents -> Έγγραφα" in glossary_context

    user_prompt = ""
    for message in prompt.get("messages", []):
        if message.get("role") == "user":
            user_prompt = str(message.get("content", ""))
            break
    assert "Glossary:" in user_prompt
    assert "- Close -> Κλείσιμο" in user_prompt
    assert "- Documents -> Έγγραφα" in user_prompt
