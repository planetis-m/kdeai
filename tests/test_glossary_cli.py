from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import spacy
from typer.testing import CliRunner

from conftest import build_config_dict
from kdeai import db as kdedb
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


def _read_pointer(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_glossary_meta(root: Path) -> dict[str, str]:
    pointer_path = root / ".kdeai" / "cache" / "glossary" / "glossary.current.json"
    payload = _read_pointer(pointer_path)
    db_file = str(payload.get("db_file", ""))
    glossary_path = pointer_path.parent / db_file
    conn = sqlite3.connect(str(glossary_path))
    try:
        return kdedb.read_meta(conn)
    finally:
        conn.close()


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
            prompt = kdellm.build_prompt_payload(
                entry,
                target_lang=str(_kwargs.get("target_lang", "")),
            )
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


def test_glossary_build_creates_new_generation(monkeypatch, tmp_path: Path) -> None:
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

    pointer_path = tmp_path / ".kdeai" / "cache" / "glossary" / "glossary.current.json"
    first_pointer = _read_pointer(pointer_path)
    first_snapshot_id = int(first_pointer.get("snapshot_id", 0))

    result = runner.invoke(app, ["glossary", "build"])
    assert result.exit_code == 0

    reference_pointer = _read_pointer(
        tmp_path / ".kdeai" / "cache" / "reference" / "reference.current.json"
    )
    reference_snapshot_id = int(reference_pointer.get("snapshot_id", 0))

    second_pointer = _read_pointer(pointer_path)
    second_snapshot_id = int(second_pointer.get("snapshot_id", 0))
    assert second_snapshot_id == first_snapshot_id + 1
    assert int(second_pointer.get("source_snapshot", {}).get("snapshot_id", 0)) == reference_snapshot_id

    meta = _load_glossary_meta(tmp_path)
    assert meta.get("snapshot_id") == str(second_snapshot_id)
    assert meta.get("source_snapshot_id") == str(reference_snapshot_id)
