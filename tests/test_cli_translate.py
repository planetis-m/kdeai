from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import polib
from typer.testing import CliRunner

from conftest import build_config_dict
from kdeai.cli import app
from kdeai import llm as kdellm
from kdeai import plan as kdeplan
import kdeai.cli as kdecli


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = PROJECT_ROOT / "tests" / "playground2" / "katefiletree.po"


def _write_config(
    root: Path,
    *,
    embedding_dim: int = 384,
    min_review_status: str = "reviewed",
) -> None:
    config = build_config_dict(
        {
            "languages": {"source": "en", "targets": ["el"]},
            "prompt": {
                "examples": {
                    "embedding_policy": {"model_id": "test-model", "dim": embedding_dim},
                    "eligibility": {
                        "min_review_status": min_review_status,
                        "allow_ai_generated": False,
                    },
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


def _copy_vector(root: Path) -> Path:
    candidates = [
        PROJECT_ROOT / "tests" / "vector.so",
        PROJECT_ROOT / "vector.so",
    ]
    for candidate in candidates:
        if candidate.exists():
            dest = root / "vector.so"
            shutil.copy(candidate, dest)
            return dest
    raise AssertionError("vector.so test fixture missing")


def _count_untranslated(po_path: Path) -> int:
    po_file = polib.pofile(str(po_path))
    count = 0
    for entry in po_file:
        if entry.obsolete or entry.msgid == "":
            continue
        if entry.msgid_plural:
            if not any(str(value).strip() for value in entry.msgstr_plural.values()):
                count += 1
        elif (entry.msgstr or "").strip() == "":
            count += 1
    return count


def _nplurals_from_file(po_path: Path) -> int:
    po_file = polib.pofile(str(po_path))
    plural_forms = po_file.metadata.get("Plural-Forms")
    if not plural_forms:
        return 2
    match = re.search(r"nplurals\s*=\s*(\d+)", plural_forms)
    if not match:
        return 2
    return int(match.group(1))


def test_translate_skips_nonempty_entries(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path)
    po_path = _copy_fixture(tmp_path)

    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0

    untranslated = _count_untranslated(po_path)
    assert untranslated == 8

    po_file = polib.pofile(str(po_path))
    initial_nonempty: dict[tuple[str, str, str], tuple[str, dict[str, str]]] = {}
    for entry in po_file:
        if entry.obsolete or entry.msgid == "":
            continue
        if entry.msgid_plural:
            has_translation = any(str(value).strip() for value in entry.msgstr_plural.values())
        else:
            has_translation = (entry.msgstr or "").strip() != ""
        if has_translation:
            key = (entry.msgctxt or "", entry.msgid, entry.msgid_plural or "")
            initial_nonempty[key] = (entry.msgstr or "", dict(entry.msgstr_plural))

    def _fake_translate(entries, _config, **_kwargs):
        nplurals = _nplurals_from_file(po_path)
        for entry in entries:
            if entry.get("action") != "llm":
                continue
            msgid_plural = str(entry.get("msgid_plural", ""))
            if msgid_plural:
                entry["translation"] = {
                    "msgstr": "",
                    "msgstr_plural": {str(idx): f"el-{idx}" for idx in range(nplurals)},
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
            "off",
            "--examples",
            "off",
            "--glossary",
            "off",
        ],
    )
    assert result.exit_code == 0

    updated = polib.pofile(str(po_path))
    for entry in updated:
        if entry.obsolete or entry.msgid == "":
            continue
        key = (entry.msgctxt or "", entry.msgid, entry.msgid_plural or "")
        if key in initial_nonempty:
            before_msgstr, before_plural = initial_nonempty[key]
            assert (entry.msgstr or "") == before_msgstr
            assert dict(entry.msgstr_plural) == before_plural
        else:
            if entry.msgid_plural:
                assert any(str(value).strip() for value in entry.msgstr_plural.values())
            else:
                assert (entry.msgstr or "").strip() != ""


def test_plan_apply_strict_skips_when_file_changes(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path)
    po_path = _copy_fixture(tmp_path)

    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0

    def _fake_translate(entries, _config, **_kwargs):
        for entry in entries:
            if entry.get("action") != "llm":
                continue
            entry["translation"] = {"msgstr": "el-text", "msgstr_plural": {}}
        return entries

    monkeypatch.setattr(kdellm, "batch_translate", _fake_translate)

    plan_path = tmp_path / "plan.json"
    result = runner.invoke(
        app,
        [
            "plan",
            "tests/playground2",
            "--lang",
            "el",
            "--out",
            str(plan_path),
            "--cache",
            "off",
            "--examples",
            "off",
            "--glossary",
            "off",
        ],
    )
    assert result.exit_code == 0

    po_path.write_text(po_path.read_text(encoding="utf-8") + "\n# touched\n", encoding="utf-8")

    result = runner.invoke(app, ["apply", str(plan_path)])
    assert result.exit_code == 0
    assert "skipped" in result.stdout.lower()

    assert _count_untranslated(po_path) == 8


def test_translate_adds_examples_to_prompt(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, embedding_dim=3, min_review_status="draft")
    po_path = _copy_fixture(tmp_path)
    _copy_vector(tmp_path)

    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0

    result = runner.invoke(app, ["index", "tests/playground2"])
    assert result.exit_code == 0

    def _fake_embedder(texts):
        embeddings = []
        for text in texts:
            seed = sum(ord(ch) for ch in text) % 7
            embeddings.append([float(seed), float(seed + 1), float(seed + 2)])
        return embeddings

    monkeypatch.setattr(kdeplan, "require_embedder", lambda policy: _fake_embedder)

    result = runner.invoke(app, ["examples", "build", "--from", "workspace", "--lang", "el"])
    assert result.exit_code == 0

    expected_llm = _count_untranslated(po_path)
    assert expected_llm > 0

    def _fake_translate(entries, _config, **_kwargs):
        llm_entries = 0
        examples_seen = 0
        nplurals = _nplurals_from_file(po_path)
        for entry in entries:
            if entry.get("action") != "llm":
                continue
            llm_entries += 1
            prompt = kdellm.build_prompt_payload(
                entry,
                target_lang=str(_kwargs.get("target_lang", "")),
            )
            few_shot = prompt.get("few_shot_examples", "")
            if isinstance(few_shot, str) and "1. Source:" in few_shot:
                examples_seen += 1
            msgid_plural = str(entry.get("msgid_plural", ""))
            if msgid_plural:
                entry["translation"] = {
                    "msgstr": "",
                    "msgstr_plural": {str(idx): f"el-{idx}" for idx in range(nplurals)},
                }
            else:
                entry["translation"] = {"msgstr": "el-text", "msgstr_plural": {}}
            entry["action"] = "llm"
        if llm_entries:
            assert llm_entries == expected_llm
            assert examples_seen == llm_entries
        return entries

    monkeypatch.setattr(kdellm, "batch_translate", _fake_translate)

    result = runner.invoke(app, ["translate", "tests/playground2", "--lang", "el"])
    assert result.exit_code == 0
