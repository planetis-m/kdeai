from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import polib
from typer.testing import CliRunner

from kdeai.cli import app
from kdeai import llm as kdellm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = PROJECT_ROOT / "tests" / "playground2" / "katefiletree.po"


def _write_config(root: Path) -> None:
    config = {
        "format": 2,
        "languages": {"source": "en", "targets": ["el"]},
        "markers": {
            "ai_flag": "kdeai-ai",
            "comment_prefixes": {
                "tool": "KDEAI:",
                "ai": "KDEAI-AI:",
                "tm": "KDEAI-TM:",
                "review": "KDEAI-REVIEW:",
            },
        },
        "tm": {
            "lookup_scopes": ["session", "workspace", "reference"],
            "selection": {
                "review_status_order": ["reviewed", "draft", "needs_review", "unreviewed"],
                "prefer_human": True,
            },
        },
        "prompt": {
            "examples": {
                "mode_default": "auto",
                "lookup_scopes": ["workspace", "reference"],
                "top_n": 6,
                "embedding_policy": {
                    "model_id": "test-model",
                    "dim": 384,
                    "distance": "cosine",
                    "encoding": "float32_le",
                    "normalization": "none",
                    "input_canonicalization": "source_text_v1",
                    "require_finite": True,
                },
                "eligibility": {
                    "min_review_status": "reviewed",
                    "allow_ai_generated": False,
                },
            },
            "glossary": {
                "mode_default": "auto",
                "lookup_scopes": ["reference"],
                "spacy_model": "en_core_web_sm",
                "normalization_id": "kdeai_glossary_norm_v1",
                "max_terms": 10,
            },
        },
        "apply": {
            "mode_default": "strict",
            "overwrite_default": "conservative",
            "tagging": {
                "tm_copy": {"add_flags": ["fuzzy"], "add_ai_flag": False, "comment_prefix_key": "tm"},
                "llm": {"add_flags": ["fuzzy"], "add_ai_flag": True, "comment_prefix_key": "ai"},
            },
        },
        "sqlite": {"workspace_tm": {"synchronous": "normal", "busy_timeout_ms": {"read": 5000, "write": 50}}},
    }
    config_path = root / ".kdeai" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")


def _copy_fixture(root: Path) -> Path:
    dest = root / "tests" / "playground2"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURE, dest / "katefiletree.po")
    return dest / "katefiletree.po"


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
    assert untranslated == 2

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

    def _fake_translate(plan, config):
        for file_item in plan.get("files", []):
            file_path = Path(file_item.get("file_path", ""))
            if not str(file_path):
                continue
            file_path = Path.cwd() / file_path
            nplurals = _nplurals_from_file(file_path)
            for entry in file_item.get("entries", []):
                if entry.get("action") != "llm":
                    continue
                msgid_plural = str(entry.get("msgid_plural", ""))
                if msgid_plural:
                    entry["translation"] = {
                        "msgstr": "",
                        "msgstr_plural": {
                            str(idx): f"el-{idx}" for idx in range(nplurals)
                        },
                    }
                else:
                    entry["translation"] = {"msgstr": "el-text", "msgstr_plural": {}}
        return plan

    monkeypatch.setattr(kdellm, "batch_translate_plan", _fake_translate)

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

    assert _count_untranslated(po_path) == 2
