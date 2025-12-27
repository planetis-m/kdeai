import hashlib
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import contextlib

import polib

from conftest import build_config
from kdeai import apply
from kdeai import locks
from kdeai import plan


def _entry_state_hash(entry, *, lang: str, config) -> str:
    compiled_placeholder_patterns = [
        re.compile(pattern) for pattern in config.apply.validation_patterns
    ]
    ctx = apply.ApplyContext.from_config(
        config,
        lang=lang,
        mode="strict",
        overwrite_policy="conservative",
        placeholder_patterns=compiled_placeholder_patterns,
    )
    return apply.entry_state_hash(
        entry,
        lang=lang,
        marker_flags=ctx.marker_flags,
        comment_prefixes=ctx.comment_prefixes,
    )


class TestApplyStrictMode(unittest.TestCase):
    def test_apply_strict_updates_translation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            locale_dir = root / "locale"
            locale_dir.mkdir(parents=True, exist_ok=True)
            po_path = locale_dir / "de.po"

            po_text = (
                'msgid ""\n'
                'msgstr ""\n'
                '"Content-Type: text/plain; charset=UTF-8\\n"\n'
                '"Plural-Forms: nplurals=2; plural=(n != 1);\\n"\n'
                "\n"
                'msgctxt "menu"\n'
                'msgid "File"\n'
                'msgstr ""\n'
            )
            po_path.write_text(po_text, encoding="utf-8")

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            po_file = polib.pofile(str(po_path))
            entry = po_file.find("File", msgctxt="menu")
            config = build_config()
            base_state_hash = _entry_state_hash(entry, lang="de", config=config)

            config = build_config()
            plan_payload = {
                "format": 1,
                "project_id": "proj",
                "config_hash": config.config_hash,
                "lang": "de",
                "apply_defaults": {
                    "mode": "strict",
                    "overwrite": "conservative",
                    "post_index": "off",
                },
                "files": [
                    {
                        "file_path": "locale/de.po",
                        "base_sha256": base_sha256,
                        "entries": [
                            {
                                "msgctxt": "menu",
                                "msgid": "File",
                                "msgid_plural": "",
                                "base_state_hash": base_state_hash,
                                "action": "copy_tm",
                                "tag_profile": "tm_copy",
                                "translation": {"msgstr": "Datei", "msgstr_plural": {}},
                            }
                        ],
                    }
                ],
            }
            finalized_plan = plan.finalize_plan(plan_payload)
            result = apply.apply_plan(
                finalized_plan,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                config=config,
                apply_mode="strict",
                overwrite="conservative",
            )

            self.assertEqual(result.files_written, ["locale/de.po"])
            self.assertEqual(result.entries_applied, 1)

            updated = polib.pofile(str(po_path))
            updated_entry = updated.find("File", msgctxt="menu")
            self.assertEqual(updated_entry.msgstr, "Datei")

    def test_apply_strict_skips_on_entry_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            locale_dir = root / "locale"
            locale_dir.mkdir(parents=True, exist_ok=True)
            po_path = locale_dir / "de.po"

            po_text = (
                'msgid ""\n'
                'msgstr ""\n'
                '"Content-Type: text/plain; charset=UTF-8\\n"\n'
                '"Plural-Forms: nplurals=2; plural=(n != 1);\\n"\n'
                "\n"
                'msgctxt "menu"\n'
                'msgid "File"\n'
                'msgstr ""\n'
            )
            po_path.write_text(po_text, encoding="utf-8")

            po_file = polib.pofile(str(po_path))
            entry = po_file.find("File", msgctxt="menu")
            config = build_config()
            base_state_hash = _entry_state_hash(entry, lang="de", config=config)

            entry.msgstr = "Alt"
            po_file.save(str(po_path))
            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()

            config = build_config()
            plan_payload = {
                "format": 1,
                "project_id": "proj",
                "config_hash": config.config_hash,
                "lang": "de",
                "apply_defaults": {
                    "mode": "strict",
                    "overwrite": "conservative",
                    "post_index": "off",
                },
                "files": [
                    {
                        "file_path": "locale/de.po",
                        "base_sha256": base_sha256,
                        "entries": [
                            {
                                "msgctxt": "menu",
                                "msgid": "File",
                                "msgid_plural": "",
                                "base_state_hash": base_state_hash,
                                "action": "copy_tm",
                                "tag_profile": "tm_copy",
                                "translation": {"msgstr": "Datei", "msgstr_plural": {}},
                            }
                        ],
                    }
                ],
            }
            finalized_plan = plan.finalize_plan(plan_payload)
            result = apply.apply_plan(
                finalized_plan,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                config=config,
                apply_mode="strict",
                overwrite="conservative",
            )

            self.assertEqual(result.files_written, [])
            self.assertEqual(result.files_skipped, ["locale/de.po"])
            updated = polib.pofile(str(po_path))
            updated_entry = updated.find("File", msgctxt="menu")
            self.assertEqual(updated_entry.msgstr, "Alt")


class TestApplyAdditionalCases(unittest.TestCase):
    def _write_sample_po(self, po_path: Path) -> None:
        po_text = (
            'msgid ""\n'
            'msgstr ""\n'
            '"Content-Type: text/plain; charset=UTF-8\\n"\n'
            '"Plural-Forms: nplurals=2; plural=(n != 1);\\n"\n'
            "\n"
            'msgctxt "menu"\n'
            'msgid "File"\n'
            'msgstr ""\n'
            "\n"
            'msgctxt "menu"\n'
            'msgid "Edit"\n'
            'msgstr ""\n'
        )
        po_path.write_text(po_text, encoding="utf-8")

    def _build_plan(
        self,
        *,
        file_path: str,
        base_sha256: str,
        base_state_hash: str,
        msgctxt: str,
        msgid: str,
        translation: dict,
        config_hash: str,
        action: str = "copy_tm",
        tag_profile: str | None = None,
        extra: dict | None = None,
    ) -> dict:
        resolved_tag_profile = tag_profile
        if resolved_tag_profile is None:
            resolved_tag_profile = "tm_copy" if action == "copy_tm" else "llm"
        plan_payload = {
            "format": 1,
            "project_id": "proj",
            "config_hash": config_hash,
            "lang": "de",
            "apply_defaults": {
                "mode": "strict",
                "overwrite": "conservative",
                "post_index": "off",
            },
            "files": [
                {
                    "file_path": file_path,
                    "base_sha256": base_sha256,
                    "entries": [
                    {
                        "msgctxt": msgctxt,
                        "msgid": msgid,
                        "msgid_plural": "",
                        "base_state_hash": base_state_hash,
                        "action": action,
                        "tag_profile": resolved_tag_profile,
                        "translation": translation,
                    }
                ],
            }
        ],
        }
        if extra:
            plan_payload.update(extra)
        return plan.finalize_plan(plan_payload)

    def test_validate_plan_header_requires_project_id(self):
        config = build_config()
        marker_flags, comment_prefixes, _, _, ai_flag = apply._marker_settings_from_config(config)
        placeholder_patterns = list(config.apply.validation_patterns)

        missing = apply._validate_plan_header(
            {"config_hash": config.config_hash},
            project_id="proj",
            config=config,
            marker_flags=marker_flags,
            comment_prefixes=comment_prefixes,
            ai_flag=ai_flag,
            placeholder_patterns=placeholder_patterns,
        )
        self.assertEqual(missing, ["plan project_id missing"])

        mismatched = apply._validate_plan_header(
            {"project_id": "other", "config_hash": config.config_hash},
            project_id="proj",
            config=config,
            marker_flags=marker_flags,
            comment_prefixes=comment_prefixes,
            ai_flag=ai_flag,
            placeholder_patterns=placeholder_patterns,
        )
        self.assertEqual(mismatched, ["plan project_id does not match current project"])

        ok = apply._validate_plan_header(
            {"project_id": "proj", "config_hash": config.config_hash},
            project_id="proj",
            config=config,
            marker_flags=marker_flags,
            comment_prefixes=comment_prefixes,
            ai_flag=ai_flag,
            placeholder_patterns=placeholder_patterns,
        )
        self.assertEqual(ok, [])

    def test_strict_skips_on_file_hash_change(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            po_path = root / "locale" / "de.po"
            po_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_sample_po(po_path)

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            po_file = polib.pofile(str(po_path))
            entry = po_file.find("File", msgctxt="menu")
            config = build_config()
            base_state_hash = _entry_state_hash(entry, lang="de", config=config)
            config = build_config()

            plan_payload = self._build_plan(
                file_path="locale/de.po",
                base_sha256=base_sha256,
                base_state_hash=base_state_hash,
                msgctxt="menu",
                msgid="File",
                translation={"msgstr": "Datei", "msgstr_plural": {}},
                config_hash=config.config_hash,
            )

            po_path.write_text(po_path.read_text(encoding="utf-8") + "\n# touched\n", encoding="utf-8")

            result = apply.apply_plan(
                plan_payload,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                config=config,
                apply_mode="strict",
                overwrite="conservative",
            )
            self.assertEqual(result.files_written, [])
            self.assertEqual(result.files_skipped, ["locale/de.po"])

    def test_rebase_skips_mismatched_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            po_path = root / "locale" / "de.po"
            po_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_sample_po(po_path)

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            po_file = polib.pofile(str(po_path))
            entry_file = po_file.find("File", msgctxt="menu")
            entry_edit = po_file.find("Edit", msgctxt="menu")
            config = build_config()
            base_state_file = _entry_state_hash(entry_file, lang="de", config=config)
            config = build_config()
            base_state_edit = _entry_state_hash(entry_edit, lang="de", config=config)

            config = build_config()
            plan_payload = {
                "format": 1,
                "project_id": "proj",
                "config_hash": config.config_hash,
                "lang": "de",
                "apply_defaults": {"mode": "strict", "overwrite": "conservative", "post_index": "off"},
                "files": [
                    {
                        "file_path": "locale/de.po",
                        "base_sha256": base_sha256,
                        "entries": [
                            {
                                "msgctxt": "menu",
                                "msgid": "File",
                                "msgid_plural": "",
                                "base_state_hash": base_state_file,
                                "action": "copy_tm",
                                "tag_profile": "tm_copy",
                                "translation": {"msgstr": "Datei", "msgstr_plural": {}},
                            },
                            {
                                "msgctxt": "menu",
                                "msgid": "Edit",
                                "msgid_plural": "",
                                "base_state_hash": base_state_edit,
                                "action": "copy_tm",
                                "tag_profile": "tm_copy",
                                "translation": {"msgstr": "Bearbeiten", "msgstr_plural": {}},
                            },
                        ],
                    }
                ],
            }
            plan_payload = plan.finalize_plan(plan_payload)

            po_file.find("Edit", msgctxt="menu").msgstr = "Alt"
            po_file.save(str(po_path))

            result = apply.apply_plan(
                plan_payload,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                config=config,
                apply_mode="rebase",
                overwrite="conservative",
            )

            updated = polib.pofile(str(po_path))
            self.assertEqual(updated.find("File", msgctxt="menu").msgstr, "Datei")
            self.assertEqual(updated.find("Edit", msgctxt="menu").msgstr, "Alt")
            self.assertEqual(result.entries_applied, 1)

    def test_placeholder_validation_blocks_apply(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            po_path = root / "locale" / "de.po"
            po_path.parent.mkdir(parents=True, exist_ok=True)
            po_text = (
                'msgid ""\n'
                'msgstr ""\n'
                '"Content-Type: text/plain; charset=UTF-8\\n"\n'
                '"Plural-Forms: nplurals=2; plural=(n != 1);\\n"\n'
                "\n"
                'msgctxt "menu"\n'
                'msgid "Hello %s"\n'
                'msgstr ""\n'
            )
            po_path.write_text(po_text, encoding="utf-8")

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            po_file = polib.pofile(str(po_path))
            entry = po_file.find("Hello %s", msgctxt="menu")
            config = build_config()
            base_state_hash = _entry_state_hash(entry, lang="de", config=config)
            config = build_config({"apply": {"validation_patterns": [r"%\w"]}})

            plan_payload = self._build_plan(
                file_path="locale/de.po",
                base_sha256=base_sha256,
                base_state_hash=base_state_hash,
                msgctxt="menu",
                msgid="Hello %s",
                translation={"msgstr": "Hallo", "msgstr_plural": {}},
                config_hash=config.config_hash,
            )

            result = apply.apply_plan(
                plan_payload,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                config=config,
                apply_mode="strict",
                overwrite="conservative",
            )
            self.assertEqual(result.files_written, [])
            self.assertTrue(result.errors)
            updated = polib.pofile(str(po_path))
            updated_entry = updated.find("Hello %s", msgctxt="menu")
            self.assertEqual(updated_entry.msgstr, "")

    def test_placeholder_patterns_mismatch_blocks_apply(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            po_path = root / "locale" / "de.po"
            po_path.parent.mkdir(parents=True, exist_ok=True)
            po_text = (
                'msgid ""\n'
                'msgstr ""\n'
                '"Content-Type: text/plain; charset=UTF-8\\n"\n'
                '"Plural-Forms: nplurals=2; plural=(n != 1);\\n"\n'
                "\n"
                'msgctxt "menu"\n'
                'msgid "Hello %s"\n'
                'msgstr ""\n'
            )
            po_path.write_text(po_text, encoding="utf-8")

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            po_file = polib.pofile(str(po_path))
            entry = po_file.find("Hello %s", msgctxt="menu")
            config = build_config()
            base_state_hash = _entry_state_hash(entry, lang="de", config=config)
            config = build_config({"apply": {"validation_patterns": [r"%\w"]}})

            plan_payload = self._build_plan(
                file_path="locale/de.po",
                base_sha256=base_sha256,
                base_state_hash=base_state_hash,
                msgctxt="menu",
                msgid="Hello %s",
                translation={"msgstr": "Hallo %s", "msgstr_plural": {}},
                config_hash=config.config_hash,
                extra={"placeholder_patterns": []},
            )

            result = apply.apply_plan(
                plan_payload,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                config=config,
                apply_mode="strict",
                overwrite="conservative",
            )
            self.assertEqual(result.files_written, [])
            self.assertIn("plan placeholder_patterns do not match config", result.errors)
            updated = polib.pofile(str(po_path))
            updated_entry = updated.find("Hello %s", msgctxt="menu")
            self.assertEqual(updated_entry.msgstr, "")

    def test_apply_comments_removes_ensures_appends(self):
        entry = polib.POEntry(msgid="File", msgstr="")
        entry.tcomment = (
            "KDEAI: old\r\n"
            "Note line\n"
            "KDEAI-TM: remove me\n"
            "KDEAI: keep\n"
            "KDEAI: ensure-me\n"
            "Other line\n"
        )
        remove_prefixes = ["KDEAI-TM:"]
        ensure_lines = ["KDEAI: ensure-me", "KDEAI: ensure-new"]
        append = "KDEAI-AI: model=x\nKDEAI: append2\n"

        apply._apply_comments(entry, remove_prefixes, ensure_lines, append)

        expected = "\n".join(
            [
                "KDEAI: old",
                "Note line",
                "KDEAI: keep",
                "Other line",
                "KDEAI: ensure-me",
                "KDEAI: ensure-new",
                "KDEAI-AI: model=x",
                "KDEAI: append2",
                "",
            ]
        )
        self.assertEqual(entry.tcomment, expected)

    def test_apply_flags_removes_ai_flag(self):
        entry = polib.POEntry(msgid="File", msgstr="")
        entry.flags = ["fuzzy", "kdeai-ai", "keep"]

        apply._apply_flags(entry, add_flags=["fuzzy"], remove_flags=["kdeai-ai"])

        self.assertEqual(entry.flags, ["fuzzy", "keep"])

    def test_apply_comments_removes_tool_prefixes_only(self):
        entry = polib.POEntry(msgid="File", msgstr="")
        entry.tcomment = (
            "KDEAI-AI: model=old\n"
            "Note line\n"
            "KDEAI-TM: copied\n"
            "KDEAI-REVIEW: ok\n"
            "KDEAI: keep\n"
        )
        remove_prefixes = ["KDEAI-AI:", "KDEAI-TM:"]
        ensure_lines = ["KDEAI-AI: model=new"]
        append = ""

        apply._apply_comments(entry, remove_prefixes, ensure_lines, append)

        expected = "\n".join(
            [
                "Note line",
                "KDEAI-REVIEW: ok",
                "KDEAI: keep",
                "KDEAI-AI: model=new",
                "",
            ]
        )
        self.assertEqual(entry.tcomment, expected)

    def test_plan_rejects_comments_field(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            po_path = root / "locale" / "de.po"
            po_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_sample_po(po_path)

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            po_file = polib.pofile(str(po_path))
            entry = po_file.find("File", msgctxt="menu")
            config = build_config()
            base_state_hash = _entry_state_hash(entry, lang="de", config=config)
            config = build_config()

            plan_payload = self._build_plan(
                file_path="locale/de.po",
                base_sha256=base_sha256,
                base_state_hash=base_state_hash,
                msgctxt="menu",
                msgid="File",
                translation={"msgstr": "Datei", "msgstr_plural": {}},
                config_hash=config.config_hash,
            )
            plan_payload["files"][0]["entries"][0]["comments"] = {
                "remove_prefixes": ["#"],
                "ensure_lines": [],
                "append": "",
            }

            result = apply.apply_plan(
                plan_payload,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                config=config,
                apply_mode="strict",
                overwrite="conservative",
            )
            self.assertEqual(result.files_written, [])
            self.assertTrue(
                any("plan entries must not include comments" in error for error in result.errors)
            )
            updated = polib.pofile(str(po_path))
            updated_entry = updated.find("File", msgctxt="menu")
            self.assertEqual(updated_entry.msgstr, "")

    def test_plan_rejects_flags_field(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            po_path = root / "locale" / "de.po"
            po_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_sample_po(po_path)

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            po_file = polib.pofile(str(po_path))
            entry = po_file.find("File", msgctxt="menu")
            config = build_config()
            base_state_hash = _entry_state_hash(entry, lang="de", config=config)
            config = build_config()

            plan_payload = self._build_plan(
                file_path="locale/de.po",
                base_sha256=base_sha256,
                base_state_hash=base_state_hash,
                msgctxt="menu",
                msgid="File",
                translation={"msgstr": "Datei", "msgstr_plural": {}},
                config_hash=config.config_hash,
            )
            plan_payload["files"][0]["entries"][0]["flags"] = {
                "add": ["fuzzy"],
                "remove": [],
            }

            result = apply.apply_plan(
                plan_payload,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                config=config,
                apply_mode="strict",
                overwrite="conservative",
            )
            self.assertEqual(result.files_written, [])
            self.assertTrue(
                any("plan entries must not include flags" in error for error in result.errors)
            )
            updated = polib.pofile(str(po_path))
            updated_entry = updated.find("File", msgctxt="menu")
            self.assertEqual(updated_entry.msgstr, "")

    def test_apply_derives_tm_tag_comment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            po_path = root / "locale" / "de.po"
            po_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_sample_po(po_path)

            po_file = polib.pofile(str(po_path))
            entry = po_file.find("File", msgctxt="menu")
            entry.tcomment = (
                "KDEAI-TM: copied_from=old\n"
                "KDEAI: keep"
            )
            po_file.save(str(po_path))

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            config = build_config()
            base_state_hash = _entry_state_hash(entry, lang="de", config=config)
            config = build_config()

            plan_payload = self._build_plan(
                file_path="locale/de.po",
                base_sha256=base_sha256,
                base_state_hash=base_state_hash,
                msgctxt="menu",
                msgid="File",
                translation={"msgstr": "Datei", "msgstr_plural": {}},
                config_hash=config.config_hash,
            )
            plan_payload["files"][0]["entries"][0]["tm_scope"] = "workspace"

            result = apply.apply_plan(
                plan_payload,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                config=config,
                apply_mode="strict",
                overwrite="conservative",
            )
            self.assertEqual(result.files_written, ["locale/de.po"])
            updated = polib.pofile(str(po_path))
            updated_entry = updated.find("File", msgctxt="menu")
            self.assertEqual(
                updated_entry.tcomment,
                "KDEAI: keep\nKDEAI-TM: copied_from=workspace",
            )
            self.assertIn("fuzzy", updated_entry.flags)

    def test_apply_derives_llm_tag_comment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            po_path = root / "locale" / "de.po"
            po_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_sample_po(po_path)

            po_file = polib.pofile(str(po_path))
            entry = po_file.find("File", msgctxt="menu")
            entry.tcomment = "KDEAI-AI: model=old\nKDEAI: keep"
            po_file.save(str(po_path))

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            config = build_config()
            base_state_hash = _entry_state_hash(entry, lang="de", config=config)
            config = build_config()

            plan_payload = self._build_plan(
                file_path="locale/de.po",
                base_sha256=base_sha256,
                base_state_hash=base_state_hash,
                msgctxt="menu",
                msgid="File",
                translation={"msgstr": "Datei", "msgstr_plural": {}},
                config_hash=config.config_hash,
                action="llm",
                tag_profile="llm",
            )

            result = apply.apply_plan(
                plan_payload,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                config=config,
                apply_mode="strict",
                overwrite="conservative",
            )
            self.assertEqual(result.files_written, ["locale/de.po"])
            updated = polib.pofile(str(po_path))
            updated_entry = updated.find("File", msgctxt="menu")
            self.assertEqual(
                updated_entry.tcomment,
                "KDEAI: keep\nKDEAI-AI: model=test-generation-model",
            )
            self.assertIn("fuzzy", updated_entry.flags)

    def test_apply_comments_preserves_trailing_newline(self):
        entry = polib.POEntry(msgid="File", msgstr="")
        entry.tcomment = "KDEAI: old\n"
        remove_prefixes = []
        ensure_lines = []
        append = ""

        apply._apply_comments(entry, remove_prefixes, ensure_lines, append)

        self.assertEqual(entry.tcomment, "KDEAI: old\n")

    def test_strict_mode_skips_file_on_any_entry_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            po_path = root / "locale" / "de.po"
            po_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_sample_po(po_path)

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            po_file = polib.pofile(str(po_path))
            entry_file = po_file.find("File", msgctxt="menu")
            entry_edit = po_file.find("Edit", msgctxt="menu")
            config = build_config()
            base_state_hash_file = _entry_state_hash(entry_file, lang="de", config=config)
            config = build_config()
            base_state_hash_edit = _entry_state_hash(entry_edit, lang="de", config=config)
            config = build_config()

            plan_payload = self._build_plan(
                file_path="locale/de.po",
                base_sha256=base_sha256,
                base_state_hash=base_state_hash_file,
                msgctxt="menu",
                msgid="File",
                translation={"msgstr": "Datei", "msgstr_plural": {}},
                config_hash=config.config_hash,
            )
            plan_payload["files"][0]["entries"].append(
                {
                    "msgctxt": "menu",
                    "msgid": "Edit",
                    "msgid_plural": "",
                    "base_state_hash": base_state_hash_edit + "-mismatch",
                    "action": "copy_tm",
                    "tag_profile": "tm_copy",
                    "translation": {"msgstr": "Bearbeiten", "msgstr_plural": {}},
                }
            )

            result = apply.apply_plan(
                plan_payload,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                config=config,
                apply_mode="strict",
                overwrite="conservative",
            )

            self.assertEqual(result.files_written, [])
            self.assertEqual(result.files_skipped, ["locale/de.po"])
            updated = polib.pofile(str(po_path))
            updated_file = updated.find("File", msgctxt="menu")
            updated_edit = updated.find("Edit", msgctxt="menu")
            self.assertEqual(updated_file.msgstr, "")
            self.assertEqual(updated_edit.msgstr, "")

    def test_rebase_mode_applies_matching_entries_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            po_path = root / "locale" / "de.po"
            po_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_sample_po(po_path)

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            po_file = polib.pofile(str(po_path))
            entry_file = po_file.find("File", msgctxt="menu")
            entry_edit = po_file.find("Edit", msgctxt="menu")
            config = build_config()
            base_state_hash_file = _entry_state_hash(entry_file, lang="de", config=config)
            config = build_config()
            base_state_hash_edit = _entry_state_hash(entry_edit, lang="de", config=config)
            config = build_config()

            plan_payload = self._build_plan(
                file_path="locale/de.po",
                base_sha256=base_sha256,
                base_state_hash=base_state_hash_file + "-mismatch",
                msgctxt="menu",
                msgid="File",
                translation={"msgstr": "Datei", "msgstr_plural": {}},
                config_hash=config.config_hash,
            )
            plan_payload["files"][0]["entries"].append(
                {
                    "msgctxt": "menu",
                    "msgid": "Edit",
                    "msgid_plural": "",
                    "base_state_hash": base_state_hash_edit,
                    "action": "copy_tm",
                    "tag_profile": "tm_copy",
                    "translation": {"msgstr": "Bearbeiten", "msgstr_plural": {}},
                }
            )

            result = apply.apply_plan(
                plan_payload,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                config=config,
                apply_mode="rebase",
                overwrite="conservative",
            )

            self.assertEqual(result.files_written, ["locale/de.po"])
            self.assertEqual(result.entries_applied, 1)
            updated = polib.pofile(str(po_path))
            updated_file = updated.find("File", msgctxt="menu")
            updated_edit = updated.find("Edit", msgctxt="menu")
            self.assertEqual(updated_file.msgstr, "")
            self.assertEqual(updated_edit.msgstr, "Bearbeiten")

    def test_overwrite_policy_matrix(self):
        cases = [
            ("conservative", "Alt", False, False),
            ("conservative", "", True, False),
            ("allow-nonempty", "Alt", False, True),
            ("allow-nonempty", "Alt", True, False),
            ("allow-reviewed", "", True, True),
            ("allow-reviewed", "Alt", True, False),
            ("all", "Alt", True, True),
        ]
        config = build_config()

        for policy, current_msgstr, reviewed, should_apply in cases:
            with self.subTest(policy=policy, reviewed=reviewed, current=current_msgstr):
                with tempfile.TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    po_path = root / "locale" / "de.po"
                    po_path.parent.mkdir(parents=True, exist_ok=True)
                    self._write_sample_po(po_path)

                    po_file = polib.pofile(str(po_path))
                    entry = po_file.find("File", msgctxt="menu")
                    entry.msgstr = current_msgstr
                    if reviewed:
                        entry.tcomment = "KDEAI-REVIEW: ok"
                    po_file.save(str(po_path))

                    base_bytes = po_path.read_bytes()
                    base_sha256 = hashlib.sha256(base_bytes).hexdigest()
                    config = build_config()
                    base_state_hash = _entry_state_hash(entry, lang="de", config=config)

                    plan_payload = self._build_plan(
                        file_path="locale/de.po",
                        base_sha256=base_sha256,
                        base_state_hash=base_state_hash,
                        msgctxt="menu",
                        msgid="File",
                        translation={"msgstr": "Neu", "msgstr_plural": {}},
                        config_hash=config.config_hash,
                    )

                    result = apply.apply_plan(
                        plan_payload,
                        project_root=root,
                        project_id="proj",
                        path_casefold=False,
                        config=config,
                        apply_mode="strict",
                        overwrite=policy,
                    )

                    updated = polib.pofile(str(po_path))
                    updated_entry = updated.find("File", msgctxt="menu")
                    if should_apply:
                        self.assertEqual(result.files_written, ["locale/de.po"])
                        self.assertEqual(updated_entry.msgstr, "Neu")
                    else:
                        self.assertEqual(result.files_written, [])
                        self.assertEqual(updated_entry.msgstr, current_msgstr)

    def test_overwrite_policy_changes_without_replan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            po_path = root / "locale" / "de.po"
            po_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_sample_po(po_path)

            po_file = polib.pofile(str(po_path))
            entry = po_file.find("File", msgctxt="menu")
            entry.msgstr = "Alt"
            po_file.save(str(po_path))

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            config = build_config()
            base_state_hash = _entry_state_hash(entry, lang="de", config=config)

            plan_payload = self._build_plan(
                file_path="locale/de.po",
                base_sha256=base_sha256,
                base_state_hash=base_state_hash,
                msgctxt="menu",
                msgid="File",
                translation={"msgstr": "Neu", "msgstr_plural": {}},
                config_hash=config.config_hash,
            )

            result_blocked = apply.apply_plan(
                plan_payload,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                config=config,
                apply_mode="strict",
                overwrite="conservative",
            )

            result_allowed = apply.apply_plan(
                plan_payload,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                config=config,
                apply_mode="strict",
                overwrite="allow-nonempty",
            )

            updated = polib.pofile(str(po_path))
            updated_entry = updated.find("File", msgctxt="menu")
            self.assertEqual(result_blocked.files_written, [])
            self.assertEqual(result_allowed.files_written, ["locale/de.po"])
            self.assertEqual(updated_entry.msgstr, "Neu")

    def test_validator_gate_blocks_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            po_path = root / "locale" / "de.po"
            po_path.parent.mkdir(parents=True, exist_ok=True)
            po_text = (
                'msgid ""\n'
                'msgstr ""\n'
                '"Content-Type: text/plain; charset=UTF-8\\n"\n'
                '"Plural-Forms: nplurals=2; plural=(n != 1);\\n"\n'
                "\n"
                'msgctxt "menu"\n'
                'msgid "File"\n'
                'msgid_plural "Files"\n'
                'msgstr[0] ""\n'
                'msgstr[1] ""\n'
            )
            po_path.write_text(po_text, encoding="utf-8")

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            po_file = polib.pofile(str(po_path))
            entry = po_file.find("File", msgctxt="menu")
            config = build_config()
            base_state_hash = _entry_state_hash(entry, lang="de", config=config)
            config = build_config()

            plan_payload = self._build_plan(
                file_path="locale/de.po",
                base_sha256=base_sha256,
                base_state_hash=base_state_hash,
                msgctxt="menu",
                msgid="File",
                translation={"msgstr": "", "msgstr_plural": {"0": "Datei"}},
                config_hash=config.config_hash,
            )
            plan_payload["files"][0]["entries"][0]["msgid_plural"] = "Files"

            result = apply.apply_plan(
                plan_payload,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                config=config,
                apply_mode="strict",
                overwrite="conservative",
            )

            self.assertEqual(result.files_written, [])
            self.assertIn("plural key consistency", result.errors)
            updated = polib.pofile(str(po_path))
            updated_entry = updated.find("File", msgctxt="menu")
            self.assertEqual(updated_entry.msgstr_plural, {0: "", 1: ""})

    def test_skip_when_file_changes_during_apply(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            po_path = root / "locale" / "de.po"
            po_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_sample_po(po_path)

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            po_file = polib.pofile(str(po_path))
            entry = po_file.find("File", msgctxt="menu")
            config = build_config()
            base_state_hash = _entry_state_hash(entry, lang="de", config=config)
            config = build_config()

            plan_payload = self._build_plan(
                file_path="locale/de.po",
                base_sha256=base_sha256,
                base_state_hash=base_state_hash,
                msgctxt="menu",
                msgid="File",
                translation={"msgstr": "Datei", "msgstr_plural": {}},
                config_hash=config.config_hash,
            )

            lock_calls = 0

            @contextlib.contextmanager
            def _mutate_file_lock(_lock_path):
                nonlocal lock_calls
                lock_calls += 1
                if lock_calls > 1:
                    po_path.write_text(
                        po_path.read_text(encoding="utf-8") + "\n# change",
                        encoding="utf-8",
                    )
                yield

            with mock.patch(
                "kdeai.apply.locks.acquire_file_lock",
                side_effect=_mutate_file_lock,
            ):
                result = apply.apply_plan(
                    plan_payload,
                    project_root=root,
                    project_id="proj",
                    path_casefold=False,
                    config=config,
                    apply_mode="strict",
                    overwrite="conservative",
                )

            self.assertEqual(result.files_written, [])
            self.assertEqual(result.files_skipped, ["locale/de.po"])
            self.assertIn("locale/de.po: skipped: file changed since phase A", result.warnings)
            self.assertIn("# change", po_path.read_text(encoding="utf-8"))
            updated = polib.pofile(str(po_path))
            updated_entry = updated.find("File", msgctxt="menu")
            self.assertEqual(updated_entry.msgstr, "")
            po_files = sorted(path.name for path in po_path.parent.glob("*.po"))
            self.assertEqual(po_files, ["de.po"])

    def test_needs_llm_action_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            po_path = root / "locale" / "de.po"
            po_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_sample_po(po_path)

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            po_file = polib.pofile(str(po_path))
            entry = po_file.find("File", msgctxt="menu")
            config = build_config()
            base_state_hash = _entry_state_hash(entry, lang="de", config=config)
            config = build_config()

            plan_payload = self._build_plan(
                file_path="locale/de.po",
                base_sha256=base_sha256,
                base_state_hash=base_state_hash,
                msgctxt="menu",
                msgid="File",
                translation={"msgstr": "", "msgstr_plural": {}},
                config_hash=config.config_hash,
            )
            plan_payload["files"][0]["entries"][0]["action"] = "needs_llm"

            result = apply.apply_plan(
                plan_payload,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                config=config,
                apply_mode="strict",
                overwrite="conservative",
            )
            self.assertEqual(result.files_written, [])
            self.assertEqual(
                result.errors,
                ["locale/de.po: unsupported action: needs_llm"],
            )
            self.assertEqual(result.files_skipped, ["locale/de.po"])
            updated = polib.pofile(str(po_path))
            updated_entry = updated.find("File", msgctxt="menu")
            self.assertEqual(updated_entry.msgstr, "")

    def test_skip_action_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            po_path = root / "locale" / "de.po"
            po_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_sample_po(po_path)

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            po_file = polib.pofile(str(po_path))
            entry = po_file.find("File", msgctxt="menu")
            config = build_config()
            base_state_hash = _entry_state_hash(entry, lang="de", config=config)
            config = build_config()

            plan_payload = self._build_plan(
                file_path="locale/de.po",
                base_sha256=base_sha256,
                base_state_hash=base_state_hash,
                msgctxt="menu",
                msgid="File",
                translation={"msgstr": "Datei", "msgstr_plural": {}},
                config_hash=config.config_hash,
                action="skip",
            )

            result = apply.apply_plan(
                plan_payload,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                config=config,
                apply_mode="strict",
                overwrite="conservative",
            )

            self.assertEqual(result.files_written, [])
            self.assertEqual(result.files_skipped, ["locale/de.po"])
            self.assertEqual(result.errors, [])
            updated = polib.pofile(str(po_path))
            updated_entry = updated.find("File", msgctxt="menu")
            self.assertEqual(updated_entry.msgstr, "")

    def test_apply_uses_single_temp_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            po_path = root / "locale" / "de.po"
            po_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_sample_po(po_path)

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            po_file = polib.pofile(str(po_path))
            entry = po_file.find("File", msgctxt="menu")
            config = build_config()
            base_state_hash = _entry_state_hash(entry, lang="de", config=config)
            config = build_config()

            plan_payload = self._build_plan(
                file_path="locale/de.po",
                base_sha256=base_sha256,
                base_state_hash=base_state_hash,
                msgctxt="menu",
                msgid="File",
                translation={"msgstr": "Datei", "msgstr_plural": {}},
                config_hash=config.config_hash,
            )

            call_count = 0
            original_tempfile = tempfile.NamedTemporaryFile

            def _counting_named_tempfile(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                return original_tempfile(*args, **kwargs)

            with mock.patch(
                "kdeai.apply.tempfile.NamedTemporaryFile",
                side_effect=_counting_named_tempfile,
            ):
                result = apply.apply_plan(
                    plan_payload,
                    project_root=root,
                    project_id="proj",
                    path_casefold=False,
                    config=config,
                    apply_mode="strict",
                    overwrite="conservative",
                )

            self.assertEqual(result.files_written, ["locale/de.po"])
            self.assertEqual(call_count, 2)

    def test_llm_action_empty_translation_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            po_path = root / "locale" / "de.po"
            po_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_sample_po(po_path)

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            po_file = polib.pofile(str(po_path))
            entry = po_file.find("File", msgctxt="menu")
            config = build_config()
            base_state_hash = _entry_state_hash(entry, lang="de", config=config)
            config = build_config()

            plan_payload = self._build_plan(
                file_path="locale/de.po",
                base_sha256=base_sha256,
                base_state_hash=base_state_hash,
                msgctxt="menu",
                msgid="File",
                translation={"msgstr": "", "msgstr_plural": {}},
                config_hash=config.config_hash,
            )
            plan_payload["files"][0]["entries"][0]["action"] = "llm"
            plan_payload["files"][0]["entries"][0]["tag_profile"] = "llm"

            result = apply.apply_plan(
                plan_payload,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                config=config,
                apply_mode="strict",
                overwrite="conservative",
            )
            self.assertEqual(result.files_written, [])
            self.assertIn("non-empty translation", result.errors)
            updated = polib.pofile(str(po_path))
            updated_entry = updated.find("File", msgctxt="menu")
            self.assertEqual(updated_entry.msgstr, "")

    def test_strict_skips_when_entry_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            po_path = root / "locale" / "de.po"
            po_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_sample_po(po_path)

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            po_file = polib.pofile(str(po_path))
            entry = po_file.find("File", msgctxt="menu")
            config = build_config()
            base_state_hash = _entry_state_hash(entry, lang="de", config=config)

            config = build_config()
            plan_payload = {
                "format": 1,
                "project_id": "proj",
                "config_hash": config.config_hash,
                "lang": "de",
                "apply_defaults": {"mode": "strict", "overwrite": "conservative", "post_index": "off"},
                "files": [
                    {
                        "file_path": "locale/de.po",
                        "base_sha256": base_sha256,
                        "entries": [
                            {
                                "msgctxt": "menu",
                                "msgid": "File",
                                "msgid_plural": "",
                                "base_state_hash": base_state_hash,
                                "action": "copy_tm",
                                "tag_profile": "tm_copy",
                                "translation": {"msgstr": "Datei", "msgstr_plural": {}},
                            },
                            {
                                "msgctxt": "menu",
                                "msgid": "Missing",
                                "msgid_plural": "",
                                "base_state_hash": "missing",
                                "action": "copy_tm",
                                "tag_profile": "tm_copy",
                                "translation": {"msgstr": "Fehlt", "msgstr_plural": {}},
                            },
                        ],
                    }
                ],
            }
            plan_payload = plan.finalize_plan(plan_payload)

            result = apply.apply_plan(
                plan_payload,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                config=config,
                apply_mode="strict",
                overwrite="conservative",
            )

            self.assertEqual(result.files_written, [])
            self.assertEqual(result.files_skipped, ["locale/de.po"])
            updated = polib.pofile(str(po_path))
            updated_entry = updated.find("File", msgctxt="menu")
            self.assertEqual(updated_entry.msgstr, "")

    def test_rebase_applies_when_entry_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            po_path = root / "locale" / "de.po"
            po_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_sample_po(po_path)

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            po_file = polib.pofile(str(po_path))
            entry = po_file.find("File", msgctxt="menu")
            config = build_config()
            base_state_hash = _entry_state_hash(entry, lang="de", config=config)

            config = build_config()
            plan_payload = {
                "format": 1,
                "project_id": "proj",
                "config_hash": config.config_hash,
                "lang": "de",
                "apply_defaults": {"mode": "strict", "overwrite": "conservative", "post_index": "off"},
                "files": [
                    {
                        "file_path": "locale/de.po",
                        "base_sha256": base_sha256,
                        "entries": [
                            {
                                "msgctxt": "menu",
                                "msgid": "File",
                                "msgid_plural": "",
                                "base_state_hash": base_state_hash,
                                "action": "copy_tm",
                                "tag_profile": "tm_copy",
                                "translation": {"msgstr": "Datei", "msgstr_plural": {}},
                            },
                            {
                                "msgctxt": "menu",
                                "msgid": "Missing",
                                "msgid_plural": "",
                                "base_state_hash": "missing",
                                "action": "copy_tm",
                                "tag_profile": "tm_copy",
                                "translation": {"msgstr": "Fehlt", "msgstr_plural": {}},
                            },
                        ],
                    }
                ],
            }
            plan_payload = plan.finalize_plan(plan_payload)

            result = apply.apply_plan(
                plan_payload,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                config=config,
                apply_mode="rebase",
                overwrite="conservative",
            )

            self.assertEqual(result.files_written, ["locale/de.po"])
            self.assertEqual(result.entries_applied, 1)
            updated = polib.pofile(str(po_path))
            updated_entry = updated.find("File", msgctxt="menu")
            self.assertEqual(updated_entry.msgstr, "Datei")

    def test_post_index_failure_is_warning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            po_path = root / "locale" / "de.po"
            po_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_sample_po(po_path)

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            po_file = polib.pofile(str(po_path))
            entry = po_file.find("File", msgctxt="menu")
            config = build_config()
            base_state_hash = _entry_state_hash(entry, lang="de", config=config)
            config = build_config()

            plan_payload = self._build_plan(
                file_path="locale/de.po",
                base_sha256=base_sha256,
                base_state_hash=base_state_hash,
                msgctxt="menu",
                msgid="File",
                translation={"msgstr": "Datei", "msgstr_plural": {}},
                config_hash=config.config_hash,
            )

            result = apply.apply_plan(
                plan_payload,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                apply_mode="strict",
                overwrite="conservative",
                post_index=True,
                workspace_conn=object(),
                config=config,
            )

            self.assertEqual(result.errors, [])
            self.assertEqual(result.warnings, [])
            updated = polib.pofile(str(po_path))
            updated_entry = updated.find("File", msgctxt="menu")
            self.assertEqual(updated_entry.msgstr, "Datei")

    def test_apply_rejects_unsupported_plan_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            po_path = root / "locale" / "de.po"
            po_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_sample_po(po_path)

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            po_file = polib.pofile(str(po_path))
            entry = po_file.find("File", msgctxt="menu")
            config = build_config()
            base_state_hash = _entry_state_hash(entry, lang="de", config=config)

            config = build_config()
            plan_payload = self._build_plan(
                file_path="locale/de.po",
                base_sha256=base_sha256,
                base_state_hash=base_state_hash,
                msgctxt="menu",
                msgid="File",
                translation={"msgstr": "Datei", "msgstr_plural": {}},
                config_hash=config.config_hash,
            )
            plan_payload["format"] = 999

            result = apply.apply_plan(
                plan_payload,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                config=config,
                apply_mode="strict",
                overwrite="conservative",
            )

            self.assertTrue(result.errors)
            self.assertIn("unsupported plan format", result.errors)

    def test_apply_rejects_unsupported_apply_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            po_path = root / "locale" / "de.po"
            po_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_sample_po(po_path)

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            po_file = polib.pofile(str(po_path))
            entry = po_file.find("File", msgctxt="menu")
            config = build_config()
            base_state_hash = _entry_state_hash(entry, lang="de", config=config)

            config = build_config()
            plan_payload = self._build_plan(
                file_path="locale/de.po",
                base_sha256=base_sha256,
                base_state_hash=base_state_hash,
                msgctxt="menu",
                msgid="File",
                translation={"msgstr": "Datei", "msgstr_plural": {}},
                config_hash=config.config_hash,
            )

            result = apply.apply_plan(
                plan_payload,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                config=config,
                apply_mode="weird",
                overwrite="conservative",
            )

            self.assertTrue(result.errors)
            self.assertTrue(any("unsupported apply mode" in error for error in result.errors))

    def test_apply_rejects_missing_lang(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            po_path = root / "locale" / "de.po"
            po_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_sample_po(po_path)

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            po_file = polib.pofile(str(po_path))
            entry = po_file.find("File", msgctxt="menu")
            config = build_config()
            base_state_hash = _entry_state_hash(entry, lang="de", config=config)

            config = build_config()
            plan_payload = self._build_plan(
                file_path="locale/de.po",
                base_sha256=base_sha256,
                base_state_hash=base_state_hash,
                msgctxt="menu",
                msgid="File",
                translation={"msgstr": "Datei", "msgstr_plural": {}},
                config_hash=config.config_hash,
            )
            plan_payload["lang"] = ""

            result = apply.apply_plan(
                plan_payload,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                config=config,
                apply_mode="strict",
                overwrite="conservative",
            )

            self.assertTrue(result.errors)
            self.assertIn("plan lang missing", result.errors)

    def test_apply_skips_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            po_path = root / "locale" / "de.po"
            po_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_sample_po(po_path)

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            po_file = polib.pofile(str(po_path))
            entry = po_file.find("File", msgctxt="menu")
            config = build_config()
            base_state_hash = _entry_state_hash(entry, lang="de", config=config)

            config = build_config()
            plan_payload = self._build_plan(
                file_path="locale/de.po",
                base_sha256=base_sha256,
                base_state_hash=base_state_hash,
                msgctxt="menu",
                msgid="File",
                translation={"msgstr": "Datei", "msgstr_plural": {}},
                config_hash=config.config_hash,
            )
            plan_payload["files"].append(
                {
                    "file_path": "locale/missing.po",
                    "base_sha256": "",
                    "entries": [],
                }
            )

            result = apply.apply_plan(
                plan_payload,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                config=config,
                apply_mode="strict",
                overwrite="conservative",
            )

            self.assertIn("locale/missing.po", result.files_skipped)
            self.assertTrue(any("locale/missing.po" in error for error in result.errors))
            self.assertEqual(result.files_written, ["locale/de.po"])
            self.assertEqual(result.entries_applied, 1)
            self.assertFalse((root / "locale/missing.po").exists())

    def test_apply_errors_include_entry_identity_on_missing_translation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            po_path = root / "locale" / "de.po"
            po_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_sample_po(po_path)

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            po_file = polib.pofile(str(po_path))
            entry = po_file.find("File", msgctxt="menu")
            config = build_config()
            base_state_hash = _entry_state_hash(entry, lang="de", config=config)

            config = build_config()
            plan_payload = {
                "format": 1,
                "project_id": "proj",
                "config_hash": config.config_hash,
                "lang": "de",
                "apply_defaults": {
                    "mode": "strict",
                    "overwrite": "conservative",
                    "post_index": "off",
                },
                "files": [
                    {
                        "file_path": "locale/de.po",
                        "base_sha256": base_sha256,
                        "entries": [
                            {
                                "msgctxt": "menu",
                                "msgid": "File",
                                "msgid_plural": "",
                                "base_state_hash": base_state_hash,
                                "action": "llm",
                                "tag_profile": "llm",
                            }
                        ],
                    }
                ],
            }
            plan_payload = plan.finalize_plan(plan_payload)

            result = apply.apply_plan(
                plan_payload,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                config=config,
                apply_mode="strict",
                overwrite="conservative",
            )

            self.assertTrue(result.errors)
            self.assertTrue(
                any("locale/de.po: invalid plan entry 0" in error for error in result.errors)
            )

    def test_apply_preserves_warnings_on_strict_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            po_path = root / "locale" / "de.po"
            po_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_sample_po(po_path)

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            po_file = polib.pofile(str(po_path))
            entry = po_file.find("File", msgctxt="menu")
            config = build_config()
            base_state_hash = _entry_state_hash(entry, lang="de", config=config)

            config = build_config()
            plan_payload = {
                "format": 1,
                "project_id": "proj",
                "config_hash": config.config_hash,
                "lang": "de",
                "apply_defaults": {
                    "mode": "strict",
                    "overwrite": "conservative",
                    "post_index": "off",
                },
                "files": [
                    {
                        "file_path": "locale/de.po",
                        "base_sha256": base_sha256,
                        "entries": [
                            {
                                "msgctxt": "menu",
                                "msgid": "File",
                                "msgid_plural": "",
                                "base_state_hash": base_state_hash,
                                "action": "weird",
                            },
                            {
                                "msgctxt": "menu",
                                "msgid": "Missing",
                                "msgid_plural": "",
                                "base_state_hash": "missing",
                                "action": "copy_tm",
                                "tag_profile": "tm_copy",
                            },
                        ],
                    }
                ],
            }
            plan_payload = plan.finalize_plan(plan_payload)

            result = apply.apply_plan(
                plan_payload,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                config=config,
                apply_mode="strict",
                overwrite="conservative",
            )

            self.assertIn("locale/de.po", result.files_skipped)
            self.assertTrue(any("unsupported action" in error for error in result.errors))

    def test_apply_fails_on_project_id_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            po_path = root / "locale" / "de.po"
            po_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_sample_po(po_path)

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            po_file = polib.pofile(str(po_path))
            entry = po_file.find("File", msgctxt="menu")
            config = build_config()
            base_state_hash = _entry_state_hash(entry, lang="de", config=config)

            config = build_config()
            plan_payload = self._build_plan(
                file_path="locale/de.po",
                base_sha256=base_sha256,
                base_state_hash=base_state_hash,
                msgctxt="menu",
                msgid="File",
                translation={"msgstr": "Datei", "msgstr_plural": {}},
                config_hash=config.config_hash,
            )

            result = apply.apply_plan(
                plan_payload,
                project_root=root,
                project_id="other-proj",
                path_casefold=False,
                config=config,
                apply_mode="strict",
                overwrite="conservative",
            )

            self.assertEqual(result.files_written, [])
            self.assertEqual(result.entries_applied, 0)
            self.assertTrue(result.errors)
            self.assertIn("project_id", result.errors[0])
            updated = polib.pofile(str(po_path))
            updated_entry = updated.find("File", msgctxt="menu")
            self.assertEqual(updated_entry.msgstr, "")

    def test_apply_rejects_parent_path_escape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            po_path = root / "locale" / "de.po"
            po_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_sample_po(po_path)

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            po_file = polib.pofile(str(po_path))
            entry = po_file.find("File", msgctxt="menu")
            config = build_config()
            base_state_hash = _entry_state_hash(entry, lang="de", config=config)

            config = build_config()
            plan_payload = self._build_plan(
                file_path="../secrets.po",
                base_sha256=base_sha256,
                base_state_hash=base_state_hash,
                msgctxt="menu",
                msgid="File",
                translation={"msgstr": "Datei", "msgstr_plural": {}},
                config_hash=config.config_hash,
            )

            result = apply.apply_plan(
                plan_payload,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                config=config,
                apply_mode="strict",
                overwrite="conservative",
            )

            self.assertEqual(result.files_written, [])
            self.assertEqual(result.entries_applied, 0)
            self.assertTrue(result.errors)
            self.assertTrue(any("plan file_path is invalid" in error for error in result.errors))
            outside_path = (root / "../secrets.po").resolve()
            self.assertFalse(outside_path.exists())

    def test_apply_rejects_dot_path_segment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            po_path = root / "x.po"
            po_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_sample_po(po_path)

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            po_file = polib.pofile(str(po_path))
            entry = po_file.find("File", msgctxt="menu")
            config = build_config()
            base_state_hash = _entry_state_hash(entry, lang="de", config=config)

            config = build_config()
            plan_payload = self._build_plan(
                file_path="./x.po",
                base_sha256=base_sha256,
                base_state_hash=base_state_hash,
                msgctxt="menu",
                msgid="File",
                translation={"msgstr": "Datei", "msgstr_plural": {}},
                config_hash=config.config_hash,
            )

            result = apply.apply_plan(
                plan_payload,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                config=config,
                apply_mode="strict",
                overwrite="conservative",
            )

            self.assertTrue(result.errors)
            self.assertTrue(any("plan file_path is invalid" in error for error in result.errors))

    def test_apply_rejects_double_slash_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            po_path = root / "a" / "b.po"
            po_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_sample_po(po_path)

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            po_file = polib.pofile(str(po_path))
            entry = po_file.find("File", msgctxt="menu")
            config = build_config()
            base_state_hash = _entry_state_hash(entry, lang="de", config=config)

            config = build_config()
            plan_payload = self._build_plan(
                file_path="a//b.po",
                base_sha256=base_sha256,
                base_state_hash=base_state_hash,
                msgctxt="menu",
                msgid="File",
                translation={"msgstr": "Datei", "msgstr_plural": {}},
                config_hash=config.config_hash,
            )

            result = apply.apply_plan(
                plan_payload,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                config=config,
                apply_mode="strict",
                overwrite="conservative",
            )

            self.assertTrue(result.errors)
            self.assertTrue(any("plan file_path is invalid" in error for error in result.errors))

    def test_apply_accepts_normalized_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            po_path = root / "a" / "b.po"
            po_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_sample_po(po_path)

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            po_file = polib.pofile(str(po_path))
            entry = po_file.find("File", msgctxt="menu")
            config = build_config()
            base_state_hash = _entry_state_hash(entry, lang="de", config=config)

            config = build_config()
            plan_payload = self._build_plan(
                file_path="a/b.po",
                base_sha256=base_sha256,
                base_state_hash=base_state_hash,
                msgctxt="menu",
                msgid="File",
                translation={"msgstr": "Datei", "msgstr_plural": {}},
                config_hash=config.config_hash,
            )

            result = apply.apply_plan(
                plan_payload,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                config=config,
                apply_mode="strict",
                overwrite="conservative",
            )

            self.assertFalse(any("plan file_path is invalid" in error for error in result.errors))
            self.assertEqual(result.files_written, ["a/b.po"])

    def test_apply_rejects_absolute_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            po_path = root / "locale" / "de.po"
            po_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_sample_po(po_path)

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            po_file = polib.pofile(str(po_path))
            entry = po_file.find("File", msgctxt="menu")
            config = build_config()
            base_state_hash = _entry_state_hash(entry, lang="de", config=config)

            config = build_config()
            outside_path = (root.parent / "outside.po").resolve()
            plan_payload = self._build_plan(
                file_path=str(outside_path),
                base_sha256=base_sha256,
                base_state_hash=base_state_hash,
                msgctxt="menu",
                msgid="File",
                translation={"msgstr": "Datei", "msgstr_plural": {}},
                config_hash=config.config_hash,
            )

            result = apply.apply_plan(
                plan_payload,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                config=config,
                apply_mode="strict",
                overwrite="conservative",
            )

            self.assertEqual(result.files_written, [])
            self.assertEqual(result.entries_applied, 0)
            self.assertTrue(result.errors)
            self.assertTrue(any("plan file_path is invalid" in error for error in result.errors))
            self.assertFalse(outside_path.exists())

    def test_apply_fails_on_config_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            po_path = root / "locale" / "de.po"
            po_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_sample_po(po_path)

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            po_file = polib.pofile(str(po_path))
            entry = po_file.find("File", msgctxt="menu")
            config = build_config()
            base_state_hash = _entry_state_hash(entry, lang="de", config=config)

            config = build_config()
            plan_payload = self._build_plan(
                file_path="locale/de.po",
                base_sha256=base_sha256,
                base_state_hash=base_state_hash,
                msgctxt="menu",
                msgid="File",
                translation={"msgstr": "Datei", "msgstr_plural": {}},
                config_hash="wrong-hash",
            )

            result = apply.apply_plan(
                plan_payload,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                config=config,
                apply_mode="strict",
                overwrite="conservative",
            )

            self.assertEqual(result.files_written, [])
            self.assertEqual(result.entries_applied, 0)
            self.assertEqual(result.errors, ["plan config_hash does not match current config"])
            self.assertFalse((root / ".kdeai" / "locks").exists())
            updated = polib.pofile(str(po_path))
            updated_entry = updated.find("File", msgctxt="menu")
            self.assertEqual(updated_entry.msgstr, "")

    def test_apply_fails_on_missing_config_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            po_path = root / "locale" / "de.po"
            po_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_sample_po(po_path)

            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()
            po_file = polib.pofile(str(po_path))
            entry = po_file.find("File", msgctxt="menu")
            config = build_config()
            base_state_hash = _entry_state_hash(entry, lang="de", config=config)

            config = build_config()
            plan_payload = {
                "format": 1,
                "project_id": "proj",
                "lang": "de",
                "apply_defaults": {
                    "mode": "strict",
                    "overwrite": "conservative",
                    "post_index": "off",
                },
                "files": [
                    {
                        "file_path": "locale/de.po",
                        "base_sha256": base_sha256,
                        "entries": [
                            {
                                "msgctxt": "menu",
                                "msgid": "File",
                                "msgid_plural": "",
                                "base_state_hash": base_state_hash,
                                "action": "copy_tm",
                                "tag_profile": "tm_copy",
                                "translation": {"msgstr": "Datei", "msgstr_plural": {}},
                            }
                        ],
                    }
                ],
            }
            plan_payload = plan.finalize_plan(plan_payload)

            result = apply.apply_plan(
                plan_payload,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                config=config,
                apply_mode="strict",
                overwrite="conservative",
            )

            self.assertEqual(result.files_written, [])
            self.assertEqual(result.entries_applied, 0)
            self.assertEqual(result.errors, ["plan config_hash does not match current config"])
            self.assertFalse((root / ".kdeai" / "locks").exists())
            updated = polib.pofile(str(po_path))
            updated_entry = updated.find("File", msgctxt="menu")
            self.assertEqual(updated_entry.msgstr, "")


class TestLockId(unittest.TestCase):
    def test_lock_id_varies_by_project(self):
        lock_a = locks.lock_id("proj-a", "locale/de.po")
        lock_b = locks.lock_id("proj-b", "locale/de.po")
        self.assertNotEqual(lock_a, lock_b)


if __name__ == "__main__":
    unittest.main()
