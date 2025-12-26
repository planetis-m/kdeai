import hashlib
import tempfile
import unittest
from pathlib import Path

import polib

from conftest import build_config
from kdeai import apply
from kdeai import plan


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
            base_state_hash = apply.entry_state_hash(entry, lang="de")

            plan_payload = {
                "format": 1,
                "project_id": "proj",
                "config_hash": "cfg",
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
                config=build_config(),
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
            base_state_hash = apply.entry_state_hash(entry, lang="de")

            entry.msgstr = "Alt"
            po_file.save(str(po_path))
            base_bytes = po_path.read_bytes()
            base_sha256 = hashlib.sha256(base_bytes).hexdigest()

            plan_payload = {
                "format": 1,
                "project_id": "proj",
                "config_hash": "cfg",
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
                config=build_config(),
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
        extra: dict | None = None,
    ) -> dict:
        plan_payload = {
            "format": 1,
            "project_id": "proj",
            "config_hash": "cfg",
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
                            "translation": translation,
                        }
                    ],
                }
            ],
        }
        if extra:
            plan_payload.update(extra)
        return plan.finalize_plan(plan_payload)

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
            base_state_hash = apply.entry_state_hash(entry, lang="de")

            plan_payload = self._build_plan(
                file_path="locale/de.po",
                base_sha256=base_sha256,
                base_state_hash=base_state_hash,
                msgctxt="menu",
                msgid="File",
                translation={"msgstr": "Datei", "msgstr_plural": {}},
            )

            po_path.write_text(po_path.read_text(encoding="utf-8") + "\n# touched\n", encoding="utf-8")

            result = apply.apply_plan(
                plan_payload,
                project_root=root,
                config=build_config(),
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
            base_state_file = apply.entry_state_hash(entry_file, lang="de")
            base_state_edit = apply.entry_state_hash(entry_edit, lang="de")

            plan_payload = {
                "format": 1,
                "project_id": "proj",
                "config_hash": "cfg",
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
                                "translation": {"msgstr": "Datei", "msgstr_plural": {}},
                            },
                            {
                                "msgctxt": "menu",
                                "msgid": "Edit",
                                "msgid_plural": "",
                                "base_state_hash": base_state_edit,
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
                config=build_config(),
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
            base_state_hash = apply.entry_state_hash(entry, lang="de")

            plan_payload = self._build_plan(
                file_path="locale/de.po",
                base_sha256=base_sha256,
                base_state_hash=base_state_hash,
                msgctxt="menu",
                msgid="Hello %s",
                translation={"msgstr": "Hallo", "msgstr_plural": {}},
                extra={"placeholder_patterns": [r"%\w"]},
            )

            result = apply.apply_plan(
                plan_payload,
                project_root=root,
                config=build_config(),
                apply_mode="strict",
                overwrite="conservative",
            )
            self.assertEqual(result.files_written, [])
            self.assertTrue(result.errors)
            updated = polib.pofile(str(po_path))
            updated_entry = updated.find("Hello %s", msgctxt="menu")
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
            base_state_hash = apply.entry_state_hash(entry, lang="de")

            plan_payload = {
                "format": 1,
                "project_id": "proj",
                "config_hash": "cfg",
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
                                "translation": {"msgstr": "Datei", "msgstr_plural": {}},
                            },
                            {
                                "msgctxt": "menu",
                                "msgid": "Missing",
                                "msgid_plural": "",
                                "base_state_hash": "missing",
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
                config=build_config(),
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
            base_state_hash = apply.entry_state_hash(entry, lang="de")

            plan_payload = {
                "format": 1,
                "project_id": "proj",
                "config_hash": "cfg",
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
                                "translation": {"msgstr": "Datei", "msgstr_plural": {}},
                            },
                            {
                                "msgctxt": "menu",
                                "msgid": "Missing",
                                "msgid_plural": "",
                                "base_state_hash": "missing",
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
                config=build_config(),
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
            base_state_hash = apply.entry_state_hash(entry, lang="de")

            plan_payload = self._build_plan(
                file_path="locale/de.po",
                base_sha256=base_sha256,
                base_state_hash=base_state_hash,
                msgctxt="menu",
                msgid="File",
                translation={"msgstr": "Datei", "msgstr_plural": {}},
            )

            result = apply.apply_plan(
                plan_payload,
                project_root=root,
                apply_mode="strict",
                overwrite="conservative",
                post_index=True,
                workspace_conn=object(),
                config=build_config({"tm": {"selection": {}}}),
            )

            self.assertEqual(result.errors, [])
            self.assertEqual(result.warnings, [])
            updated = polib.pofile(str(po_path))
            updated_entry = updated.find("File", msgctxt="menu")
            self.assertEqual(updated_entry.msgstr, "Datei")


if __name__ == "__main__":
    unittest.main()
