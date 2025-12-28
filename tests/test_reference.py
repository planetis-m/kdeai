import sqlite3
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

import polib

from conftest import build_config
from kdeai import po_model
from kdeai import reference as kderef


def _write_po(
    path: Path,
    *,
    msgid: str,
    msgstr: str,
    reviewed: bool,
    ai_generated: bool,
    lang: str = "de",
) -> None:
    po_file = polib.POFile()
    po_file.metadata["Language"] = lang
    entry = polib.POEntry(msgid=msgid, msgstr=msgstr)
    if reviewed:
        entry.tcomment = "KDEAI-REVIEW: ok"
    if ai_generated:
        entry.flags.append("kdeai-ai")
    po_file.append(entry)
    po_file.save(str(path))


class TestReferenceBestSelection(unittest.TestCase):
    def test_best_translation_prefers_human_when_review_equal(self):
        config = build_config()
        project_id = "project-1"
        with TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir)
            first = project_root / "first.po"
            second = project_root / "second.po"

            _write_po(
                first,
                msgid="File",
                msgstr="Datei (human)",
                reviewed=True,
                ai_generated=False,
            )
            _write_po(
                second,
                msgid="File",
                msgstr="Datei (ai)",
                reviewed=True,
                ai_generated=True,
            )

            snapshot = kderef.build_reference_snapshot(
                project_root,
                project_id=project_id,
                path_casefold=False,
                config=config,
                config_hash=config.config_hash,
                paths=[first, second],
            )

            source_key = po_model.source_key_for("", "File", "")
            with closing(sqlite3.connect(snapshot.db_path)) as conn:
                row = conn.execute(
                    "SELECT msgstr, review_status, is_ai_generated "
                    "FROM best_translations WHERE source_key = ? AND lang = ?",
                    (source_key, "de"),
                ).fetchone()

        self.assertIsNotNone(row)
        msgstr, review_status, is_ai_generated = row
        self.assertEqual(msgstr, "Datei (human)")
        self.assertEqual(review_status, "reviewed")
        self.assertEqual(is_ai_generated, 0)

    def test_best_translation_prefers_review_status_over_human(self):
        config = build_config()
        project_id = "project-1"
        with TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir)
            reviewed_ai = project_root / "reviewed_ai.po"
            draft_human = project_root / "draft_human.po"

            _write_po(
                reviewed_ai,
                msgid="Open",
                msgstr="Oeffnen (ai)",
                reviewed=True,
                ai_generated=True,
            )
            _write_po(
                draft_human,
                msgid="Open",
                msgstr="Oeffnen (human)",
                reviewed=False,
                ai_generated=False,
            )

            snapshot = kderef.build_reference_snapshot(
                project_root,
                project_id=project_id,
                path_casefold=False,
                config=config,
                config_hash=config.config_hash,
                paths=[reviewed_ai, draft_human],
            )

            source_key = po_model.source_key_for("", "Open", "")
            with closing(sqlite3.connect(snapshot.db_path)) as conn:
                row = conn.execute(
                    "SELECT msgstr, review_status, is_ai_generated "
                    "FROM best_translations WHERE source_key = ? AND lang = ?",
                    (source_key, "de"),
                ).fetchone()

        self.assertIsNotNone(row)
        msgstr, review_status, is_ai_generated = row
        self.assertEqual(msgstr, "Oeffnen (ai)")
        self.assertEqual(review_status, "reviewed")
        self.assertEqual(is_ai_generated, 1)
