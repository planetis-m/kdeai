import sqlite3
import unittest

from kdeai import db as kde_db
from kdeai import hash as kdehash
from kdeai import retrieve_tm
from kdeai import workspace_tm


def _setup_workspace_db():
    conn = sqlite3.connect(":memory:")
    conn.executescript(kde_db.WORKSPACE_TM_SCHEMA)
    return conn


def _setup_reference_db():
    conn = sqlite3.connect(":memory:")
    conn.executescript(kde_db.REFERENCE_TM_SCHEMA)
    return conn


def _sample_po_bytes():
    po_text = (
        'msgid ""\n'
        'msgstr ""\n'
        '"Content-Type: text/plain; charset=UTF-8\\n"\n'
        "\n"
        'msgctxt "menu"\n'
        'msgid "File"\n'
        'msgstr "Datei"\n'
    )
    return po_text.encode("utf-8")


class TestWorkspaceIndexing(unittest.TestCase):
    def test_index_atomicity_rolls_back(self):
        conn = _setup_workspace_db()
        original = workspace_tm._recompute_best_translations

        def _explode(*_args, **_kwargs):
            raise RuntimeError("boom")

        workspace_tm._recompute_best_translations = _explode
        try:
            with self.assertRaises(RuntimeError):
                workspace_tm.index_file_snapshot_tm(
                    conn,
                    file_path="locale/de.po",
                    lang="de",
                    bytes=_sample_po_bytes(),
                    sha256="abc",
                    mtime_ns=123,
                    size=10,
                )
        finally:
            workspace_tm._recompute_best_translations = original

        for table in ["files", "sources", "translations", "best_translations"]:
            rows = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            self.assertEqual(rows, 0)

    def test_index_populates_best_translations(self):
        conn = _setup_workspace_db()
        workspace_tm.index_file_snapshot_tm(
            conn,
            file_path="locale/de.po",
            lang="de",
            bytes=_sample_po_bytes(),
            sha256="abc",
            mtime_ns=123,
            size=10,
        )
        row = conn.execute(
            "SELECT file_path, msgstr FROM best_translations WHERE lang = 'de'"
        ).fetchone()
        self.assertEqual(row, ("locale/de.po", "Datei"))

    def test_best_translations_selection_prefers_reviewed(self):
        conn = _setup_workspace_db()
        conn.execute(
            "INSERT INTO files (file_path, lang, indexed_sha256, indexed_mtime_ns, "
            "indexed_size, indexed_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("a.po", "de", "aaa", 1, 1, "now"),
        )
        conn.execute(
            "INSERT INTO files (file_path, lang, indexed_sha256, indexed_mtime_ns, "
            "indexed_size, indexed_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("b.po", "de", "bbb", 1, 1, "now"),
        )
        source_key = "source-key"
        msgstr_plural_json = kdehash.canonical_msgstr_plural({})
        conn.execute(
            "INSERT INTO translations (file_path, lang, source_key, msgstr, msgstr_plural, "
            "review_status, is_ai_generated, translation_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "a.po",
                "de",
                source_key,
                "alpha",
                msgstr_plural_json,
                "draft",
                0,
                kdehash.translation_hash(source_key, "de", "alpha", {}),
            ),
        )
        conn.execute(
            "INSERT INTO translations (file_path, lang, source_key, msgstr, msgstr_plural, "
            "review_status, is_ai_generated, translation_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "b.po",
                "de",
                source_key,
                "beta",
                msgstr_plural_json,
                "reviewed",
                0,
                kdehash.translation_hash(source_key, "de", "beta", {}),
            ),
        )

        workspace_tm._recompute_best_translations(
            conn,
            source_keys=[source_key],
            lang="de",
            review_status_order=workspace_tm.DEFAULT_REVIEW_STATUS_ORDER,
            prefer_human=True,
        )
        row = conn.execute(
            "SELECT file_path FROM best_translations WHERE source_key = ? AND lang = ?",
            (source_key, "de"),
        ).fetchone()
        self.assertEqual(row[0], "b.po")


class TestTmRetrieval(unittest.TestCase):
    def test_exact_only_skips_empty_then_workspace(self):
        workspace_conn = _setup_workspace_db()
        source_key = "source-key"
        msgstr_plural_json = kdehash.canonical_msgstr_plural({})
        workspace_conn.execute(
            "INSERT INTO best_translations (source_key, lang, file_path, msgstr, msgstr_plural, "
            "review_status, is_ai_generated, translation_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                source_key,
                "de",
                "a.po",
                "Hallo",
                msgstr_plural_json,
                "draft",
                0,
                kdehash.translation_hash(source_key, "de", "Hallo", {}),
            ),
        )

        session_tm = {
            (source_key, "de"): {
                "msgstr": "",
                "msgstr_plural": {},
                "review_status": "draft",
                "is_ai_generated": 0,
            }
        }

        result = retrieve_tm.lookup_tm_exact(
            source_key,
            "de",
            has_plural=False,
            session_tm=session_tm,
            workspace_conn=workspace_conn,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.scope, "workspace")
        self.assertEqual(result.msgstr, "Hallo")

    def test_scope_order_prefers_reference(self):
        workspace_conn = _setup_workspace_db()
        reference_conn = _setup_reference_db()
        source_key = "source-key"
        msgstr_plural_json = kdehash.canonical_msgstr_plural({})

        workspace_conn.execute(
            "INSERT INTO best_translations (source_key, lang, file_path, msgstr, msgstr_plural, "
            "review_status, is_ai_generated, translation_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                source_key,
                "de",
                "a.po",
                "Workspace",
                msgstr_plural_json,
                "draft",
                0,
                kdehash.translation_hash(source_key, "de", "Workspace", {}),
            ),
        )
        reference_conn.execute(
            "INSERT INTO best_translations (source_key, lang, file_path, file_sha256, msgstr, "
            "msgstr_plural, review_status, is_ai_generated, translation_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                source_key,
                "de",
                "b.po",
                "sha",
                "Reference",
                msgstr_plural_json,
                "draft",
                0,
                kdehash.translation_hash(source_key, "de", "Reference", {}),
            ),
        )

        config = {"tm": {"lookup_scopes": ["reference", "workspace"]}}
        result = retrieve_tm.lookup_tm_exact(
            source_key,
            "de",
            has_plural=False,
            config=config,
            workspace_conn=workspace_conn,
            reference_conn=reference_conn,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.scope, "reference")
        self.assertEqual(result.msgstr, "Reference")

    def test_plural_exact_only_requires_non_empty(self):
        workspace_conn = _setup_workspace_db()
        source_key = "source-key"
        msgstr_plural_json = kdehash.canonical_msgstr_plural({"0": ""})
        workspace_conn.execute(
            "INSERT INTO best_translations (source_key, lang, file_path, msgstr, msgstr_plural, "
            "review_status, is_ai_generated, translation_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                source_key,
                "de",
                "a.po",
                "",
                msgstr_plural_json,
                "draft",
                0,
                kdehash.translation_hash(source_key, "de", "", {"0": ""}),
            ),
        )

        result = retrieve_tm.lookup_tm_exact(
            source_key,
            "de",
            has_plural=True,
            workspace_conn=workspace_conn,
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
