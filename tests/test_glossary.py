import sqlite3
import tempfile
import unittest
from pathlib import Path

import spacy

from kdeai import db as kdedb
from kdeai import hash as kdehash
from kdeai import glossary as kdeglo
from kdeai import po_model
from kdeai.glossary import GlossaryMatcher, GlossaryTerm


class DummyNormalizer:
    def normalize(self, text: str) -> list[str]:
        return [token.casefold() for token in text.split()]


def _load_playground_units(msgids: set[str]) -> tuple[Path, list[po_model.PoUnit]]:
    playground = Path(__file__).resolve().parent / "playground" / "dolphin.po"
    units = po_model.parse_po_path(playground)
    selected = [
        unit for unit in units
        if unit.msgid in msgids and unit.msgstr.strip()
    ]
    if not selected:
        raise AssertionError("Expected sample msgids not found in playground .po")
    return playground, selected


def _setup_reference_db(project_id: str, config_hash: str) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(kdedb.REFERENCE_TM_SCHEMA)
    meta = {
        "schema_version": "1",
        "kind": "reference_tm",
        "project_id": project_id,
        "config_hash": config_hash,
        "created_at": "2024-01-01T00:00:00Z",
        "snapshot_id": "1",
    }
    conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta.items())
    return conn


def _insert_reference_units(
    conn: sqlite3.Connection,
    *,
    units: list[po_model.PoUnit],
    lang: str,
    file_path: str,
    file_sha256: str,
) -> None:
    for unit in units:
        conn.execute(
            "INSERT OR IGNORE INTO sources "
            "(source_key, msgctxt, msgid, msgid_plural, source_text) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                unit.source_key,
                unit.msgctxt,
                unit.msgid,
                unit.msgid_plural,
                unit.source_text,
            ),
        )
        msgstr_plural_json = kdehash.canonical_msgstr_plural(unit.msgstr_plural)
        conn.execute(
            "INSERT INTO best_translations "
            "(source_key, lang, file_path, file_sha256, msgstr, msgstr_plural, "
            "review_status, is_ai_generated, translation_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                unit.source_key,
                lang,
                file_path,
                file_sha256,
                unit.msgstr,
                msgstr_plural_json,
                "reviewed",
                0,
                kdehash.translation_hash(unit.source_key, lang, unit.msgstr, unit.msgstr_plural),
            ),
        )
    conn.commit()


class TestGlossaryMatcher(unittest.TestCase):
    def test_trie_matching_prefers_longer_spans(self):
        terms = [
            GlossaryTerm(
                term_key=kdehash.term_key(["open", "file"]),
                src_lang="en",
                tgt_lang="de",
                src_surface="open file",
                src_lemma_seq=["open", "file"],
                token_count=2,
                tgt_primary="Datei oeffnen",
                tgt_alternates=[],
                freq=1,
                score=1.0,
                evidence_msgid="open file",
                evidence_msgstr="Datei oeffnen",
                file_path="",
                source_key="",
                file_sha256="",
            ),
            GlossaryTerm(
                term_key=kdehash.term_key(["file"]),
                src_lang="en",
                tgt_lang="de",
                src_surface="file",
                src_lemma_seq=["file"],
                token_count=1,
                tgt_primary="Datei",
                tgt_alternates=[],
                freq=1,
                score=1.0,
                evidence_msgid="file",
                evidence_msgstr="Datei",
                file_path="",
                source_key="",
                file_sha256="",
            ),
            GlossaryTerm(
                term_key=kdehash.term_key(["file", "menu"]),
                src_lang="en",
                tgt_lang="de",
                src_surface="file menu",
                src_lemma_seq=["file", "menu"],
                token_count=2,
                tgt_primary="Datei-Menue",
                tgt_alternates=[],
                freq=1,
                score=1.0,
                evidence_msgid="file menu",
                evidence_msgstr="Datei-Menue",
                file_path="",
                source_key="",
                file_sha256="",
            ),
        ]
        matcher = GlossaryMatcher(terms=terms, normalizer=DummyNormalizer())
        matches = matcher.match("open file menu", max_terms=5)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].term.tgt_primary, "Datei oeffnen")
        self.assertEqual(matches[0].span_start, 0)
        self.assertEqual(matches[0].span_len, 2)

    def test_matcher_prefers_score_over_position(self):
        terms = [
            GlossaryTerm(
                term_key=kdehash.term_key(["open", "file"]),
                src_lang="en",
                tgt_lang="de",
                src_surface="open file",
                src_lemma_seq=["open", "file"],
                token_count=2,
                tgt_primary="Datei oeffnen",
                tgt_alternates=[],
                freq=1,
                score=1.0,
                evidence_msgid="open file",
                evidence_msgstr="Datei oeffnen",
                file_path="",
                source_key="",
                file_sha256="",
            ),
            GlossaryTerm(
                term_key=kdehash.term_key(["file", "menu"]),
                src_lang="en",
                tgt_lang="de",
                src_surface="file menu",
                src_lemma_seq=["file", "menu"],
                token_count=2,
                tgt_primary="Datei-Menue",
                tgt_alternates=[],
                freq=1,
                score=2.0,
                evidence_msgid="file menu",
                evidence_msgstr="Datei-Menue",
                file_path="",
                source_key="",
                file_sha256="",
            ),
        ]
        matcher = GlossaryMatcher(terms=terms, normalizer=DummyNormalizer())
        matches = matcher.match("open file menu", max_terms=5)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].term.tgt_primary, "Datei-Menue")
        self.assertEqual(matches[0].span_start, 1)
        self.assertEqual(matches[0].span_len, 2)

    def test_matcher_dedupes_term_key(self):
        term = GlossaryTerm(
            term_key=kdehash.term_key(["open", "file"]),
            src_lang="en",
            tgt_lang="de",
            src_surface="open file",
            src_lemma_seq=["open", "file"],
            token_count=2,
            tgt_primary="Datei oeffnen",
            tgt_alternates=[],
            freq=1,
            score=1.0,
            evidence_msgid="open file",
            evidence_msgstr="Datei oeffnen",
            file_path="",
            source_key="",
            file_sha256="",
        )
        matcher = GlossaryMatcher(terms=[term], normalizer=DummyNormalizer())
        matches = matcher.match("open file open file", max_terms=5)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].span_start, 0)
        self.assertEqual(matches[0].span_len, 2)


class TestGlossaryMining(unittest.TestCase):
    def test_build_glossary_db_from_playground_po(self):
        try:
            spacy.load("en_core_web_sm")
        except OSError:
            self.skipTest("en_core_web_sm not installed; run: python -m spacy download en_core_web_sm")

        msgids = {"Acting as Admin", "Finish", "Empty Trash"}
        playground, units = _load_playground_units(msgids)
        file_sha256 = kdehash.sha256_hex_bytes(playground.read_bytes())

        reference_conn = _setup_reference_db("project-1", "config-1")
        _insert_reference_units(
            reference_conn,
            units=units,
            lang="el",
            file_path=str(playground),
            file_sha256=file_sha256,
        )

        config = {
            "languages": {"source": "en"},
            "prompt": {
                "glossary": {
                    "spacy_model": "en_core_web_sm",
                    "normalization_id": kdeglo.NORMALIZATION_ID,
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "glossary.sqlite"
            kdeglossary_path = kdeglo.build_glossary_db(
                reference_conn,
                output_path=output_path,
                config=config,
                project_id="project-1",
                config_hash="config-1",
            )

            self.assertTrue(kdeglossary_path.exists())
            glossary_conn = sqlite3.connect(str(kdeglossary_path))
            meta = kdedb.read_meta(glossary_conn)
            self.assertEqual(meta["normalization_id"], kdeglo.NORMALIZATION_ID)

            terms = kdeglo.load_terms(glossary_conn, src_lang="en", tgt_lang="el")
            term_by_surface = {term.src_surface: term for term in terms}

            for msgid in {unit.msgid for unit in units}:
                self.assertIn(msgid, term_by_surface)
            for unit in units:
                self.assertEqual(term_by_surface[unit.msgid].tgt_primary, unit.msgstr)

            glossary_conn.close()


if __name__ == "__main__":
    unittest.main()
