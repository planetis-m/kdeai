import unittest

from kdeai import hash as kdehash
from kdeai.glossary import GlossaryMatcher, GlossaryTerm


class DummyNormalizer:
    def normalize(self, text: str) -> list[str]:
        return [token.casefold() for token in text.split()]


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
