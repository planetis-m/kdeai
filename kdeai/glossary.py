from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import json
import sqlite3

import spacy

from kdeai import db as kdedb
from kdeai.config import Config
from kdeai import hash as kdehash
from kdeai.constants import DbKind, GLOSSARY_SCHEMA_VERSION
from kdeai.po_utils import parse_msgstr_plural

NORMALIZATION_ID = "kdeai_glossary_norm_v1"
MAX_TERM_TOKENS = 4


@dataclass(frozen=True)
class GlossaryTerm:
    term_key: str
    src_lang: str
    tgt_lang: str
    src_surface: str
    src_lemma_seq: list[str]
    token_count: int
    tgt_primary: str
    tgt_alternates: list[str]
    freq: int
    score: float
    evidence_msgid: str
    evidence_msgstr: str
    file_path: str
    source_key: str
    file_sha256: str


@dataclass(frozen=True)
class GlossaryMatch:
    term: GlossaryTerm
    span_start: int
    span_len: int


class GlossaryNormalizer:
    def __init__(self, nlp, *, normalization_id: str = NORMALIZATION_ID) -> None:
        self._nlp = nlp
        self.normalization_id = normalization_id

    def normalize(self, text: str) -> list[str]:
        doc = self._nlp(text)
        return _lemma_tokens_from_doc(doc)


def _lemma_tokens_from_doc(doc) -> list[str]:
    tokens: list[str] = []
    for token in doc:
        if token.is_space or token.is_punct:
            continue
        lemma = token.lemma_ if token.lemma_ else token.text
        tokens.append(lemma.casefold())
    return tokens


def _has_disallowed_multitoken(doc) -> bool:
    for token in doc:
        if token.is_space or token.is_punct:
            continue
        if token.pos_ in {"DET", "ADP", "PART", "PRON", "CCONJ", "SCONJ"}:
            return True
    return False


def _load_spacy_model(model_name: str):
    return spacy.load(model_name)


def build_normalizer_from_config(config: Config) -> GlossaryNormalizer:
    glossary_cfg = config.prompt.glossary
    nlp = _load_spacy_model(str(glossary_cfg.spacy_model))
    return GlossaryNormalizer(nlp, normalization_id=str(glossary_cfg.normalization_id))


def _canonical_json_list(values: Sequence[str]) -> str:
    return kdehash.canonical_json([str(value) for value in values])


def _parse_json_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return []


def _select_translation(msgstr: str, msgstr_plural: Mapping[str, str]) -> str:
    if msgstr.strip():
        return msgstr
    for key in sorted(msgstr_plural.keys()):
        value = str(msgstr_plural[key])
        if value.strip():
            return value
    return ""


def _iter_reference_rows(conn: sqlite3.Connection):
    query = (
        "SELECT s.source_key, s.msgid, s.msgid_plural, "
        "t.lang, t.msgstr, t.msgstr_plural, "
        "t.file_path, t.file_sha256 "
        "FROM sources s "
        "JOIN best_translations t ON t.source_key = s.source_key"
    )
    return conn.execute(query)


def build_glossary_db(
    reference_conn: sqlite3.Connection,
    *,
    output_path: Path,
    config: Config,
    project_id: str,
    config_hash: str,
    glossary_snapshot_id: int,
) -> Path:
    normalizer = build_normalizer_from_config(config)
    meta = kdedb.read_meta(reference_conn)
    kdedb.validate_meta(
        meta,
        expected_project_id=project_id,
        expected_config_hash=config_hash,
        expected_kind=DbKind.REFERENCE_TM,
    )

    src_lang = config.languages.source
    if not src_lang:
        raise ValueError("languages.source missing")

    source_snapshot_id = int(meta.get("snapshot_id", "0"))
    if source_snapshot_id <= 0:
        raise ValueError("reference snapshot_id missing or invalid")
    if glossary_snapshot_id <= 0:
        raise ValueError("glossary snapshot_id missing or invalid")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    conn = kdedb.connect_writable(output_path)
    conn.executescript(kdedb.GLOSSARY_SCHEMA)

    term_map: dict[tuple[str, str, str], GlossaryTerm] = {}
    for row in _iter_reference_rows(reference_conn):
        source_key = str(row[0])
        msgid = str(row[1])
        lang = str(row[3])
        msgstr = str(row[4])
        msgstr_plural = parse_msgstr_plural(row[5])
        file_path = str(row[6])
        file_sha256 = str(row[7])

        stripped_msgid = msgid.strip()
        if (
            not stripped_msgid
            or stripped_msgid.endswith((".", "?", "!"))
            or "&" in msgid
            or "<" in msgid
            or ">" in msgid
            or "\n" in msgid
        ):
            continue

        tgt_primary = _select_translation(msgstr, msgstr_plural)
        if not tgt_primary:
            continue

        doc = normalizer._nlp(msgid)
        if any(token.is_punct for token in doc):
            continue
        lemma_tokens = _lemma_tokens_from_doc(doc)
        token_count = len(lemma_tokens)
        if (
            not lemma_tokens
            or token_count > MAX_TERM_TOKENS
        ):
            continue
        if token_count > 1 and _has_disallowed_multitoken(doc):
            continue

        term_key = kdehash.term_key(lemma_tokens)
        key = (src_lang, lang, term_key)
        existing = term_map.get(key)
        freq = 1 if existing is None else existing.freq + 1
        score = float(freq)
        term_map[key] = GlossaryTerm(
            term_key=term_key,
            src_lang=src_lang,
            tgt_lang=lang,
            src_surface=msgid,
            src_lemma_seq=list(lemma_tokens),
            token_count=token_count,
            tgt_primary=tgt_primary,
            tgt_alternates=[],
            freq=freq,
            score=score,
            evidence_msgid=msgid,
            evidence_msgstr=tgt_primary,
            file_path=file_path,
            source_key=source_key,
            file_sha256=file_sha256,
        )

    payload = []
    for term in sorted(term_map.values(), key=lambda item: (item.src_lang, item.tgt_lang, item.term_key)):
        payload.append(
            (
                term.term_key,
                term.src_lang,
                term.tgt_lang,
                term.src_surface,
                _canonical_json_list(term.src_lemma_seq),
                term.token_count,
                term.tgt_primary,
                _canonical_json_list(term.tgt_alternates),
                term.freq,
                term.score,
                term.evidence_msgid,
                term.evidence_msgstr,
                term.file_path,
                term.source_key,
                term.file_sha256,
            )
        )

    if payload:
        conn.executemany(
            "INSERT INTO terms ("
            "term_key, src_lang, tgt_lang, src_surface, src_lemma_seq_json, token_count, "
            "tgt_primary, tgt_alternates_json, freq, score, evidence_msgid, evidence_msgstr, "
            "file_path, source_key, file_sha256"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            payload,
        )

    spacy_model_name = str(config.prompt.glossary.spacy_model)
    meta_payload = {
        "schema_version": GLOSSARY_SCHEMA_VERSION,
        "kind": DbKind.GLOSSARY,
        "project_id": project_id,
        "config_hash": config_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": str(glossary_snapshot_id),
        "source_snapshot_kind": DbKind.REFERENCE_TM,
        "source_snapshot_id": str(source_snapshot_id),
        "glossary_src_lang": src_lang,
        "tokenizer_id": f"spacy@{spacy.__version__}:{spacy_model_name}",
        "normalization_id": normalizer.normalization_id,
        "spacy_version": spacy.__version__,
        "spacy_model": spacy_model_name,
        "spacy_model_version": "unknown",
    }
    conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta_payload.items())
    conn.commit()
    conn.close()
    return output_path


def load_terms(
    conn: sqlite3.Connection,
    *,
    src_lang: str,
    tgt_lang: str,
) -> list[GlossaryTerm]:
    rows = conn.execute(
        "SELECT term_key, src_lang, tgt_lang, src_surface, src_lemma_seq_json, token_count, "
        "tgt_primary, tgt_alternates_json, freq, score, evidence_msgid, evidence_msgstr, "
        "file_path, source_key, file_sha256 "
        "FROM terms WHERE src_lang = ? AND tgt_lang = ?",
        (src_lang, tgt_lang),
    ).fetchall()
    terms: list[GlossaryTerm] = []
    for row in rows:
        terms.append(
            GlossaryTerm(
                term_key=str(row[0]),
                src_lang=str(row[1]),
                tgt_lang=str(row[2]),
                src_surface=str(row[3]),
                src_lemma_seq=_parse_json_list(row[4]),
                token_count=int(row[5]),
                tgt_primary=str(row[6]),
                tgt_alternates=_parse_json_list(row[7]),
                freq=int(row[8]),
                score=float(row[9]),
                evidence_msgid=str(row[10]),
                evidence_msgstr=str(row[11]),
                file_path=str(row[12]),
                source_key=str(row[13]),
                file_sha256=str(row[14]),
            )
        )
    return terms


class TrieNode:
    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.terms: list[GlossaryTerm] = []


class GlossaryTrie:
    def __init__(self) -> None:
        self._root = TrieNode()

    def insert(self, term: GlossaryTerm) -> None:
        node = self._root
        for token in term.src_lemma_seq:
            node = node.children.setdefault(token, TrieNode())
        node.terms.append(term)

    def match(self, tokens: Sequence[str]) -> list[GlossaryMatch]:
        matches: list[GlossaryMatch] = []
        for start in range(len(tokens)):
            node = self._root
            for offset in range(len(tokens) - start):
                token = tokens[start + offset]
                node = node.children.get(token)
                if node is None:
                    break
                if node.terms:
                    span_len = offset + 1
                    for term in node.terms:
                        matches.append(
                            GlossaryMatch(term=term, span_start=start, span_len=span_len)
                        )
        return matches


class GlossaryMatcher:
    def __init__(self, *, terms: Iterable[GlossaryTerm], normalizer: GlossaryNormalizer) -> None:
        self._normalizer = normalizer
        self._trie = GlossaryTrie()
        for term in terms:
            self._trie.insert(term)

    def match(self, text: str, *, max_terms: int) -> list[GlossaryMatch]:
        tokens = self._normalizer.normalize(text)
        candidates = self._trie.match(tokens)
        if not candidates:
            return []

        def priority_key(candidate: GlossaryMatch) -> tuple:
            term = candidate.term
            return (
                -candidate.span_len,
                -float(term.score),
                -int(term.freq),
                candidate.span_start,
                term.tgt_primary,
                term.term_key,
            )

        candidates.sort(key=priority_key)
        selected: list[GlossaryMatch] = []
        used_terms: set[str] = set()
        occupied: list[tuple[int, int]] = []

        for candidate in candidates:
            term_key = candidate.term.term_key
            if term_key in used_terms:
                continue
            span = (candidate.span_start, candidate.span_start + candidate.span_len)
            if any(not (span[1] <= start or span[0] >= end) for start, end in occupied):
                continue
            selected.append(candidate)
            used_terms.add(term_key)
            occupied.append(span)
            if len(selected) >= max_terms:
                break

        selected.sort(
            key=lambda item: (item.span_start, -item.span_len, item.term.term_key)
        )
        return selected
