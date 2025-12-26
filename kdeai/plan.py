from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, MutableMapping, Optional, Sequence, TYPE_CHECKING
import json
import logging
import os
import sqlite3
import sys

import polib

from kdeai import apply as kdeapply
from kdeai.config import Config, ExamplesEligibility
from kdeai import db as kdedb
from kdeai import examples as kdeexamples
from kdeai import hash as kdehash
from kdeai import locks
from kdeai import po_utils
from kdeai import po_model
from kdeai import prompt as kdeprompt
from kdeai import retrieve_tm
from kdeai import snapshot
from kdeai.tm_types import SessionTmView

PLAN_FORMAT_VERSION = 1
logger = logging.getLogger(__name__)


EmbeddingFunc = Callable[[Sequence[str]], Sequence[Sequence[float]]]

if TYPE_CHECKING:
    from kdeai import glossary as kdeglo


@dataclass(frozen=True)
class PlannerAssets:
    workspace_conn: sqlite3.Connection | None
    reference_conn: sqlite3.Connection | None
    examples_db: kdeexamples.ExamplesDb | None
    glossary_conn: sqlite3.Connection | None
    glossary_terms: list[object]
    glossary_matcher: object | None
    examples_top_n: int
    glossary_max_terms: int


DraftPlan = dict[str, object]


class PlanBuilder:
    def __init__(
        self,
        *,
        project_root: Path,
        project_id: str,
        config: Config,
        lang: str,
        cache: str = "on",
        examples_mode: str | None = None,
        glossary_mode: str | None = None,
        overwrite: str | None = None,
        session_tm: SessionTmView | None = None,
        embedder: EmbeddingFunc | None = None,
        sqlite_vector_path: str | None = None,
    ) -> None:
        self.config = config
        self.lang = lang
        self.session_tm = session_tm
        (
            self.marker_flags,
            self.comment_prefixes,
            self.review_prefix,
            _ai_prefix,
            self.ai_flag,
        ) = po_utils.marker_settings_from_config(config)
        self.selected_overwrite = str(overwrite or config.apply.overwrite_default)
        self.assets = _build_assets(
            project_root=project_root,
            project_id=project_id,
            config=config,
            lang=lang,
            cache=cache,
            examples_mode=examples_mode,
            glossary_mode=glossary_mode,
            embedder=embedder,
            sqlite_vector_path=sqlite_vector_path,
        )
        self.embedder = embedder
        self._debug_enabled = bool(os.getenv("KDEAI_DEBUG"))

    def _build_tm_tags_and_comments(
        self,
        tm_candidate: retrieve_tm.TmCandidate,
    ) -> tuple[dict[str, object], dict[str, object]]:
        return _tm_tags_and_comments(self.config, tm_candidate)

    def close(self) -> None:
        if self.assets.examples_db is not None:
            self.assets.examples_db.conn.close()
        if self.assets.workspace_conn is not None:
            self.assets.workspace_conn.close()
        if self.assets.reference_conn is not None:
            self.assets.reference_conn.close()
        if self.assets.glossary_conn is not None:
            self.assets.glossary_conn.close()

    def build_draft(
        self,
        file_path: str,
        source_entries: Iterable[polib.POEntry],
    ) -> DraftPlan:
        entries_payload: list[dict] = []
        total_entries = 0
        skipped_overwrite = 0
        tm_entries = 0
        llm_entries = 0

        for entry in source_entries:
            if entry.obsolete or entry.msgid == "":
                continue
            total_entries += 1
            current_non_empty = po_utils.has_non_empty_translation(entry)
            reviewed = po_utils.is_reviewed(entry, self.review_prefix)
            if not po_utils.can_overwrite(current_non_empty, reviewed, self.selected_overwrite):
                skipped_overwrite += 1
                continue
            msgctxt = entry.msgctxt or ""
            msgid = entry.msgid
            msgid_plural = entry.msgid_plural or ""
            base_state_hash = kdeapply.entry_state_hash(
                entry,
                lang=self.lang,
                marker_flags=self.marker_flags,
                comment_prefixes=self.comment_prefixes,
            )
            source_key = po_model.source_key_for(msgctxt, msgid, msgid_plural)
            has_plural = bool(msgid_plural)
            tm_candidate = retrieve_tm.lookup_tm_exact(
                source_key,
                self.lang,
                has_plural=has_plural,
                config=self.config,
                session_tm=self.session_tm,
                workspace_conn=self.assets.workspace_conn,
                reference_conn=self.assets.reference_conn,
            )
            if tm_candidate is not None:
                flags, comments = self._build_tm_tags_and_comments(tm_candidate)
                entries_payload.append(
                    {
                        "msgctxt": msgctxt,
                        "msgid": msgid,
                        "msgid_plural": msgid_plural,
                        "base_state_hash": base_state_hash,
                        "action": "copy_tm",
                        "translation": {
                            "msgstr": tm_candidate.msgstr,
                            "msgstr_plural": tm_candidate.msgstr_plural,
                        },
                        "flags": flags,
                        "comments": comments,
                        "tm_scope": tm_candidate.scope,
                    }
                )
                tm_entries += 1
                continue

            source_text = po_model.source_text_v1(msgctxt, msgid, msgid_plural)
            examples = _collect_examples(
                examples_db=self.assets.examples_db,
                embedder=self.embedder,
                source_text=source_text,
                top_n=self.assets.examples_top_n,
                lang=self.lang,
                eligibility=self.config.prompt.examples.eligibility,
                review_status_order=self.config.tm.selection.review_status_order,
            )
            glossary_matches = _collect_glossary(
                matcher=self.assets.glossary_matcher,
                source_text=msgid,
                max_terms=self.assets.glossary_max_terms,
            )
            entries_payload.append(
                {
                    "msgctxt": msgctxt,
                    "msgid": msgid,
                    "msgid_plural": msgid_plural,
                    "base_state_hash": base_state_hash,
                    "action": "needs_llm",
                    "translation": {"msgstr": "", "msgstr_plural": {}},
                    "examples": kdeprompt.examples_context(examples),
                    "glossary_terms": kdeprompt.glossary_context(glossary_matches),
                }
            )
            llm_entries += 1

        if self._debug_enabled:
            print(
                "[kdeai][debug] plan",
                file_path,
                f"total_entries={total_entries}",
                f"skipped_overwrite={skipped_overwrite}",
                f"tm_entries={tm_entries}",
                f"llm_entries={llm_entries}",
                file=sys.stderr,
            )

        return {
            "file_path": file_path,
            "entries": entries_payload,
        }


def generate_plan_for_file(
    *,
    project_root: Path,
    project_id: str,
    path: Path,
    path_casefold: bool,
    builder: PlanBuilder,
    config: Config,
    run_llm: bool = False,
) -> DraftPlan:
    relpath = _normalize_relpath(project_root, path)
    relpath_key = _relpath_key(relpath, path_casefold)
    lock_path = locks.per_file_lock_path(
        project_root,
        locks.lock_id(project_id, relpath_key),
    )
    locked = snapshot.locked_read_file(path, lock_path)
    po_file = po_utils.load_po_from_bytes(locked.bytes)
    file_draft = builder.build_draft(relpath, po_file)
    file_draft["base_sha256"] = locked.sha256

    entries = file_draft.get("entries")
    if isinstance(entries, list):
        needs_llm = [
            entry
            for entry in entries
            if isinstance(entry, MutableMapping) and entry.get("action") == "needs_llm"
        ]
    else:
        needs_llm = []

    if run_llm and needs_llm:
        from kdeai import llm as kdellm

        kdellm.batch_translate(needs_llm, config, target_lang=builder.lang)

    return file_draft


def _sorted_entries(entries: list[dict]) -> list[dict]:
    return sorted(
        entries,
        key=lambda entry: (
            str(entry.get("msgctxt", "")),
            str(entry.get("msgid", "")),
            str(entry.get("msgid_plural", "")),
        ),
    )


def _sorted_files(files: list[dict]) -> list[dict]:
    return sorted(files, key=lambda item: str(item.get("file_path", "")))


def normalize_plan(plan: dict) -> dict:
    normalized = dict(plan)
    files = [dict(item) for item in plan.get("files", [])]
    for item in files:
        item["entries"] = _sorted_entries(list(item.get("entries", [])))
    normalized["files"] = _sorted_files(files)
    return normalized


def compute_plan_id(plan_without_plan_id: dict) -> str:
    canonical = kdehash.canonical_json(plan_without_plan_id)
    return kdehash.sha256_hex_text(canonical)


def finalize_plan(plan: dict) -> dict:
    normalized = normalize_plan(plan)
    plan_without_id = {key: value for key, value in normalized.items() if key != "plan_id"}
    normalized["plan_id"] = compute_plan_id(plan_without_id)
    return normalized


def render_plan_json(plan: dict) -> str:
    finalized = finalize_plan(plan)
    return kdehash.canonical_json(finalized)


def write_plan(path: Path, plan: dict) -> dict:
    finalized = finalize_plan(plan)
    path.write_text(kdehash.canonical_json(finalized), encoding="utf-8")
    return finalized


def load_plan(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


_iter_po_paths = po_utils.iter_po_paths
_normalize_relpath = po_utils.normalize_relpath
_relpath_key = po_utils.relpath_key
_read_json = po_utils.read_json


def _tm_tags_and_comments(
    config: Config,
    tm_candidate: retrieve_tm.TmCandidate,
) -> tuple[dict[str, object], dict[str, object]]:
    tm_cfg = config.apply.tagging.tm_copy
    add_flags = list(tm_cfg.add_flags)
    add_ai_flag = bool(tm_cfg.add_ai_flag)
    comment_prefix_key = str(tm_cfg.comment_prefix_key or "tm")
    comment_prefixes = config.markers.comment_prefixes
    comment_prefix = getattr(comment_prefixes, comment_prefix_key, comment_prefixes.tm)
    ai_flag = config.markers.ai_flag

    if add_ai_flag:
        add_flags.append(ai_flag)

    ensured_line = f"{comment_prefix} copied_from={tm_candidate.scope}"
    flags = {"add": add_flags, "remove": []}
    comments = {
        "remove_prefixes": [comment_prefix],
        "ensure_lines": [ensured_line],
        "append": "",
    }
    return flags, comments


def _examples_settings(config: Config) -> tuple[list[str], int, str]:
    examples_cfg = config.prompt.examples
    return (
        list(examples_cfg.lookup_scopes),
        int(examples_cfg.top_n),
        str(examples_cfg.mode_default),
    )


def _glossary_settings(config: Config) -> tuple[list[str], int, str, str]:
    glossary_cfg = config.prompt.glossary
    return (
        list(glossary_cfg.lookup_scopes),
        int(glossary_cfg.max_terms),
        str(glossary_cfg.mode_default),
        str(glossary_cfg.normalization_id),
    )


def _open_workspace_tm(
    project_root: Path,
    *,
    project_id: str,
    config_hash: str,
    config: Config,
) -> sqlite3.Connection | None:
    db_path = project_root / ".kdeai" / "cache" / "workspace.tm.sqlite"
    if not db_path.exists():
        return None
    conn = kdedb.connect_readonly(db_path)
    tm_sqlite = config.sqlite.workspace_tm
    conn.execute(f"PRAGMA synchronous = {tm_sqlite.synchronous}")
    conn.execute(f"PRAGMA busy_timeout = {int(tm_sqlite.busy_timeout_ms.read)}")
    try:
        kdedb.validate_meta_table(
            conn,
            expected_project_id=project_id,
            expected_config_hash=config_hash,
            expected_kind="workspace_tm",
        )
    except Exception as exc:
        logger.debug("Workspace TM validation failed: %s", exc)
        conn.close()
        return None
    return conn


def _open_reference_tm(
    project_root: Path,
    *,
    project_id: str,
    config_hash: str,
) -> sqlite3.Connection | None:
    pointer_path = project_root / ".kdeai" / "cache" / "reference" / "reference.current.json"
    if not pointer_path.exists():
        return None
    try:
        pointer = _read_json(pointer_path, "reference.current.json")
    except Exception as exc:
        logger.debug("Reference pointer read failed: %s", exc)
        return None
    db_file = pointer.get("db_file")
    if not db_file:
        return None
    db_path = pointer_path.parent / str(db_file)
    if not db_path.exists():
        return None
    conn = kdedb.connect_readonly(db_path)
    try:
        kdedb.validate_meta_table(
            conn,
            expected_project_id=project_id,
            expected_config_hash=config_hash,
            expected_kind="reference_tm",
        )
    except Exception as exc:
        logger.debug("Reference TM validation failed: %s", exc)
        conn.close()
        return None
    return conn


def _examples_pointer_path(
    project_root: Path,
    *,
    scope: str,
    lang: str,
) -> Path:
    return (
        project_root
        / ".kdeai"
        / "cache"
        / "examples"
        / scope
        / f"examples.{scope}.{lang}.current.json"
    )


def _open_examples_db(
    project_root: Path,
    *,
    scope: str,
    lang: str,
    project_id: str,
    config_hash: str,
    embed_policy_hash: str,
) -> kdeexamples.ExamplesDb | None:
    pointer_path = _examples_pointer_path(project_root, scope=scope, lang=lang)
    if not pointer_path.exists():
        return None
    try:
        pointer = _read_json(pointer_path, f"examples {scope} pointer")
    except Exception as exc:
        logger.debug("Examples pointer read failed: %s", exc)
        return None
    db_file = pointer.get("db_file")
    if not db_file:
        return None
    db_path = pointer_path.parent / str(db_file)
    if not db_path.exists():
        return None
    try:
        return kdeexamples.open_examples_db(
            db_path,
            project_id=project_id,
            config_hash=config_hash,
            embed_policy_hash=embed_policy_hash,
        )
    except Exception as exc:
        logger.debug("Examples DB open failed: %s", exc)
        return None


def _open_glossary_db(
    project_root: Path,
    *,
    project_id: str,
    config_hash: str,
    normalization_id: str,
) -> sqlite3.Connection | None:
    pointer_path = project_root / ".kdeai" / "cache" / "glossary" / "glossary.current.json"
    if not pointer_path.exists():
        return None
    try:
        pointer = _read_json(pointer_path, "glossary.current.json")
    except Exception as exc:
        logger.debug("Glossary pointer read failed: %s", exc)
        return None
    db_file = pointer.get("db_file")
    if not db_file:
        return None
    db_path = pointer_path.parent / str(db_file)
    if not db_path.exists():
        return None
    conn = kdedb.connect_readonly(db_path)
    try:
        kdedb.validate_meta_table(
            conn,
            expected_project_id=project_id,
            expected_config_hash=config_hash,
            expected_kind="glossary",
            expected_normalization_id=normalization_id,
        )
    except Exception as exc:
        logger.debug("Glossary DB validation failed: %s", exc)
        conn.close()
        return None
    return conn


def _build_assets(
    *,
    project_root: Path,
    project_id: str,
    config: Config,
    lang: str,
    cache: str,
    examples_mode: str | None,
    glossary_mode: str | None,
    embedder: EmbeddingFunc | None,
    sqlite_vector_path: str | None,
) -> PlannerAssets:
    config_hash = config.config_hash
    embed_policy_hash = config.embed_policy_hash
    if cache == "off":
        examples_mode = "off"
        glossary_mode = "off"

    workspace_conn = None
    reference_conn = None
    if cache != "off":
        workspace_conn = _open_workspace_tm(
            project_root, project_id=project_id, config_hash=config_hash, config=config
        )
        reference_conn = _open_reference_tm(
            project_root, project_id=project_id, config_hash=config_hash
        )

    examples_scopes, examples_top_n, examples_default = _examples_settings(config)
    if examples_mode is None:
        examples_mode = examples_default

    examples_db = None
    if examples_mode != "off" and cache != "off":
        for scope in examples_scopes:
            examples_db = _open_examples_db(
                project_root,
                scope=scope,
                lang=lang,
                project_id=project_id,
                config_hash=config_hash,
                embed_policy_hash=embed_policy_hash,
            )
            if examples_db is not None:
                break

    if examples_mode == "required" and (examples_db is None or embedder is None):
        raise ValueError("examples required but unavailable")

    if examples_db is not None:
        if sqlite_vector_path:
            try:
                kdedb.enable_sqlite_vector(
                    examples_db.conn, extension_path=sqlite_vector_path
                )
            except Exception:
                if examples_mode == "required":
                    raise
                examples_db.conn.close()
                examples_db = None
        else:
            if examples_mode == "required":
                raise ValueError("examples required but sqlite-vector unavailable")
            examples_db.conn.close()
            examples_db = None

    glossary_scopes, glossary_max_terms, glossary_default, normalization_id = _glossary_settings(
        config
    )
    if glossary_mode is None:
        glossary_mode = glossary_default

    glossary_terms: list[object] = []
    glossary_matcher: object | None = None
    glossary_conn = None
    if glossary_mode != "off" and cache != "off":
        if "reference" in glossary_scopes:
            glossary_conn = _open_glossary_db(
                project_root,
                project_id=project_id,
                config_hash=config_hash,
                normalization_id=normalization_id,
            )
    if glossary_conn is not None:
        try:
            from kdeai import glossary as kdeglo

            glossary_terms = kdeglo.load_terms(
                glossary_conn, src_lang=config.languages.source, tgt_lang=lang
            )
            normalizer = kdeglo.build_normalizer_from_config(config)
            glossary_matcher = kdeglo.GlossaryMatcher(
                terms=glossary_terms, normalizer=normalizer
            )
        except Exception:
            glossary_conn.close()
            glossary_conn = None
            if glossary_mode == "required":
                raise
            glossary_terms = []
            glossary_matcher = None
    elif glossary_mode == "required":
        raise ValueError("glossary required but unavailable")

    if embedder is None:
        if examples_mode == "required":
            raise ValueError("examples required but no embedder provided")
        if examples_db is not None:
            examples_db.conn.close()
            examples_db = None

    return PlannerAssets(
        workspace_conn=workspace_conn,
        reference_conn=reference_conn,
        examples_db=examples_db if embedder is not None else None,
        glossary_conn=glossary_conn,
        glossary_terms=glossary_terms,
        glossary_matcher=glossary_matcher,
        examples_top_n=examples_top_n,
        glossary_max_terms=glossary_max_terms,
    )


def _collect_examples(
    *,
    examples_db: kdeexamples.ExamplesDb | None,
    embedder: EmbeddingFunc | None,
    source_text: str,
    top_n: int,
    lang: str,
    eligibility: ExamplesEligibility,
    review_status_order: Sequence[str],
) -> list[kdeexamples.ExampleMatch]:
    if examples_db is None or embedder is None:
        return []
    embedding = embedder([source_text])[0]
    try:
        return kdeexamples.query_examples(
            examples_db,
            query_embedding=embedding,
            top_n=top_n,
            lang=lang,
            eligibility=eligibility,
            review_status_order=review_status_order,
        )
    except Exception:
        return []


def _collect_glossary(
    *,
    matcher: object | None,
    source_text: str,
    max_terms: int,
) -> list[object]:
    if matcher is None:
        return []
    return matcher.match(source_text, max_terms=max_terms)
