from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence, TYPE_CHECKING
import json
import logging
import sqlite3

import polib

from kdeai import state as kdestate
from kdeai.config import Config, EmbeddingPolicy
from kdeai import db as kdedb
from kdeai import examples as kdeexamples
from kdeai import hash as kdehash
from kdeai import locks
from kdeai import po_utils
from kdeai import po_model
from kdeai import retrieve_examples
from kdeai import retrieve_tm
from kdeai import snapshot
from kdeai.constants import (
    ApplyMode,
    AssetMode,
    AssetModeLiteral,
    CacheMode,
    CacheModeLiteral,
    DbKind,
    OverwritePolicy,
    PlanAction,
    PostIndex,
    TmScope,
)
from kdeai.tm_types import SessionTmView

PLAN_FORMAT_VERSION = 1
logger = logging.getLogger(__name__)

EmbeddingFunc = Callable[[Sequence[str]], Sequence[Sequence[float]]]

if TYPE_CHECKING:
    from kdeai import glossary as kdeglo


def build_plan_header(
    *,
    project_id: str,
    config: Config,
    lang: str,
    builder: "PlanBuilder",
    apply_defaults: dict,
) -> dict:
    return {
        "format": PLAN_FORMAT_VERSION,
        "project_id": project_id,
        "config_hash": config.config_hash,
        "lang": lang,
        "marker_flags": sorted(builder.marker_flags),
        "comment_prefixes": sorted(builder.comment_prefixes),
        "ai_flag": builder.ai_flag,
        "placeholder_patterns": list(config.apply.validation_patterns),
        "apply_defaults": {
            "apply_mode": str(apply_defaults.get("apply_mode", ApplyMode.STRICT)),
            "overwrite": str(apply_defaults.get("overwrite", OverwritePolicy.CONSERVATIVE)),
            "post_index": str(apply_defaults.get("post_index", PostIndex.OFF)),
        },
    }


def examples_mode_from_config(
    config: Config, override: AssetModeLiteral | None
) -> AssetModeLiteral:
    return str(override or config.prompt.examples.mode_default or AssetMode.AUTO)


def glossary_mode_from_config(
    config: Config, override: AssetModeLiteral | None
) -> AssetModeLiteral:
    return str(override or config.prompt.glossary.mode_default or AssetMode.AUTO)


def require_embedder(policy: EmbeddingPolicy) -> kdeexamples.EmbeddingFunc:
    from kdeai.embed_client import compute_embedding

    def _embed(texts: Sequence[str]) -> list[list[float]]:
        return [compute_embedding(text, policy=policy) for text in texts]

    return _embed


def _maybe_embedder(
    examples_mode: AssetModeLiteral,
    policy: EmbeddingPolicy,
) -> kdeexamples.EmbeddingFunc | None:
    if examples_mode == AssetMode.OFF:
        return None
    try:
        return require_embedder(policy)
    except Exception:
        if examples_mode == AssetMode.REQUIRED:
            raise
        return None


def resolve_planner_inputs(
    *,
    cache_mode: CacheModeLiteral,
    examples: AssetModeLiteral | None,
    glossary: AssetModeLiteral | None,
    config: Config,
    project_root: Path,
) -> tuple[
    AssetModeLiteral, AssetModeLiteral, kdeexamples.EmbeddingFunc | None, str | None
]:
    resolved_examples_mode = examples_mode_from_config(config, examples)
    resolved_glossary_mode = glossary_mode_from_config(config, glossary)
    if cache_mode == CacheMode.OFF and (
        resolved_examples_mode == AssetMode.REQUIRED
        or resolved_glossary_mode == AssetMode.REQUIRED
    ):
        raise ValueError("cache=off cannot be combined with required examples/glossary")
    if cache_mode == CacheMode.OFF:
        return AssetMode.OFF, AssetMode.OFF, None, None
    embedder = _maybe_embedder(
        resolved_examples_mode,
        config.prompt.examples.embedding_policy,
    )
    sqlite_vector_path = None
    if resolved_examples_mode != AssetMode.OFF:
        candidate = project_root / "vector.so"
        if candidate.exists():
            sqlite_vector_path = str(candidate)
    return resolved_examples_mode, resolved_glossary_mode, embedder, sqlite_vector_path


def _examples_payload(
    examples: Sequence[kdeexamples.ExampleMatch],
) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for example in examples:
        msgstr_plural = getattr(example, "msgstr_plural", {})
        if isinstance(msgstr_plural, Mapping):
            normalized_plural = {str(k): str(v) for k, v in msgstr_plural.items()}
        else:
            normalized_plural = {}
        payload.append(
            {
                "source_text": str(getattr(example, "source_text", "")),
                "msgstr": str(getattr(example, "msgstr", "")),
                "msgstr_plural": normalized_plural,
            }
        )
    return payload


def _glossary_terms_payload(glossary: Sequence[object]) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for match in glossary:
        term = getattr(match, "term", None)
        if term is None:
            continue
        alternates = getattr(term, "tgt_alternates", [])
        if isinstance(alternates, Sequence) and not isinstance(alternates, (str, bytes)):
            alt_values = [str(value) for value in alternates if str(value).strip()]
        else:
            alt_values = []
        payload.append(
            {
                "src_surface": str(getattr(term, "src_surface", "")),
                "tgt_primary": str(getattr(term, "tgt_primary", "")),
                "tgt_alternates": alt_values,
            }
        )
    return payload


@dataclass(frozen=True)
class PlannerAssets:
    workspace_conn: sqlite3.Connection | None
    reference_conn: sqlite3.Connection | None
    examples_db: kdeexamples.ExamplesDb | None
    glossary_conn: sqlite3.Connection | None
    glossary_matcher: object | None
    examples_top_n: int
    glossary_max_terms: int

    def close(self) -> None:
        """Close all database connections."""
        if self.examples_db is not None:
            self.examples_db.conn.close()
        if self.workspace_conn is not None:
            self.workspace_conn.close()
        if self.reference_conn is not None:
            self.reference_conn.close()
        if self.glossary_conn is not None:
            self.glossary_conn.close()

    def __enter__(self) -> "PlannerAssets":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


DraftPlan = dict[str, object]


@dataclass(frozen=True)
class PlannerOptions:
    cache: str = CacheMode.ON
    examples_mode: AssetModeLiteral | None = None
    glossary_mode: AssetModeLiteral | None = None
    embedder: EmbeddingFunc | None = None
    sqlite_vector_path: str | None = None


class PlanBuilder:
    def __init__(
        self,
        *,
        project_root: Path,
        project_id: str,
        config: Config,
        lang: str,
        options: "PlannerOptions | None" = None,
        session_tm: SessionTmView | None = None,
    ) -> None:
        resolved_options = options or PlannerOptions()
        self.config = config
        self.lang = lang
        self.session_tm = session_tm if session_tm is not None else {}
        (
            self.marker_flags,
            self.comment_prefixes,
            self.review_prefix,
            _ai_prefix,
            self.ai_flag,
        ) = po_utils.marker_settings_from_config(config)
        assets, effective_examples_mode, effective_glossary_mode = _build_assets(
            project_root=project_root,
            project_id=project_id,
            config=config,
            lang=lang,
            cache=resolved_options.cache,
            examples_mode=resolved_options.examples_mode,
            glossary_mode=resolved_options.glossary_mode,
            embedder=resolved_options.embedder,
            sqlite_vector_path=resolved_options.sqlite_vector_path,
        )
        self.assets = assets
        self.examples_mode = effective_examples_mode
        self.glossary_mode = effective_glossary_mode
        self.embedder = resolved_options.embedder

    def close(self) -> None:
        self.assets.close()

    def __enter__(self) -> "PlanBuilder":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def build_draft(
        self,
        file_path: str,
        source_entries: Iterable[polib.POEntry],
    ) -> DraftPlan:
        entries_payload: list[dict] = []
        total_entries = 0
        tm_entries = 0
        llm_entries = 0
        marker_flags = po_utils.ensure_ai_flag_in_markers(self.marker_flags, self.ai_flag)
        comment_prefixes = list(self.comment_prefixes)

        for entry in source_entries:
            if entry.obsolete or entry.msgid == "":
                continue
            total_entries += 1
            msgctxt = entry.msgctxt or ""
            msgid = entry.msgid
            msgid_plural = entry.msgid_plural or ""
            base_state_hash = kdestate.entry_state_hash(
                entry,
                lang=self.lang,
                marker_flags=marker_flags,
                comment_prefixes=comment_prefixes,
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
                entries_payload.append(
                    {
                        "msgctxt": msgctxt,
                        "msgid": msgid,
                        "msgid_plural": msgid_plural,
                        "base_state_hash": base_state_hash,
                        "action": PlanAction.COPY_TM,
                        "tag_profile": "tm_copy",
                        "tm_scope": tm_candidate.scope,
                        "translation": {
                            "msgstr": tm_candidate.msgstr,
                            "msgstr_plural": tm_candidate.msgstr_plural,
                        },
                    }
                )
                tm_entries += 1
                continue

            source_text = po_model.source_text_v1(msgctxt, msgid, msgid_plural)
            examples = retrieve_examples.collect_examples(
                examples_db=self.assets.examples_db,
                embedder=self.embedder,
                source_text=source_text,
                top_n=self.assets.examples_top_n,
                lang=self.lang,
                eligibility=self.config.prompt.examples.eligibility,
                review_status_order=self.config.tm.selection.review_status_order,
                required=self.examples_mode == AssetMode.REQUIRED,
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
                    "action": PlanAction.LLM,
                    "tag_profile": "llm",
                    "translation": {"msgstr": "", "msgstr_plural": {}},
                    "examples": _examples_payload(examples),
                    "glossary_terms": _glossary_terms_payload(glossary_matches),
                }
            )
            llm_entries += 1

        logger.debug(
            "plan %s: total_entries=%d tm_entries=%d llm_entries=%d",
            file_path,
            total_entries,
            tm_entries,
            llm_entries,
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
) -> DraftPlan:
    relpath = po_utils.normalize_relpath(project_root, path)
    relpath_key = po_utils.relpath_key(relpath, path_casefold)
    lock_path = locks.per_file_lock_path(
        project_root,
        locks.lock_id(project_id, relpath_key),
    )
    locked = snapshot.locked_read_file(path, lock_path, relpath=relpath)
    po_file = po_utils.load_po_from_bytes(locked.bytes)
    if po_file is None:
        raise ValueError(f"Failed to parse PO file: {relpath}")
    file_draft = builder.build_draft(relpath, po_file)
    file_draft["base_sha256"] = locked.sha256

    entries = file_draft.get("entries")
    if isinstance(entries, list):
        needs_llm = [
            entry
            for entry in entries
            if isinstance(entry, MutableMapping)
            and entry.get("action") == PlanAction.LLM
            and not _has_non_empty_translation(
                entry.get("translation"),
                entry.get("msgid_plural", ""),
            )
        ]
    else:
        needs_llm = []

    if needs_llm:
        from kdeai import llm as kdellm
        from kdeai.llm_provider import configure_dspy

        configure_dspy(builder.config)
        needs_llm = _sorted_entries(needs_llm)
        kdellm.batch_translate(needs_llm, builder.config, target_lang=builder.lang)
        if any(
            not _has_non_empty_translation(entry.get("translation"), entry.get("msgid_plural", ""))
            for entry in needs_llm
        ):
            raise ValueError(
                f"LLM translations required to produce an applyable plan; translation output empty ({relpath})."
            )

    return file_draft


def _has_non_empty_translation(translation: object, msgid_plural: object) -> bool:
    if not isinstance(translation, Mapping):
        return False
    if str(msgid_plural or ""):
        msgstr_plural = translation.get("msgstr_plural")
        if isinstance(msgstr_plural, Mapping):
            return any(str(value).strip() for value in msgstr_plural.values())
        return False
    return bool(str(translation.get("msgstr", "")).strip())


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
    conn.execute(f"PRAGMA busy_timeout = {int(tm_sqlite.busy_timeout_ms.read)}")
    try:
        kdedb.validate_meta_table(
            conn,
            expected_project_id=project_id,
            expected_config_hash=config_hash,
            expected_kind=DbKind.WORKSPACE_TM,
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
        pointer = po_utils.read_json(pointer_path, "reference.current.json")
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
            expected_kind=DbKind.REFERENCE_TM,
        )
    except Exception as exc:
        logger.debug("Reference TM validation failed: %s", exc)
        conn.close()
        return None
    return conn


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
        pointer = po_utils.read_json(pointer_path, "glossary.current.json")
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
            expected_kind=DbKind.GLOSSARY,
            expected_normalization_id=normalization_id,
        )
    except Exception as exc:
        logger.debug("Glossary DB validation failed: %s", exc)
        conn.close()
        return None
    return conn


def _open_tm_connections(
    *,
    project_root: Path,
    project_id: str,
    config_hash: str,
    config: Config,
    cache: str,
) -> tuple[sqlite3.Connection | None, sqlite3.Connection | None]:
    """Open workspace and reference TM connections if cache is enabled."""
    if cache == CacheMode.OFF:
        return None, None
    workspace_conn = _open_workspace_tm(
        project_root, project_id=project_id, config_hash=config_hash, config=config
    )
    reference_conn = _open_reference_tm(
        project_root, project_id=project_id, config_hash=config_hash
    )
    return workspace_conn, reference_conn


def _open_examples(
    *,
    project_root: Path,
    project_id: str,
    config_hash: str,
    embed_policy_hash: str,
    lang: str,
    cache: str,
    examples_mode: AssetModeLiteral,
    examples_scopes: Sequence[str],
    embedder: EmbeddingFunc | None,
    sqlite_vector_path: str | None,
) -> kdeexamples.ExamplesDb | None:
    """Open examples DB if enabled, available, and cache is on."""
    if examples_mode == AssetMode.REQUIRED:
        if embedder is None:
            raise ValueError("examples required but no embedder provided")
        if sqlite_vector_path is None:
            raise ValueError("examples required but sqlite-vector path missing")

    examples_db = None
    if examples_mode != AssetMode.OFF and cache != CacheMode.OFF and embedder is not None:
        for scope in examples_scopes:
            examples_db = retrieve_examples.open_examples_best_effort(
                project_root,
                scope=scope,
                lang=lang,
                project_id=project_id,
                config_hash=config_hash,
                embed_policy_hash=embed_policy_hash,
                sqlite_vector_path=sqlite_vector_path,
                required=examples_mode == AssetMode.REQUIRED,
            )
            if examples_db is not None:
                break

    if examples_mode == AssetMode.REQUIRED and examples_db is None:
        raise ValueError("examples required but unavailable")

    return examples_db


def _open_glossary(
    *,
    project_root: Path,
    project_id: str,
    config_hash: str,
    normalization_id: str,
    config: Config,
    lang: str,
    cache: str,
    glossary_mode: AssetModeLiteral,
    glossary_scopes: Sequence[str],
) -> tuple[sqlite3.Connection | None, object | None]:
    """Open glossary DB and build matcher if enabled."""
    glossary_conn = None
    glossary_matcher: object | None = None
    if glossary_mode != AssetMode.OFF and cache != CacheMode.OFF:
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
            if glossary_mode == AssetMode.REQUIRED:
                raise
            glossary_matcher = None
    elif glossary_mode == AssetMode.REQUIRED:
        raise ValueError("glossary required but unavailable")

    return glossary_conn, glossary_matcher


def _build_assets(
    *,
    project_root: Path,
    project_id: str,
    config: Config,
    lang: str,
    cache: str,
    examples_mode: AssetModeLiteral | None,
    glossary_mode: AssetModeLiteral | None,
    embedder: EmbeddingFunc | None,
    sqlite_vector_path: str | None,
) -> tuple[PlannerAssets, AssetModeLiteral, AssetModeLiteral]:
    config_hash = config.config_hash
    embed_policy_hash = config.embed_policy_hash
    if cache == CacheMode.OFF:
        examples_mode = AssetMode.OFF
        glossary_mode = AssetMode.OFF

    workspace_conn, reference_conn = _open_tm_connections(
        project_root=project_root,
        project_id=project_id,
        config_hash=config_hash,
        config=config,
        cache=cache,
    )

    examples_scopes, examples_top_n, examples_default, _eligibility = (
        config.examples_runtime_settings()
    )
    if examples_mode is None:
        examples_mode = examples_default

    examples_db = _open_examples(
        project_root=project_root,
        project_id=project_id,
        config_hash=config_hash,
        embed_policy_hash=embed_policy_hash,
        lang=lang,
        cache=cache,
        examples_mode=examples_mode,
        examples_scopes=examples_scopes,
        embedder=embedder,
        sqlite_vector_path=sqlite_vector_path,
    )

    glossary_scopes, glossary_max_terms, glossary_default, normalization_id = _glossary_settings(
        config
    )
    if glossary_mode is None:
        glossary_mode = glossary_default

    glossary_conn, glossary_matcher = _open_glossary(
        project_root=project_root,
        project_id=project_id,
        config_hash=config_hash,
        normalization_id=normalization_id,
        config=config,
        lang=lang,
        cache=cache,
        glossary_mode=glossary_mode,
        glossary_scopes=glossary_scopes,
    )

    return (
        PlannerAssets(
            workspace_conn=workspace_conn,
            reference_conn=reference_conn,
            examples_db=examples_db if embedder is not None else None,
            glossary_conn=glossary_conn,
            glossary_matcher=glossary_matcher,
            examples_top_n=examples_top_n,
            glossary_max_terms=glossary_max_terms,
        ),
        examples_mode,
        glossary_mode,
    )


def _collect_glossary(
    *,
    matcher: object | None,
    source_text: str,
    max_terms: int,
) -> list[object]:
    if matcher is None:
        return []
    return matcher.match(source_text, max_terms=max_terms)
