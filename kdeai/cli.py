from __future__ import annotations

from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, NoReturn
import os
from datetime import datetime, timezone

import portalocker
import typer

from kdeai import apply as kdeapply
from kdeai.config import (
    Config,
    examples_embed_policy,
    glossary_normalization_id,
    workspace_tm_settings,
)
from kdeai import db as kdedb
from kdeai import doctor as kdedoctor
from kdeai import examples as kdeexamples
from kdeai import gc as kdegc
from kdeai import glossary as kdeglo
from kdeai import hash as kdehash
from kdeai import locks
from kdeai import plan as kdeplan
from kdeai import po_utils
from kdeai import reference as kderef
from kdeai import snapshot
from kdeai import workspace_tm
from kdeai.constants import (
    ApplyModeLiteral,
    AssetModeLiteral,
    CacheMode,
    CacheModeLiteral,
    DbKind,
    ExamplesScope,
    OverwritePolicyLiteral,
    PostIndex,
    PostIndexLiteral,
    TmScope,
    WORKSPACE_TM_SCHEMA_VERSION,
)
from kdeai.project import Project


def _exit_with_error(message: str, code: int = 1) -> NoReturn:
    """Print error message to stderr and exit with given code."""
    typer.secho(f"Error: {message}", fg="red", err=True)
    raise typer.Exit(code)


app = typer.Typer(add_completion=False, no_args_is_help=True)
reference_app = typer.Typer(no_args_is_help=True)
examples_app = typer.Typer(no_args_is_help=True)
glossary_app = typer.Typer(no_args_is_help=True)
app.add_typer(reference_app, name="reference")
app.add_typer(examples_app, name="examples")
app.add_typer(glossary_app, name="glossary")


@dataclass
class CLIState:
    root: Path
    project: Project | None = None
    config: Config | None = None
    project_id: str | None = None
    path_casefold: bool = os.name == "nt"
    run_lock_cm: object | None = None


def _project_dir(project_root: Path) -> Path:
    return project_root / ".kdeai"


def _cache_path(project_root: Path, *parts: str) -> Path:
    return project_root / ".kdeai" / "cache" / Path(*parts)


@contextmanager
def _atomic_cache_update(project_root: Path, category: str, filename: str):
    cache_dir = _cache_path(project_root, *Path(category).parts)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pointer_path = cache_dir / filename
    tmp_path = pointer_path.with_suffix(pointer_path.suffix + ".tmp")
    try:
        yield pointer_path, tmp_path
        os.replace(tmp_path, pointer_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _glossary_is_current(
    *,
    pointer_path: Path,
    reference_snapshot_id: int,
    project_id: str,
    config: Config,
) -> bool:
    """Check whether the glossary cache is already current."""
    if not pointer_path.exists():
        return False
    try:
        pointer = po_utils.read_json(pointer_path, pointer_path.name)
        db_file = str(pointer.get("db_file", ""))
        pointer_snapshot_id = int(pointer.get("snapshot_id", 0))
        if not db_file or pointer_snapshot_id <= 0:
            return False
        db_path = pointer_path.parent / db_file
        if not db_path.exists():
            return False
        with closing(kdedb.connect_readonly(db_path)) as conn:
            kdedb.validate_meta_table(
                conn,
                expected_project_id=project_id,
                expected_config_hash=config.config_hash,
                expected_kind=DbKind.GLOSSARY,
                expected_normalization_id=glossary_normalization_id(config),
            )
            meta = kdedb.read_meta(conn)
        meta_snapshot_id = int(meta.get("snapshot_id", "0"))
        source_snapshot_id = int(meta.get("source_snapshot_id", "0"))
        if meta_snapshot_id != pointer_snapshot_id:
            return False
        if meta.get("source_snapshot_kind") != DbKind.REFERENCE_TM:
            return False
        if source_snapshot_id != reference_snapshot_id:
            return False
        return True
    except Exception:
        return False


def _acquire_run_lock(project_root: Path, ctx: typer.Context, state: CLIState) -> None:
    lock_cm = locks.acquire_run_lock(project_root)
    try:
        lock_cm.__enter__()
    except portalocker.exceptions.LockException:
        _exit_with_error("Global run lock is held; another KDEAI process is running.")
    state.run_lock_cm = lock_cm
    ctx.call_on_close(lambda: lock_cm.__exit__(None, None, None))


def _load_project(project_root: Path, command_name: str) -> Project:
    """Load project or exit with descriptive error."""
    try:
        return Project.load_or_init(project_root)
    except Exception as exc:
        _exit_with_error(f"{command_name} failed: {exc}")


def _require_state(ctx: typer.Context) -> CLIState:
    state = ctx.obj
    if isinstance(state, CLIState):
        return state
    return CLIState(root=Path.cwd())


def _ensure_project(state: CLIState, command_name: str) -> Project:
    if state.project is not None:
        return state.project
    project = _load_project(state.root, command_name)
    state.project = project
    state.config = project.config
    state.project_id = str(project.project_data["project_id"])
    state.path_casefold = bool(project.project_data.get("path_casefold", os.name == "nt"))
    return project


def get_planner_context(
    *,
    state: CLIState,
    command_name: str,
    cache: Optional[CacheModeLiteral],
    cache_write: Optional[CacheModeLiteral],
    examples: Optional[AssetModeLiteral],
    glossary: Optional[AssetModeLiteral],
) -> tuple[Config, str, kdeplan.PlannerOptions, bool]:
    _ensure_project(state, command_name)
    config = state.config
    project_id = state.project_id
    project_root = state.root
    cache_mode = cache or CacheMode.ON
    cache_write_flag = cache_write or CacheMode.ON
    if cache_write_flag == CacheMode.OFF:
        _note_cache_write_noop(command_name.lower())
    try:
        resolved_examples_mode, resolved_glossary_mode, embedder, sqlite_vector_path = (
            kdeplan.resolve_planner_inputs(
                cache_mode=cache_mode,
                examples=examples,
                glossary=glossary,
                config=config,
                project_root=project_root,
            )
        )
    except ValueError as exc:
        _exit_with_error(str(exc))
    options = kdeplan.PlannerOptions(
        cache=cache_mode,
        examples_mode=resolved_examples_mode,
        glossary_mode=resolved_glossary_mode,
        embedder=embedder,
        sqlite_vector_path=sqlite_vector_path,
    )
    path_casefold = bool(state.path_casefold)
    return config, project_id, options, path_casefold


def _next_pointer_id(pointer_path: Path, key: str) -> int:
    if not pointer_path.exists():
        return 1
    try:
        payload = po_utils.read_json(pointer_path, pointer_path.name)
        pointer_id = int(payload.get(key, 0))
    except (OSError, ValueError):
        pointer_id = 0
    return pointer_id + 1


def _sqlite_vector_path(project_root: Path) -> str | None:
    candidate = project_root / "vector.so"
    if candidate.exists():
        return str(candidate)
    return None


def _note_cache_write_noop(command_name: str) -> None:
    typer.secho(
        f"Note: {command_name} does not write cache; --cache-write has no effect.",
        err=True,
    )


def _apply_defaults_from_config(
    config: Config,
    *,
    overwrite_default: str | None = None,
    apply_mode_default: str | None = None,
) -> dict:
    return {
        "apply_mode": str(apply_mode_default or config.apply.mode_default),
        "overwrite": str(overwrite_default or config.apply.overwrite_default),
        "post_index": PostIndex.OFF,
    }


def _ensure_workspace_db(
    project_root: Path,
    *,
    project_id: str,
    config_hash: str,
    config: Config,
):
    db_path = _cache_path(project_root, "workspace.tm.sqlite")
    created = not db_path.exists()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    busy_timeout_ms, synchronous = workspace_tm_settings(config)
    conn = kdedb.connect_workspace_tm(
        db_path,
        busy_timeout_ms=busy_timeout_ms,
        synchronous=synchronous,
    )
    try:
        if created:
            conn.executescript(kdedb.WORKSPACE_TM_SCHEMA)
            meta = {
                "schema_version": WORKSPACE_TM_SCHEMA_VERSION,
                "kind": DbKind.WORKSPACE_TM,
                "project_id": project_id,
                "config_hash": config_hash,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta.items())
            conn.commit()
        else:
            meta = kdedb.read_meta(conn)
            kdedb.validate_meta(
                meta,
                expected_project_id=project_id,
                expected_config_hash=config_hash,
                expected_kind=DbKind.WORKSPACE_TM,
            )
    except Exception:
        conn.close()
        raise
    return conn


@contextmanager
def _optional_workspace_db(
    project_root: Path,
    *,
    project_id: str,
    config_hash: str,
    config: Config,
    enabled: bool,
):
    """Yield workspace connection if enabled and available, else None."""
    if not enabled:
        yield None
        return

    conn = None
    try:
        conn = _ensure_workspace_db(
            project_root,
            project_id=project_id,
            config_hash=config_hash,
            config=config,
        )
        yield conn
    except Exception as exc:
        typer.secho(f"Warning: workspace DB unavailable ({exc}).", err=True)
        yield None
    finally:
        if conn is not None:
            conn.close()


@app.callback()
def main(ctx: typer.Context) -> None:
    project_root = Path.cwd()
    state = CLIState(root=project_root)
    ctx.obj = state
    _acquire_run_lock(project_root, ctx, state)
    if ctx.invoked_subcommand in {None, "init"}:
        return
    _ensure_project(state, "CLI")


@app.command()
def init(ctx: typer.Context) -> None:
    state = _require_state(ctx)
    project_root = state.root
    project_data = Project.ensure_project_data(project_root)
    typer.echo(f"Initialized project {project_data['project_id']} at {_project_dir(project_root)}.")
    config_path = project_root / ".kdeai" / "config.json"
    if not config_path.exists():
        typer.secho("Missing .kdeai/config.json. Create it before planning or indexing.", err=True)


@app.command()
def plan(
    ctx: typer.Context,
    paths: Optional[list[Path]] = typer.Argument(None),
    lang: str = typer.Option(..., "--lang"),
    out: Optional[Path] = typer.Option(None, "--out"),
    cache: Optional[CacheModeLiteral] = typer.Option(None, "--cache"),
    cache_write: Optional[CacheModeLiteral] = typer.Option(None, "--cache-write"),
    examples: Optional[AssetModeLiteral] = typer.Option(None, "--examples"),
    glossary: Optional[AssetModeLiteral] = typer.Option(None, "--glossary"),
) -> None:
    state = _require_state(ctx)
    project_root = state.root
    config, project_id, options, path_casefold = get_planner_context(
        state=state,
        command_name="Plan",
        cache=cache,
        cache_write=cache_write,
        examples=examples,
        glossary=glossary,
    )
    files_payload: list[dict] = []
    plan_payload: dict | None = None
    with kdeplan.PlanBuilder(
        project_root=project_root,
        project_id=project_id,
        config=config,
        lang=lang,
        options=options,
        session_tm={},
    ) as builder:
        for path in po_utils.iter_po_paths(project_root, paths):
            file_draft = kdeplan.generate_plan_for_file(
                project_root=project_root,
                project_id=project_id,
                path=path,
                path_casefold=path_casefold,
                builder=builder,
            )
            files_payload.append(file_draft)
        files_payload.sort(key=lambda item: str(item.get("file_path", "")))
        apply_cfg = _apply_defaults_from_config(config)
        plan_payload = kdeplan.build_plan_header(
            project_id=project_id,
            config=config,
            lang=lang,
            builder=builder,
            apply_defaults=apply_cfg,
        )
        plan_payload["files"] = files_payload

    if plan_payload is None:
        _exit_with_error("Plan failed.")

    if out is None:
        typer.echo(kdeplan.render_plan_json(plan_payload))
        return
    finalized = kdeplan.write_plan(out, plan_payload)
    typer.echo(f"Wrote plan {finalized['plan_id']} to {out}.")


@app.command()
def apply(
    ctx: typer.Context,
    plan_path: Path = typer.Argument(...),
    apply_mode: Optional[ApplyModeLiteral] = typer.Option(None, "--apply-mode"),
    overwrite: Optional[OverwritePolicyLiteral] = typer.Option(None, "--overwrite"),
    post_index: PostIndexLiteral = typer.Option(PostIndex.OFF, "--post-index"),
) -> None:
    state = _require_state(ctx)
    project_root = state.root
    plan = kdeplan.load_plan(plan_path)
    post_index_flag = post_index == PostIndex.ON
    _ensure_project(state, "Apply")
    config = state.config
    project_id = state.project_id
    path_casefold = bool(state.path_casefold)

    with _optional_workspace_db(
        project_root,
        project_id=project_id,
        config_hash=config.config_hash,
        config=config,
        enabled=post_index_flag,
    ) as workspace_conn:
        result = kdeapply.apply_plan(
            plan,
            project_root=project_root,
            project_id=project_id,
            path_casefold=path_casefold,
            config=config,
            apply_mode=apply_mode,
            overwrite=overwrite,
            post_index=post_index_flag,
            workspace_conn=workspace_conn,
        )

    if result.errors:
        error_message = "\nError: ".join(result.errors)
        _exit_with_error(error_message)

    if result.warnings:
        for warning in result.warnings:
            typer.secho(f"Warning: {warning}", err=True)

    typer.echo(
        f"Applied {result.entries_applied} entries "
        f"({len(result.files_written)} files written, {len(result.files_skipped)} skipped)."
    )


@app.command()
def translate(
    ctx: typer.Context,
    paths: Optional[list[Path]] = typer.Argument(None),
    lang: str = typer.Option(..., "--lang"),
    out: Optional[Path] = typer.Option(None, "--out"),
    apply_mode: Optional[ApplyModeLiteral] = typer.Option(None, "--apply-mode"),
    overwrite: Optional[OverwritePolicyLiteral] = typer.Option(None, "--overwrite"),
    cache: Optional[CacheModeLiteral] = typer.Option(None, "--cache"),
    cache_write: Optional[CacheModeLiteral] = typer.Option(None, "--cache-write"),
    examples: Optional[AssetModeLiteral] = typer.Option(None, "--examples"),
    glossary: Optional[AssetModeLiteral] = typer.Option(None, "--glossary"),
) -> None:
    state = _require_state(ctx)
    project_root = state.root
    config, project_id, options, path_casefold = get_planner_context(
        state=state,
        command_name="Translate",
        cache=cache,
        cache_write=cache_write,
        examples=examples,
        glossary=glossary,
    )

    apply_defaults = _apply_defaults_from_config(
        config,
        overwrite_default=overwrite,
        apply_mode_default=apply_mode,
    )

    files_payload: list[dict] = []
    total_entries_applied = 0
    files_written: list[str] = []
    files_skipped: list[str] = []
    with kdeplan.PlanBuilder(
        project_root=project_root,
        project_id=project_id,
        config=config,
        lang=lang,
        options=options,
        session_tm={},
    ) as builder:
        session_tm = builder.session_tm
        plan_header = kdeplan.build_plan_header(
            project_id=project_id,
            config=config,
            lang=lang,
            builder=builder,
            apply_defaults=apply_defaults,
        )
        for path in po_utils.iter_po_paths(project_root, paths):
            file_plan = kdeplan.generate_plan_for_file(
                project_root=project_root,
                project_id=project_id,
                path=path,
                path_casefold=path_casefold,
                builder=builder,
            )
            files_payload.append(file_plan)
            per_file_plan = dict(plan_header)
            per_file_plan["files"] = [file_plan]
            result = kdeapply.apply_plan(
                per_file_plan,
                project_root=project_root,
                project_id=project_id,
                path_casefold=path_casefold,
                config=config,
                apply_mode=apply_mode,
                overwrite=overwrite,
                session_tm=session_tm,
            )

            if result.errors:
                if out is not None:
                    combined_plan = dict(plan_header)
                    files_payload.sort(key=lambda item: str(item.get("file_path", "")))
                    combined_plan["files"] = files_payload
                    kdeplan.write_plan(out, combined_plan)
                error_message = "\nError: ".join(result.errors)
                _exit_with_error(error_message)

            if result.warnings:
                for warning in result.warnings:
                    typer.secho(f"Warning: {warning}", err=True)

            total_entries_applied += result.entries_applied
            files_written.extend(result.files_written)
            files_skipped.extend(result.files_skipped)

        if out is not None:
            combined_plan = dict(plan_header)
            files_payload.sort(key=lambda item: str(item.get("file_path", "")))
            combined_plan["files"] = files_payload
            kdeplan.write_plan(out, combined_plan)

    typer.echo(
        f"Applied {total_entries_applied} entries "
        f"({len(files_written)} files written, {len(files_skipped)} skipped)."
    )


@app.command()
def index(
    ctx: typer.Context,
    paths: Optional[list[Path]] = typer.Argument(None),
    strict: bool = typer.Option(False, "--strict"),
) -> None:
    state = _require_state(ctx)
    project_root = state.root
    _ensure_project(state, "Index")
    config = state.config
    project_id = state.project_id
    path_casefold = bool(state.path_casefold)
    with closing(
        _ensure_workspace_db(
            project_root,
            project_id=project_id,
            config_hash=config.config_hash,
            config=config,
        )
    ) as conn:
        for path in po_utils.iter_po_paths(project_root, paths):
            relpath = po_utils.normalize_relpath(project_root, path)
            relpath_key = po_utils.relpath_key(relpath, path_casefold)
            lock_path = locks.per_file_lock_path(
                project_root,
                locks.lock_id(project_id, relpath_key),
            )
            try:
                locked = snapshot.locked_read_file(path, lock_path, relpath=relpath)
                po_file = po_utils.load_po_from_bytes(locked.bytes)
                lang = po_utils.get_po_language(po_file, config)
                if not lang:
                    raise ValueError(f"unable to infer language for {relpath}")
                workspace_tm.index_file_snapshot_tm(
                    conn,
                    file_path=relpath,
                    lang=lang,
                    bytes=locked.bytes,
                    sha256=locked.sha256,
                    mtime_ns=locked.mtime_ns,
                    size=locked.size,
                    config=config,
                )
            except Exception as exc:
                message = f"{relpath}: {exc}"
                if strict:
                    _exit_with_error(message)
                typer.secho(f"Warning: {message}", err=True)

    typer.echo("Workspace TM index updated.")


@reference_app.command("build")
def reference_build(
    ctx: typer.Context,
    paths: Optional[list[Path]] = typer.Argument(None),
    label: Optional[str] = typer.Option(None, "--label"),
) -> None:
    state = _require_state(ctx)
    project_root = state.root
    _ensure_project(state, "Reference build")
    config = state.config
    project_id = state.project_id
    try:
        snapshot = kderef.build_reference_snapshot(
            project_root,
            project_id=project_id,
            path_casefold=bool(state.path_casefold),
            config=config,
            config_hash=config.config_hash,
            paths=paths,
            label=label,
        )
    except Exception as exc:
        _exit_with_error(f"Reference build failed: {exc}")
    pointer_payload = {
        "snapshot_id": snapshot.snapshot_id,
        "db_file": snapshot.db_path.name,
        "created_at": snapshot.created_at,
    }
    with _atomic_cache_update(
        project_root,
        "reference",
        "reference.current.json",
    ) as (_pointer_path, tmp_path):
        tmp_path.write_text(kdehash.canonical_json(pointer_payload), encoding="utf-8")
    typer.echo(f"Reference snapshot {snapshot.snapshot_id} built at {snapshot.db_path}.")


@examples_app.command("build")
def examples_build(
    ctx: typer.Context,
    from_scope: ExamplesScope = typer.Option(..., "--from"),
    lang: str = typer.Option(..., "--lang"),
    skip_if_current: bool = typer.Option(False, "--skip-if-current"),
) -> None:
    state = _require_state(ctx)
    project_root = state.root
    _ensure_project(state, "Examples build")
    config = state.config
    project_id = state.project_id

    if lang == "all":
        targets = config.languages.targets
        if not targets:
            _exit_with_error("No target languages configured.")
        languages = [str(item) for item in targets]
    else:
        languages = [lang]

    sqlite_vector_path = _sqlite_vector_path(project_root)
    if sqlite_vector_path is None:
        _exit_with_error(
            "Examples build failed: sqlite-vector extension not found at ./vector.so"
        )

    for target_lang in languages:
        pointer_path = _cache_path(
            project_root,
            "examples",
            from_scope,
            f"examples.{from_scope}.{target_lang}.current.json",
        )
        if skip_if_current and pointer_path.exists():
            try:
                pointer = po_utils.read_json(pointer_path, pointer_path.name)
                db_file = str(pointer.get("db_file", ""))
                db_path = pointer_path.parent / db_file
                if db_path.exists():
                    db = kdeexamples.open_examples_db(
                        db_path,
                        project_id=project_id,
                        config_hash=config.config_hash,
                        embed_policy_hash=config.embed_policy_hash,
                        sqlite_vector_path=sqlite_vector_path,
                    )
                    db.conn.close()
                    typer.echo(f"Examples cache already current for {target_lang}.")
                    continue
            except Exception:
                pass

        try:
            embedder = kdeplan.require_embedder(examples_embed_policy(config))
        except Exception as exc:
            _exit_with_error(str(exc))
        output_dir = _cache_path(project_root, "examples", from_scope)
        ex_id = _next_pointer_id(pointer_path, "ex_id")
        output_path = output_dir / f"examples.{from_scope}.{target_lang}.{ex_id}.sqlite"

        if from_scope == TmScope.WORKSPACE:
            with closing(
                _ensure_workspace_db(
                    project_root,
                    project_id=project_id,
                    config_hash=config.config_hash,
                    config=config,
                )
            ) as conn:
                kdeexamples.build_examples_db_from_workspace(
                    conn,
                    output_path=output_path,
                    lang=target_lang,
                    config=config,
                    project_id=project_id,
                    config_hash=config.config_hash,
                    embed_policy_hash=config.embed_policy_hash,
                    embedder=embedder,
                    sqlite_vector_path=sqlite_vector_path,
                )
            source_snapshot = {"kind": DbKind.WORKSPACE_TM, "snapshot_id": 0}
        else:
            pointer = po_utils.read_json(
                _cache_path(project_root, "reference", "reference.current.json"),
                "reference.current.json",
            )
            db_file = str(pointer.get("db_file", ""))
            db_path = _cache_path(project_root, "reference", db_file)
            with closing(kdedb.connect_readonly(db_path)) as reference_conn:
                kdeexamples.build_examples_db_from_reference(
                    reference_conn,
                    output_path=output_path,
                    lang=target_lang,
                    config=config,
                    project_id=project_id,
                    config_hash=config.config_hash,
                    embed_policy_hash=config.embed_policy_hash,
                    embedder=embedder,
                    sqlite_vector_path=sqlite_vector_path,
                )
            source_snapshot = {
                "kind": DbKind.REFERENCE_TM,
                "snapshot_id": int(pointer.get("snapshot_id", 0)),
            }

        try:
            meta_db = kdeexamples.open_examples_db(
                output_path,
                project_id=project_id,
                config_hash=config.config_hash,
                embed_policy_hash=config.embed_policy_hash,
                sqlite_vector_path=sqlite_vector_path,
            )
            try:
                meta = dict(meta_db.meta)
            finally:
                meta_db.conn.close()
        except Exception as exc:
            _exit_with_error(f"Examples build failed: {exc}")
        created_at = meta.get("created_at") or datetime.now(timezone.utc).isoformat()
        pointer_payload = {
            "ex_id": ex_id,
            "scope": from_scope,
            "lang": target_lang,
            "db_file": output_path.name,
            "created_at": created_at,
            "embed_policy_hash": config.embed_policy_hash,
            "require_finite": config.prompt.examples.embedding_policy.require_finite,
            "embedding_model_id": meta.get("embedding_model_id", ""),
            "embedding_dim": int(meta.get("embedding_dim", 0)),
            "embedding_distance": meta.get("embedding_distance", ""),
            "vector_encoding": meta.get("vector_encoding", ""),
            "embedding_normalization": meta.get("embedding_normalization", ""),
            "source_snapshot": source_snapshot,
        }
        with _atomic_cache_update(
            project_root,
            f"examples/{from_scope}",
            pointer_path.name,
        ) as (_pointer_path, tmp_path):
            tmp_path.write_text(kdehash.canonical_json(pointer_payload), encoding="utf-8")
        typer.echo(f"Examples DB built for {target_lang} ({from_scope}).")


@glossary_app.command("build")
def glossary_build(
    ctx: typer.Context,
    skip_if_current: bool = typer.Option(False, "--skip-if-current"),
) -> None:
    state = _require_state(ctx)
    project_root = state.root
    _ensure_project(state, "Glossary build")
    config = state.config
    project_id = state.project_id
    pointer_path = _cache_path(project_root, "glossary", "glossary.current.json")

    ref_pointer = po_utils.read_json(
        _cache_path(project_root, "reference", "reference.current.json"),
        "reference.current.json",
    )
    reference_snapshot_id = int(ref_pointer.get("snapshot_id", 0))
    reference_db_file = str(ref_pointer.get("db_file", ""))
    db_path = _cache_path(project_root, "reference", reference_db_file)
    if skip_if_current and _glossary_is_current(
        pointer_path=pointer_path,
        reference_snapshot_id=reference_snapshot_id,
        project_id=project_id,
        config=config,
    ):
        typer.echo("Glossary cache already current.")
        return
    glossary_gen_id = _next_pointer_id(pointer_path, "snapshot_id")
    output_path = _cache_path(
        project_root,
        "glossary",
        f"glossary.{glossary_gen_id}.sqlite",
    )
    while output_path.exists():
        glossary_gen_id += 1
        output_path = _cache_path(
            project_root,
            "glossary",
            f"glossary.{glossary_gen_id}.sqlite",
        )
    try:
        with closing(kdedb.connect_readonly(db_path)) as reference_conn:
            kdeglo.build_glossary_db(
                reference_conn,
                output_path=output_path,
                config=config,
                project_id=project_id,
                config_hash=config.config_hash,
                glossary_snapshot_id=glossary_gen_id,
            )

        with closing(kdedb.connect_readonly(output_path)) as conn:
            kdedb.validate_meta_table(
                conn,
                expected_project_id=project_id,
                expected_config_hash=config.config_hash,
                expected_kind=DbKind.GLOSSARY,
                expected_normalization_id=glossary_normalization_id(config),
            )
            meta = kdedb.read_meta(conn)
            created_at = meta.get("created_at")
            if not created_at:
                raise ValueError("Missing created_at in glossary meta")
    except Exception as exc:
        _exit_with_error(f"Glossary build failed: {exc}")

    pointer_payload = {
        "snapshot_id": glossary_gen_id,
        "db_file": output_path.name,
        "created_at": created_at,
        "source_snapshot": {
            "kind": DbKind.REFERENCE_TM,
            "snapshot_id": reference_snapshot_id,
        },
    }
    with _atomic_cache_update(
        project_root,
        "glossary",
        "glossary.current.json",
    ) as (_pointer_path, tmp_path):
        tmp_path.write_text(kdehash.canonical_json(pointer_payload), encoding="utf-8")
    typer.echo("Glossary DB built.")


@app.command()
def doctor(
    ctx: typer.Context,
    repair_cache: bool = typer.Option(False, "--repair-cache"),
) -> None:
    state = _require_state(ctx)
    project_root = state.root
    report = kdedoctor.run_doctor(project_root, repair_cache=repair_cache)
    for note in report.notes:
        typer.echo(note)
    for message in report.warnings:
        typer.secho(f"Warning: {message}", err=True)
    if report.errors:
        error_message = "\nError: ".join(report.errors)
        _exit_with_error(error_message)
    typer.echo("Doctor checks passed.")


@app.command()
def gc(
    ctx: typer.Context,
    ttl_days: int = typer.Option(30, "--ttl-days"),
) -> None:
    state = _require_state(ctx)
    project_root = state.root
    _ensure_project(state, "GC")
    config = state.config
    project_id = state.project_id
    try:
        report = kdegc.gc_workspace_tm(
            project_root,
            project_id=project_id,
            config_hash=config.config_hash,
            config=config,
            ttl_days=ttl_days,
        )
    except Exception as exc:
        _exit_with_error(f"GC failed: {exc}")
    typer.echo(
        "GC complete: "
        f"{report.files_deleted} files, "
        f"{report.best_translations_deleted} best translations, "
        f"{report.sources_deleted} sources removed (cutoff {report.cutoff_iso})."
    )
