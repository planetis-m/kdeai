from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal, Sequence
import os
from datetime import datetime, timezone

import portalocker
import typer
import click

from kdeai import apply as kdeapply
from kdeai.config import Config, EmbeddingPolicy
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
from kdeai.project import Project

app = typer.Typer(add_completion=False, no_args_is_help=True)
reference_app = typer.Typer(no_args_is_help=True)
examples_app = typer.Typer(no_args_is_help=True)
glossary_app = typer.Typer(no_args_is_help=True)
app.add_typer(reference_app, name="reference")
app.add_typer(examples_app, name="examples")
app.add_typer(glossary_app, name="glossary")

OnOff = Literal["on", "off"]
CacheMode = Literal["off", "on"]
ExamplesMode = Literal["off", "auto", "required"]
GlossaryMode = Literal["off", "auto", "required"]
ApplyMode = Literal["strict", "rebase"]
OverwriteMode = Literal["conservative", "allow-nonempty", "allow-reviewed", "all"]
ExamplesScope = Literal["workspace", "reference"]


def _project_dir(project_root: Path) -> Path:
    return project_root / ".kdeai"


_read_json = po_utils.read_json


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(kdehash.canonical_json(payload), encoding="utf-8")
    os.replace(tmp_path, path)


def _acquire_run_lock(project_root: Path, ctx: typer.Context) -> None:
    lock_cm = locks.acquire_run_lock(project_root)
    try:
        lock_cm.__enter__()
    except portalocker.exceptions.LockException:
        typer.secho("Global run lock is held; another KDEAI process is running.", err=True)
        raise typer.Exit(1)
    ctx.obj = {"project_root": project_root, "run_lock_cm": lock_cm}
    ctx.call_on_close(lambda: lock_cm.__exit__(None, None, None))


def _project_root(ctx: typer.Context) -> Path:
    obj = ctx.obj or {}
    root = obj.get("project_root")
    if isinstance(root, Path):
        return root
    return Path.cwd()


_normalize_relpath = po_utils.normalize_relpath
_relpath_key = po_utils.relpath_key


_iter_po_paths = po_utils.iter_po_paths


def _parse_lang_from_bytes(po_bytes: bytes) -> Optional[str]:
    po_file = po_utils.load_po_from_bytes(po_bytes)
    language = po_file.metadata.get("Language") if po_file else None
    if language:
        return str(language).strip()
    return None


def _next_pointer_id(pointer_path: Path, key: str) -> int:
    if not pointer_path.exists():
        return 1
    try:
        payload = _read_json(pointer_path, pointer_path.name)
        pointer_id = int(payload.get(key, 0))
    except (OSError, ValueError):
        pointer_id = 0
    return pointer_id + 1


def _examples_embed_policy(config: Config) -> EmbeddingPolicy:
    return config.prompt.examples.embedding_policy


def _require_embedder(policy: EmbeddingPolicy) -> kdeexamples.EmbeddingFunc:
    from kdeai.embed_client import compute_embedding

    def _embed(texts: Sequence[str]) -> list[list[float]]:
        return [compute_embedding(text, policy=policy) for text in texts]

    return _embed


def _examples_mode_from_config(
    config: Config,
    override: ExamplesMode | None,
) -> ExamplesMode:
    if override is not None:
        return override
    return str(config.prompt.examples.mode_default or "auto")


def _glossary_mode_from_config(
    config: Config,
    override: GlossaryMode | None,
) -> GlossaryMode:
    if override is not None:
        return override
    return str(config.prompt.glossary.mode_default or "auto")


def _maybe_embedder(
    examples_mode: ExamplesMode,
    policy: EmbeddingPolicy,
) -> kdeexamples.EmbeddingFunc | None:
    if examples_mode == "off":
        return None
    try:
        return _require_embedder(policy)
    except Exception:
        if examples_mode == "required":
            raise
        return None


def _sqlite_vector_path(project_root: Path) -> str | None:
    candidate = project_root / "vector.so"
    if candidate.exists():
        return str(candidate)
    return None


def _glossary_normalization_id(config: Config) -> str:
    return str(config.prompt.glossary.normalization_id or kdeglo.NORMALIZATION_ID)


def _apply_defaults_from_config(config: Config) -> dict:
    return {
        "mode": str(config.apply.mode_default),
        "overwrite": str(config.apply.overwrite_default),
        "post_index": "off",
    }


def _workspace_tm_settings(config: Config) -> tuple[int, str]:
    workspace_cfg = config.sqlite.workspace_tm
    return int(workspace_cfg.busy_timeout_ms.write), str(workspace_cfg.synchronous).upper()


def _resolve_planner_inputs(
    *,
    cache_mode: CacheMode,
    examples: ExamplesMode | None,
    glossary: GlossaryMode | None,
    config: Config,
    project_root: Path,
) -> tuple[ExamplesMode, GlossaryMode, kdeexamples.EmbeddingFunc | None, str | None]:
    resolved_examples_mode = _examples_mode_from_config(config, examples)
    resolved_glossary_mode = _glossary_mode_from_config(config, glossary)
    if cache_mode == "off":
        return "off", "off", None, None
    embedder = _maybe_embedder(
        resolved_examples_mode,
        _examples_embed_policy(config),
    )
    sqlite_vector_path = _sqlite_vector_path(project_root)
    return resolved_examples_mode, resolved_glossary_mode, embedder, sqlite_vector_path


def _build_plan_header(
    *,
    project: Project,
    config: Config,
    lang: str,
    builder: kdeplan.PlanBuilder,
    apply_defaults: dict,
) -> dict:
    return {
        "format": kdeplan.PLAN_FORMAT_VERSION,
        "project_id": str(project.project_data["project_id"]),
        "config_hash": config.config_hash,
        "lang": lang,
        "marker_flags": list(builder.marker_flags),
        "comment_prefixes": list(builder.comment_prefixes),
        "ai_flag": builder.ai_flag,
        "placeholder_patterns": list(config.apply.validation_patterns),
        "apply_defaults": {
            "mode": str(apply_defaults.get("mode", "strict")),
            "overwrite": str(apply_defaults.get("overwrite", "conservative")),
            "post_index": str(apply_defaults.get("post_index", "off")),
        },
    }


def _ensure_workspace_db(
    project_root: Path,
    *,
    project_id: str,
    config_hash: str,
    config: Config,
):
    db_path = project_root / ".kdeai" / "cache" / "workspace.tm.sqlite"
    created = not db_path.exists()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    busy_timeout_ms, synchronous = _workspace_tm_settings(config)
    conn = kdedb.connect_workspace_tm(
        db_path,
        busy_timeout_ms=busy_timeout_ms,
        synchronous=synchronous,
    )
    try:
        if created:
            conn.executescript(kdedb.WORKSPACE_TM_SCHEMA)
            meta = {
                "schema_version": "1",
                "kind": "workspace_tm",
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
                expected_kind="workspace_tm",
            )
    except Exception:
        conn.close()
        raise
    return conn


@app.callback()
def main(ctx: typer.Context) -> None:
    project_root = Path.cwd()
    _acquire_run_lock(project_root, ctx)


@app.command()
def init() -> None:
    project_root = Path.cwd()
    project_data = Project.ensure_project_data(project_root)
    typer.echo(f"Initialized project {project_data['project_id']} at {_project_dir(project_root)}.")
    config_path = project_root / ".kdeai" / "config.json"
    if not config_path.exists():
        typer.secho("Missing .kdeai/config.json. Create it before planning or indexing.", err=True)


@app.command()
def plan(
    paths: Optional[list[Path]] = typer.Argument(None),
    lang: str = typer.Option(..., "--lang"),
    out: Optional[Path] = typer.Option(None, "--out"),
    cache: Optional[CacheMode] = typer.Option(None, "--cache"),
    cache_write: Optional[OnOff] = typer.Option(None, "--cache-write"),
    examples: Optional[ExamplesMode] = typer.Option(None, "--examples"),
    glossary: Optional[GlossaryMode] = typer.Option(None, "--glossary"),
) -> None:
    ctx = click.get_current_context()
    project_root = _project_root(ctx)
    project = Project.load_or_init(project_root)
    config = project.config
    cache_mode = cache or "on"
    cache_write_flag = cache_write or "on"
    if cache_write_flag == "off":
        typer.secho("Note: plan never writes cache; --cache-write has no effect.", err=True)
    resolved_examples_mode, resolved_glossary_mode, embedder, sqlite_vector_path = (
        _resolve_planner_inputs(
            cache_mode=cache_mode,
            examples=examples,
            glossary=glossary,
            config=config,
            project_root=project_root,
        )
    )
    path_casefold = bool(project.project_data.get("path_casefold", os.name == "nt"))
    builder = kdeplan.PlanBuilder(
        project_root=project_root,
        project_id=str(project.project_data["project_id"]),
        config=config,
        lang=lang,
        cache=cache_mode,
        cache_write=cache_write_flag,
        examples_mode=resolved_examples_mode,
        glossary_mode=resolved_glossary_mode,
        embedder=embedder,
        sqlite_vector_path=sqlite_vector_path,
    )

    files_payload: list[dict] = []
    try:
        for path in _iter_po_paths(project_root, paths):
            file_draft = kdeplan.generate_plan_for_file(
                project_root=project_root,
                project_id=str(project.project_data["project_id"]),
                path=path,
                path_casefold=path_casefold,
                builder=builder,
                config=config,
                run_llm=False,
            )
            files_payload.append(file_draft)
    finally:
        builder.close()

    files_payload.sort(key=lambda item: str(item.get("file_path", "")))
    apply_cfg = _apply_defaults_from_config(config)
    plan_payload = _build_plan_header(
        project=project,
        config=config,
        lang=lang,
        builder=builder,
        apply_defaults=apply_cfg,
    )
    plan_payload["files"] = files_payload

    if out is None:
        typer.echo(kdeplan.render_plan_json(plan_payload))
        return
    finalized = kdeplan.write_plan(out, plan_payload)
    typer.echo(f"Wrote plan {finalized['plan_id']} to {out}.")


@app.command()
def apply(
    plan_path: Path = typer.Argument(...),
    apply_mode: Optional[ApplyMode] = typer.Option(None, "--apply-mode"),
    overwrite: Optional[OverwriteMode] = typer.Option(None, "--overwrite"),
    post_index: OnOff = typer.Option("off", "--post-index"),
) -> None:
    ctx = click.get_current_context()
    project_root = _project_root(ctx)
    plan = kdeplan.load_plan(plan_path)
    post_index_flag = post_index == "on"
    workspace_conn = None
    try:
        project = Project.load_or_init(project_root)
        config = project.config
    except Exception as exc:
        typer.secho(f"Apply failed: {exc}", err=True)
        raise typer.Exit(1)

    project_id = str(plan.get("project_id") or project.project_data.get("project_id") or "")

    if post_index_flag:
        try:
            workspace_conn = _ensure_workspace_db(
                project_root,
                project_id=project_id,
                config_hash=config.config_hash,
                config=config,
            )
        except Exception as exc:
            typer.secho(f"Warning: post-index disabled ({exc}).", err=True)
            post_index_flag = False

    result = kdeapply.apply_plan(
        plan,
        project_root=project_root,
        config=config,
        apply_mode=apply_mode,
        overwrite=overwrite,
        post_index=post_index_flag,
        workspace_conn=workspace_conn,
    )

    if workspace_conn is not None:
        workspace_conn.close()

    if result.errors:
        for error in result.errors:
            typer.secho(error, err=True)
        raise typer.Exit(1)

    if result.warnings:
        for warning in result.warnings:
            typer.secho(f"Warning: {warning}", err=True)

    typer.echo(
        f"Applied {result.entries_applied} entries "
        f"({len(result.files_written)} files written, {len(result.files_skipped)} skipped)."
    )


@app.command()
def translate(
    paths: Optional[list[Path]] = typer.Argument(None),
    lang: str = typer.Option(..., "--lang"),
    out: Optional[Path] = typer.Option(None, "--out"),
    apply_mode: Optional[ApplyMode] = typer.Option(None, "--apply-mode"),
    overwrite: Optional[OverwriteMode] = typer.Option(None, "--overwrite"),
    cache: Optional[CacheMode] = typer.Option(None, "--cache"),
    cache_write: Optional[OnOff] = typer.Option(None, "--cache-write"),
    examples: Optional[ExamplesMode] = typer.Option(None, "--examples"),
    glossary: Optional[GlossaryMode] = typer.Option(None, "--glossary"),
) -> None:
    ctx = click.get_current_context()
    project_root = _project_root(ctx)
    project = Project.load_or_init(project_root)
    config = project.config
    cache_write_flag = cache_write or "on"
    cache_mode = cache or "on"
    resolved_examples_mode, resolved_glossary_mode, embedder, sqlite_vector_path = (
        _resolve_planner_inputs(
            cache_mode=cache_mode,
            examples=examples,
            glossary=glossary,
            config=config,
            project_root=project_root,
        )
    )
    path_casefold = bool(project.project_data.get("path_casefold", os.name == "nt"))

    apply_defaults = _apply_defaults_from_config(config)
    post_index_flag = apply_defaults.get("post_index") == "on"
    if cache_mode == "off" or cache_write_flag == "off":
        post_index_flag = False

    builder = kdeplan.PlanBuilder(
        project_root=project_root,
        project_id=str(project.project_data["project_id"]),
        config=config,
        lang=lang,
        cache=cache_mode,
        cache_write=cache_write_flag,
        examples_mode=resolved_examples_mode,
        glossary_mode=resolved_glossary_mode,
        overwrite=overwrite,
        session_tm={},
        embedder=embedder,
        sqlite_vector_path=sqlite_vector_path,
    )

    files_payload: list[dict] = []
    workspace_conn = None
    session_tm = builder.session_tm
    total_entries_applied = 0
    files_written: list[str] = []
    files_skipped: list[str] = []

    if post_index_flag:
        try:
            workspace_conn = _ensure_workspace_db(
                project_root,
                project_id=str(project.project_data["project_id"]),
                config_hash=config.config_hash,
                config=config,
            )
        except Exception as exc:
            typer.secho(f"Warning: post-index disabled ({exc}).", err=True)
            post_index_flag = False

    plan_header = _build_plan_header(
        project=project,
        config=config,
        lang=lang,
        builder=builder,
        apply_defaults=apply_defaults,
    )

    try:
        for path in _iter_po_paths(project_root, paths):
            file_plan = kdeplan.generate_plan_for_file(
                project_root=project_root,
                project_id=str(project.project_data["project_id"]),
                path=path,
                path_casefold=path_casefold,
                builder=builder,
                config=config,
                run_llm=True,
            )
            files_payload.append(file_plan)
            per_file_plan = dict(plan_header)
            per_file_plan["files"] = [file_plan]
            result = kdeapply.apply_plan(
                per_file_plan,
                project_root=project_root,
                config=config,
                apply_mode=apply_mode,
                overwrite=overwrite,
                post_index=post_index_flag,
                workspace_conn=workspace_conn,
                session_tm=session_tm,
            )

            if result.errors:
                for error in result.errors:
                    typer.secho(error, err=True)
                if out is not None:
                    combined_plan = dict(plan_header)
                    files_payload.sort(key=lambda item: str(item.get("file_path", "")))
                    combined_plan["files"] = files_payload
                    kdeplan.write_plan(out, combined_plan)
                raise typer.Exit(1)

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
    finally:
        if workspace_conn is not None:
            workspace_conn.close()
        builder.close()

    typer.echo(
        f"Applied {total_entries_applied} entries "
        f"({len(files_written)} files written, {len(files_skipped)} skipped)."
    )


@app.command()
def index(
    paths: Optional[list[Path]] = typer.Argument(None),
    strict: bool = typer.Option(False, "--strict"),
) -> None:
    ctx = click.get_current_context()
    project_root = _project_root(ctx)
    project = Project.load_or_init(project_root)
    config = project.config

    project_id = str(project.project_data["project_id"])
    path_casefold = bool(project.project_data.get("path_casefold"))
    conn = _ensure_workspace_db(
        project_root,
        project_id=project_id,
        config_hash=config.config_hash,
        config=config,
    )
    try:
        for path in _iter_po_paths(project_root, paths):
            relpath = _normalize_relpath(project_root, path)
            relpath_key = _relpath_key(relpath, path_casefold)
            lock_path = locks.per_file_lock_path(
                project_root,
                locks.lock_id(project_id, relpath_key),
            )
            try:
                locked = snapshot.locked_read_file(path, lock_path)
                lang = _parse_lang_from_bytes(locked.bytes)
                if not lang:
                    targets = config.languages.targets
                    if len(targets) == 1:
                        lang = str(targets[0])
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
                    typer.secho(message, err=True)
                    raise typer.Exit(1)
                typer.secho(f"Warning: {message}", err=True)
    finally:
        conn.close()

    typer.echo("Workspace TM index updated.")


@reference_app.command("build")
def reference_build(
    paths: Optional[list[Path]] = typer.Argument(None),
    label: Optional[str] = typer.Option(None, "--label"),
) -> None:
    ctx = click.get_current_context()
    project_root = _project_root(ctx)
    try:
        project = Project.load_or_init(project_root)
        config = project.config
        snapshot = kderef.build_reference_snapshot(
            project_root,
            project_id=str(project.project_data["project_id"]),
            path_casefold=bool(project.project_data.get("path_casefold")),
            config=config,
            config_hash=config.config_hash,
            paths=paths,
            label=label,
        )
    except Exception as exc:
        typer.secho(f"Reference build failed: {exc}", err=True)
        raise typer.Exit(1)
    pointer_path = (
        project_root
        / ".kdeai"
        / "cache"
        / "reference"
        / "reference.current.json"
    )
    pointer_payload = {
        "snapshot_id": snapshot.snapshot_id,
        "db_file": snapshot.db_path.name,
        "created_at": snapshot.created_at,
    }
    _write_json_atomic(pointer_path, pointer_payload)
    typer.echo(f"Reference snapshot {snapshot.snapshot_id} built at {snapshot.db_path}.")


@examples_app.command("build")
def examples_build(
    from_scope: ExamplesScope = typer.Option(..., "--from"),
    lang: str = typer.Option(..., "--lang"),
    skip_if_current: bool = typer.Option(False, "--skip-if-current"),
) -> None:
    ctx = click.get_current_context()
    project_root = _project_root(ctx)
    try:
        project = Project.load_or_init(project_root)
        config = project.config
    except Exception as exc:
        typer.secho(f"Examples build failed: {exc}", err=True)
        raise typer.Exit(1)

    if lang == "all":
        targets = config.languages.targets
        if not targets:
            typer.secho("No target languages configured.", err=True)
            raise typer.Exit(1)
        languages = [str(item) for item in targets]
    else:
        languages = [lang]

    for target_lang in languages:
        pointer_path = (
            project_root
            / ".kdeai"
            / "cache"
            / "examples"
            / from_scope
            / f"examples.{from_scope}.{target_lang}.current.json"
        )
        if skip_if_current and pointer_path.exists():
            try:
                pointer = _read_json(pointer_path, pointer_path.name)
                db_file = str(pointer.get("db_file", ""))
                db_path = pointer_path.parent / db_file
                if db_path.exists():
                    db = kdeexamples.open_examples_db(
                        db_path,
                        project_id=str(project.project_data["project_id"]),
                        config_hash=config.config_hash,
                        embed_policy_hash=config.embed_policy_hash,
                    )
                    db.conn.close()
                    typer.echo(f"Examples cache already current for {target_lang}.")
                    continue
            except Exception:
                pass

        try:
            embedder = _require_embedder(_examples_embed_policy(config))
        except Exception as exc:
            typer.secho(str(exc), err=True)
            raise typer.Exit(1) from exc
        output_dir = (
            project_root / ".kdeai" / "cache" / "examples" / from_scope
        )
        ex_id = _next_pointer_id(pointer_path, "ex_id")
        output_path = output_dir / f"examples.{from_scope}.{target_lang}.{ex_id}.sqlite"

        if from_scope == "workspace":
            conn = _ensure_workspace_db(
                project_root,
                project_id=str(project.project_data["project_id"]),
                config_hash=config.config_hash,
                config=config,
            )
            try:
                kdeexamples.build_examples_db_from_workspace(
                    conn,
                    output_path=output_path,
                    lang=target_lang,
                    config=config,
                    project_id=str(project.project_data["project_id"]),
                    config_hash=config.config_hash,
                    embed_policy_hash=config.embed_policy_hash,
                    embedder=embedder,
                )
            finally:
                conn.close()
            source_snapshot = {"kind": "workspace_tm", "snapshot_id": None}
        else:
            pointer = _read_json(
                project_root
                / ".kdeai"
                / "cache"
                / "reference"
                / "reference.current.json",
                "reference.current.json",
            )
            db_file = str(pointer.get("db_file", ""))
            db_path = (
                project_root / ".kdeai" / "cache" / "reference" / db_file
            )
            reference_conn = kdedb.connect_readonly(db_path)
            try:
                kdeexamples.build_examples_db_from_reference(
                    reference_conn,
                    output_path=output_path,
                    lang=target_lang,
                    config=config,
                    project_id=str(project.project_data["project_id"]),
                    config_hash=config.config_hash,
                    embed_policy_hash=config.embed_policy_hash,
                    embedder=embedder,
                )
            finally:
                reference_conn.close()
            source_snapshot = {
                "kind": "reference_tm",
                "snapshot_id": int(pointer.get("snapshot_id", 0)),
            }

        meta_db = kdeexamples.open_examples_db(
            output_path,
            project_id=str(project.project_data["project_id"]),
            config_hash=config.config_hash,
            embed_policy_hash=config.embed_policy_hash,
        )
        meta = dict(meta_db.meta)
        meta_db.conn.close()
        pointer_payload = {
            "ex_id": ex_id,
            "scope": from_scope,
            "lang": target_lang,
            "db_file": output_path.name,
            "created_at": meta.get("created_at", ""),
            "embed_policy_hash": config.embed_policy_hash,
            "embedding_model_id": meta.get("embedding_model_id", ""),
            "embedding_dim": int(meta.get("embedding_dim", 0)),
            "embedding_distance": meta.get("embedding_distance", ""),
            "vector_encoding": meta.get("vector_encoding", ""),
            "embedding_normalization": meta.get("embedding_normalization", ""),
            "source_snapshot": source_snapshot,
        }
        _write_json_atomic(pointer_path, pointer_payload)
        typer.echo(f"Examples DB built for {target_lang} ({from_scope}).")


@glossary_app.command("build")
def glossary_build(
    skip_if_current: bool = typer.Option(False, "--skip-if-current"),
) -> None:
    ctx = click.get_current_context()
    project_root = _project_root(ctx)
    try:
        project = Project.load_or_init(project_root)
        config = project.config
    except Exception as exc:
        typer.secho(f"Glossary build failed: {exc}", err=True)
        raise typer.Exit(1)
    pointer_path = (
        project_root / ".kdeai" / "cache" / "glossary" / "glossary.current.json"
    )
    if skip_if_current and pointer_path.exists():
        try:
            pointer = _read_json(pointer_path, pointer_path.name)
            db_file = str(pointer.get("db_file", ""))
            db_path = pointer_path.parent / db_file
            conn = kdedb.connect_readonly(db_path)
            kdedb.validate_meta_table(
                conn,
                expected_project_id=str(project.project_data["project_id"]),
                expected_config_hash=config.config_hash,
                expected_kind="glossary",
                expected_normalization_id=_glossary_normalization_id(config),
            )
            conn.close()
            typer.echo("Glossary cache already current.")
            return
        except Exception:
            pass

    ref_pointer = _read_json(
        project_root
        / ".kdeai"
        / "cache"
        / "reference"
        / "reference.current.json",
        "reference.current.json",
    )
    db_file = str(ref_pointer.get("db_file", ""))
    db_path = project_root / ".kdeai" / "cache" / "reference" / db_file
    reference_conn = kdedb.connect_readonly(db_path)
    try:
        output_path = (
            project_root
            / ".kdeai"
            / "cache"
            / "glossary"
            / f"glossary.{int(ref_pointer.get('snapshot_id', 0))}.sqlite"
        )
        kdeglossary_path = kdeglo.build_glossary_db(
            reference_conn,
            output_path=output_path,
            config=config,
            project_id=str(project.project_data["project_id"]),
            config_hash=config.config_hash,
        )
    finally:
        reference_conn.close()

    pointer_payload = {
        "snapshot_id": int(ref_pointer.get("snapshot_id", 0)),
        "db_file": kdeglossary_path.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_snapshot": {
            "kind": "reference_tm",
            "snapshot_id": int(ref_pointer.get("snapshot_id", 0)),
        },
    }
    _write_json_atomic(pointer_path, pointer_payload)
    typer.echo("Glossary DB built.")


@app.command()
def doctor(
    repair_cache: bool = typer.Option(False, "--repair-cache"),
) -> None:
    ctx = click.get_current_context()
    project_root = _project_root(ctx)
    report = kdedoctor.run_doctor(project_root, repair_cache=repair_cache)
    for note in report.notes:
        typer.echo(note)
    for message in report.warnings:
        typer.secho(f"Warning: {message}", err=True)
    if report.errors:
        for message in report.errors:
            typer.secho(f"Error: {message}", err=True)
        raise typer.Exit(1)
    typer.echo("Doctor checks passed.")


@app.command()
def gc(
    ttl_days: int = typer.Option(30, "--ttl-days"),
) -> None:
    ctx = click.get_current_context()
    project_root = _project_root(ctx)
    try:
        project = Project.load_or_init(project_root)
        config = project.config
        report = kdegc.gc_workspace_tm(
            project_root,
            project_id=str(project.project_data["project_id"]),
            config_hash=config.config_hash,
            config=config,
            ttl_days=ttl_days,
        )
    except Exception as exc:
        typer.secho(f"GC failed: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(
        "GC complete: "
        f"{report.files_deleted} files, "
        f"{report.best_translations_deleted} best translations, "
        f"{report.sources_deleted} sources removed (cutoff {report.cutoff_iso})."
    )
