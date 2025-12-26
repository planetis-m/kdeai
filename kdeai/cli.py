from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal
import json
import os
from datetime import datetime, timezone
import hashlib
import tempfile

import portalocker
import polib
import typer
import click

from kdeai import apply as kdeapply
from kdeai import config as kdeconfig
from kdeai import db as kdedb
from kdeai import doctor as kdedoctor
from kdeai import examples as kdeexamples
from kdeai import gc as kdegc
from kdeai import glossary as kdeglo
from kdeai import hash as kdehash
from kdeai import locks
from kdeai import plan as kdeplan
from kdeai import reference as kderef
from kdeai import snapshot
from kdeai import workspace_tm

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
ApplyMode = Literal["strict", "rebase"]
OverwriteMode = Literal["conservative", "allow-nonempty", "allow-reviewed", "all"]
ExamplesScope = Literal["workspace", "reference"]


def _project_dir(project_root: Path) -> Path:
    return project_root / ".kdeai"


def _project_path(project_root: Path) -> Path:
    return _project_dir(project_root) / "project.json"


def _read_json(path: Path, label: str) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return data


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(kdehash.canonical_json(payload), encoding="utf-8")


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(kdehash.canonical_json(payload), encoding="utf-8")
    os.replace(tmp_path, path)


def _path_casefold() -> bool:
    return os.name == "nt"


def _generate_project_id(project_dir: Path) -> tuple[str, str]:
    try:
        stat = project_dir.stat()
        if os.name != "nt" and getattr(stat, "st_ino", None) is not None:
            payload = f"{stat.st_dev}\n{stat.st_ino}"
            return kdehash.sha256_hex_text(payload), "posix_dev_ino"
    except OSError:
        pass

    payload = str(project_dir.resolve())
    return kdehash.sha256_hex_text(payload), "realpath_fallback"


def _load_project(project_root: Path) -> dict:
    project_path = _project_path(project_root)
    project = _read_json(project_path, "project.json")
    if "project_id" not in project:
        raise ValueError("project.json missing project_id")
    return project


def _ensure_project(project_root: Path) -> dict:
    project_path = _project_path(project_root)
    if project_path.exists():
        return _load_project(project_root)

    project_dir = _project_dir(project_root)
    project_dir.mkdir(parents=True, exist_ok=True)
    project_id, method = _generate_project_id(project_dir)
    payload = {
        "format": 1,
        "project_id": project_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "path_casefold": _path_casefold(),
        "method": method,
    }
    _write_json(project_path, payload)
    return payload


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


def _normalize_relpath(project_root: Path, path: Path) -> str:
    resolved = path.resolve()
    relpath = resolved.relative_to(project_root.resolve())
    return relpath.as_posix()


def _relpath_key(relpath: str, path_casefold: bool) -> str:
    return relpath.casefold() if path_casefold else relpath


def _iter_po_paths(project_root: Path, raw_paths: Optional[list[Path]]) -> list[Path]:
    roots = raw_paths if raw_paths else [project_root]
    seen: set[Path] = set()
    results: list[Path] = []

    for raw in roots:
        full = raw if raw.is_absolute() else project_root / raw
        if not full.exists():
            continue
        if full.is_file():
            candidates = [full]
        else:
            candidates = list(full.rglob("*.po"))
        for candidate in candidates:
            if candidate.suffix.lower() != ".po":
                continue
            if any(part in {".kdeai", ".git"} for part in candidate.parts):
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            results.append(resolved)
    return results


def _parse_lang_from_bytes(po_bytes: bytes) -> Optional[str]:
    with tempfile.NamedTemporaryFile(suffix=".po", delete=False) as tmp:
        tmp.write(po_bytes)
        tmp_path = tmp.name
    try:
        po_file = polib.pofile(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    language = po_file.metadata.get("Language") if po_file else None
    if language:
        return str(language).strip()
    return None


def _load_po_from_bytes(po_bytes: bytes) -> polib.POFile:
    with tempfile.NamedTemporaryFile(suffix=".po", delete=False) as tmp:
        tmp.write(po_bytes)
        tmp_path = tmp.name
    try:
        return polib.pofile(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _next_pointer_id(pointer_path: Path, key: str) -> int:
    if not pointer_path.exists():
        return 1
    try:
        payload = _read_json(pointer_path, pointer_path.name)
        pointer_id = int(payload.get(key, 0))
    except (OSError, ValueError):
        pointer_id = 0
    return pointer_id + 1


def _examples_embed_policy(config: dict) -> dict[str, object] | None:
    prompt = config.get("prompt") if isinstance(config, dict) else None
    examples_cfg = prompt.get("examples") if isinstance(prompt, dict) else None
    policy = examples_cfg.get("embedding_policy") if isinstance(examples_cfg, dict) else None
    if not isinstance(policy, dict):
        return None
    return policy


def _require_embedder(policy: dict[str, object] | None) -> kdeexamples.EmbeddingFunc:
    from kdeai.embed_client import compute_embedding

    def _embed(texts: Sequence[str]) -> list[list[float]]:
        return [compute_embedding(text, policy=policy) for text in texts]

    return _embed


def _examples_mode_from_config(
    config: dict,
    override: ExamplesMode | None,
) -> ExamplesMode:
    if override is not None:
        return override
    prompt = config.get("prompt") if isinstance(config, dict) else None
    examples_cfg = prompt.get("examples") if isinstance(prompt, dict) else None
    mode_default = (
        examples_cfg.get("mode_default")
        if isinstance(examples_cfg, dict)
        else None
    )
    if not mode_default:
        return "auto"
    return str(mode_default)


def _maybe_embedder(
    examples_mode: ExamplesMode,
    policy: dict[str, object] | None,
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


def _glossary_normalization_id(config: dict) -> str:
    prompt = config.get("prompt") if isinstance(config, dict) else None
    glossary_cfg = prompt.get("glossary") if isinstance(prompt, dict) else None
    normalization = (
        glossary_cfg.get("normalization_id")
        if isinstance(glossary_cfg, dict)
        else None
    )
    return str(normalization or kdeglo.NORMALIZATION_ID)


def _apply_defaults_from_config(config: dict) -> dict:
    apply_cfg = config.get("apply") if isinstance(config, dict) else None
    if not isinstance(apply_cfg, dict):
        return {"mode": "strict", "overwrite": "conservative", "post_index": "off"}
    return {
        "mode": str(apply_cfg.get("mode_default") or "strict"),
        "overwrite": str(apply_cfg.get("overwrite_default") or "conservative"),
        "post_index": str(apply_cfg.get("post_index_default") or "off"),
    }


def _marker_settings_from_config(config: dict) -> tuple[list[str], list[str], str]:
    markers = config.get("markers") if isinstance(config, dict) else None
    if not isinstance(markers, dict):
        return kdeapply.DEFAULT_MARKER_FLAGS, kdeapply.DEFAULT_COMMENT_PREFIXES, kdeapply.DEFAULT_AI_FLAG
    ai_flag = str(markers.get("ai_flag") or kdeapply.DEFAULT_AI_FLAG)
    comment_prefixes = markers.get("comment_prefixes")
    if isinstance(comment_prefixes, dict):
        ordered = []
        for key in ("tool", "ai", "tm", "review"):
            value = comment_prefixes.get(key)
            if value:
                ordered.append(str(value))
        if ordered:
            return kdeapply.DEFAULT_MARKER_FLAGS, ordered, ai_flag
    return kdeapply.DEFAULT_MARKER_FLAGS, kdeapply.DEFAULT_COMMENT_PREFIXES, ai_flag


def _workspace_tm_settings(config: dict) -> tuple[int, str]:
    sqlite_cfg = config.get("sqlite") if isinstance(config, dict) else None
    workspace_cfg = sqlite_cfg.get("workspace_tm") if isinstance(sqlite_cfg, dict) else None
    busy_timeout_ms = 50
    synchronous = "NORMAL"
    if isinstance(workspace_cfg, dict):
        synchronous = str(workspace_cfg.get("synchronous", synchronous)).upper()
        timeout_cfg = workspace_cfg.get("busy_timeout_ms")
        if isinstance(timeout_cfg, dict):
            busy_timeout_ms = int(timeout_cfg.get("write", busy_timeout_ms))
        elif isinstance(timeout_cfg, int):
            busy_timeout_ms = int(timeout_cfg)
    return busy_timeout_ms, synchronous


def _ensure_workspace_db(
    project_root: Path,
    *,
    project_id: str,
    config_hash: str,
    config_data: dict,
):
    db_path = project_root / ".kdeai" / "cache" / "workspace.tm.sqlite"
    created = not db_path.exists()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    busy_timeout_ms, synchronous = _workspace_tm_settings(config_data)
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
    project = _ensure_project(project_root)
    typer.echo(f"Initialized project {project['project_id']} at {_project_dir(project_root)}.")
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
    glossary: Optional[ExamplesMode] = typer.Option(None, "--glossary"),
) -> None:
    ctx = click.get_current_context()
    project_root = _project_root(ctx)
    project = _load_project(project_root)
    config = kdeconfig.load_config_from_root(project_root)
    _ = cache_write
    cache_mode = cache or "on"
    resolved_examples_mode = _examples_mode_from_config(config.data, examples)
    embedder = _maybe_embedder(
        resolved_examples_mode,
        _examples_embed_policy(config.data),
    )
    sqlite_vector_path = _sqlite_vector_path(project_root)
    path_casefold = bool(project.get("path_casefold", os.name == "nt"))
    builder = kdeplan.PlanBuilder(
        project_root=project_root,
        project_id=str(project["project_id"]),
        config=config.data,
        config_hash=config.config_hash,
        embed_policy_hash=config.embed_policy_hash,
        lang=lang,
        cache=cache_mode,
        examples_mode=examples,
        glossary_mode=glossary,
        embedder=embedder,
        sqlite_vector_path=sqlite_vector_path,
    )

    files_payload: list[dict] = []
    try:
        for path in _iter_po_paths(project_root, paths):
            file_draft = kdeplan.generate_plan_for_file(
                project_root=project_root,
                project_id=str(project["project_id"]),
                path=path,
                path_casefold=path_casefold,
                builder=builder,
                config=config,
            )
            files_payload.append(file_draft)
    finally:
        builder.close()

    apply_cfg = _apply_defaults_from_config(config.data)
    plan_payload = {
        "format": kdeplan.PLAN_FORMAT_VERSION,
        "project_id": str(project["project_id"]),
        "config_hash": config.config_hash,
        "lang": lang,
        "marker_flags": list(builder.marker_flags),
        "comment_prefixes": list(builder.comment_prefixes),
        "ai_flag": builder.ai_flag,
        "apply_defaults": {
            "mode": str(apply_cfg.get("mode", "strict")),
            "overwrite": str(apply_cfg.get("overwrite", "conservative")),
            "post_index": str(apply_cfg.get("post_index", "off")),
        },
        "files": files_payload,
    }

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
    config_data = None

    project = _load_project(project_root)
    project_id = str(plan.get("project_id") or project.get("project_id") or "")
    path_casefold = bool(project.get("path_casefold", os.name == "nt"))

    if post_index_flag:
        try:
            config = kdeconfig.load_config_from_root(project_root)
        except Exception as exc:
            typer.secho(f"Warning: post-index disabled ({exc}).", err=True)
            post_index_flag = False
        else:
            config_data = config.data
            try:
                workspace_conn = _ensure_workspace_db(
                    project_root,
                    project_id=str(project["project_id"]),
                    config_hash=config.config_hash,
                    config_data=config.data,
                )
            except Exception as exc:
                typer.secho(f"Warning: post-index disabled ({exc}).", err=True)
                post_index_flag = False

    defaults = plan.get("apply_defaults") if isinstance(plan, dict) else None
    defaults = defaults if isinstance(defaults, dict) else {}
    selected_mode = str(apply_mode or defaults.get("mode") or "strict")
    selected_overwrite = str(overwrite or defaults.get("overwrite") or "conservative")

    marker_flags = plan.get("marker_flags") or kdeapply.DEFAULT_MARKER_FLAGS
    comment_prefixes = plan.get("comment_prefixes") or kdeapply.DEFAULT_COMMENT_PREFIXES
    placeholder_patterns = plan.get("placeholder_patterns") or []
    lang = str(plan.get("lang", ""))

    files_written: list[str] = []
    files_skipped: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []
    entries_applied = 0

    post_index_warnings: list[str] = []
    files = plan.get("files") if isinstance(plan, dict) else None
    if not isinstance(files, list):
        raise typer.Exit(1)

    for file_item in files:
        if not isinstance(file_item, dict):
            continue
        file_path = str(file_item.get("file_path", ""))
        base_sha256 = str(file_item.get("base_sha256", ""))
        entries = list(file_item.get("entries", []))

        relpath_key = file_path.casefold() if path_casefold else file_path
        lock_path = locks.per_file_lock_path(
            project_root,
            locks.lock_id(project_id, relpath_key),
        )
        full_path = project_root / file_path
        file_result = kdeapply.apply_plan_to_file(
            file_path,
            entries,
            selected_mode,
            selected_overwrite,
            full_path=full_path,
            lock_path=lock_path,
            base_sha256=base_sha256,
            lang=lang,
            marker_flags=marker_flags,
            comment_prefixes=comment_prefixes,
            placeholder_patterns=placeholder_patterns,
        )

        if file_result.errors:
            errors.extend(file_result.errors)
            files_skipped.append(file_path)
            continue
        if file_result.warnings:
            warnings.extend(file_result.warnings)

        if file_result.skipped:
            files_skipped.append(file_path)
            continue
        if file_result.wrote:
            files_written.append(file_path)
            entries_applied += file_result.entries_applied
            if post_index_flag and workspace_conn is not None:
                try:
                    data = full_path.read_bytes()
                    sha256_hex = hashlib.sha256(data).hexdigest()
                    stat = full_path.stat()
                    workspace_tm.index_file_snapshot_tm(
                        workspace_conn,
                        file_path=file_path,
                        lang=lang,
                        bytes=data,
                        sha256=sha256_hex,
                        mtime_ns=stat.st_mtime_ns,
                        size=stat.st_size,
                        config=config_data,
                    )
                except Exception as exc:
                    post_index_warnings.append(f"post-index failed for {file_path}: {exc}")

    if workspace_conn is not None:
        workspace_conn.close()

    if errors:
        for error in errors:
            typer.secho(error, err=True)
        raise typer.Exit(1)

    if warnings or post_index_warnings:
        for warning in warnings + post_index_warnings:
            typer.secho(f"Warning: {warning}", err=True)

    typer.echo(
        f"Applied {entries_applied} entries "
        f"({len(files_written)} files written, {len(files_skipped)} skipped)."
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
    glossary: Optional[ExamplesMode] = typer.Option(None, "--glossary"),
) -> None:
    ctx = click.get_current_context()
    project_root = _project_root(ctx)
    project = _load_project(project_root)
    config = kdeconfig.load_config_from_root(project_root)
    _ = cache_write
    cache_mode = cache or "on"
    resolved_examples_mode = _examples_mode_from_config(config.data, examples)
    embedder = _maybe_embedder(
        resolved_examples_mode,
        _examples_embed_policy(config.data),
    )
    sqlite_vector_path = _sqlite_vector_path(project_root)
    path_casefold = bool(project.get("path_casefold", os.name == "nt"))

    apply_defaults = _apply_defaults_from_config(config.data)
    selected_mode = str(apply_mode or apply_defaults.get("mode") or "strict")
    selected_overwrite = str(overwrite or apply_defaults.get("overwrite") or "conservative")
    post_index_flag = apply_defaults.get("post_index") == "on"

    builder = kdeplan.PlanBuilder(
        project_root=project_root,
        project_id=str(project["project_id"]),
        config=config.data,
        config_hash=config.config_hash,
        embed_policy_hash=config.embed_policy_hash,
        lang=lang,
        cache=cache_mode,
        examples_mode=examples,
        glossary_mode=glossary,
        overwrite=overwrite,
        embedder=embedder,
        sqlite_vector_path=sqlite_vector_path,
    )

    combined_plan = None
    files_written: list[str] = []
    files_skipped: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    entries_applied = 0
    workspace_conn = None

    if post_index_flag:
        try:
            workspace_conn = _ensure_workspace_db(
                project_root,
                project_id=str(project["project_id"]),
                config_hash=config.config_hash,
                config_data=config.data,
            )
        except Exception as exc:
            typer.secho(f"Warning: post-index disabled ({exc}).", err=True)
            post_index_flag = False

    try:
        for path in _iter_po_paths(project_root, paths):
            file_plan = kdeplan.generate_plan_for_file(
                project_root=project_root,
                project_id=str(project["project_id"]),
                path=path,
                path_casefold=path_casefold,
                builder=builder,
                config=config,
            )

            file_path = str(file_plan.get("file_path", ""))
            base_sha256 = str(file_plan.get("base_sha256", ""))
            entries = list(file_plan.get("entries", []))
            relpath_key = file_path.casefold() if path_casefold else file_path
            lock_path = locks.per_file_lock_path(
                project_root,
                locks.lock_id(str(project["project_id"]), relpath_key),
            )
            full_path = project_root / file_path
            file_result = kdeapply.apply_plan_to_file(
                file_path,
                entries,
                selected_mode,
                selected_overwrite,
                full_path=full_path,
                lock_path=lock_path,
                base_sha256=base_sha256,
                lang=lang,
                marker_flags=builder.marker_flags,
                comment_prefixes=builder.comment_prefixes,
                placeholder_patterns=[],
            )

            if file_result.errors:
                errors.extend(file_result.errors)
                files_skipped.append(file_path)
                continue
            if file_result.warnings:
                warnings.extend(file_result.warnings)

            if file_result.skipped:
                files_skipped.append(file_path)
                continue
            if file_result.wrote:
                files_written.append(file_path)
                entries_applied += file_result.entries_applied
                if post_index_flag and workspace_conn is not None:
                    try:
                        data = full_path.read_bytes()
                        sha256_hex = hashlib.sha256(data).hexdigest()
                        stat = full_path.stat()
                        workspace_tm.index_file_snapshot_tm(
                            workspace_conn,
                            file_path=file_path,
                            lang=lang,
                            bytes=data,
                            sha256=sha256_hex,
                            mtime_ns=stat.st_mtime_ns,
                            size=stat.st_size,
                            config=config.data,
                        )
                    except Exception as exc:
                        warnings.append(f"post-index failed for {file_path}: {exc}")

            if out is not None:
                if combined_plan is None:
                    combined_plan = {
                        "format": kdeplan.PLAN_FORMAT_VERSION,
                        "project_id": str(project["project_id"]),
                        "config_hash": config.config_hash,
                        "lang": lang,
                        "marker_flags": list(builder.marker_flags),
                        "comment_prefixes": list(builder.comment_prefixes),
                        "ai_flag": builder.ai_flag,
                        "apply_defaults": {
                            "mode": str(apply_defaults.get("mode", "strict")),
                            "overwrite": str(apply_defaults.get("overwrite", "conservative")),
                            "post_index": str(apply_defaults.get("post_index", "off")),
                        },
                        "files": [],
                    }
                combined_plan["files"].append(file_plan)
    finally:
        builder.close()
        if workspace_conn is not None:
            workspace_conn.close()

    if errors:
        for error in errors:
            typer.secho(error, err=True)
        raise typer.Exit(1)

    if out is not None and combined_plan is not None:
        kdeplan.write_plan(out, combined_plan)

    if warnings:
        for warning in warnings:
            typer.secho(f"Warning: {warning}", err=True)
    typer.echo(
        f"Applied {entries_applied} entries "
        f"({len(files_written)} files written, {len(files_skipped)} skipped)."
    )


@app.command()
def index(
    paths: Optional[list[Path]] = typer.Argument(None),
    strict: bool = typer.Option(False, "--strict"),
) -> None:
    ctx = click.get_current_context()
    project_root = _project_root(ctx)
    project = _load_project(project_root)
    config = kdeconfig.load_config_from_root(project_root)

    errors: list[str] = []
    project_id = str(project["project_id"])
    path_casefold = bool(project.get("path_casefold"))
    conn = _ensure_workspace_db(
        project_root,
        project_id=project_id,
        config_hash=config.config_hash,
        config_data=config.data,
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
                    targets = config.data.get("languages", {}).get("targets", [])
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
                    config=config.data,
                )
            except Exception as exc:
                message = f"{relpath}: {exc}"
                if strict:
                    typer.secho(message, err=True)
                    raise typer.Exit(1)
                typer.secho(f"Warning: {message}", err=True)
                errors.append(message)
    finally:
        conn.close()

    if errors:
        raise typer.Exit(1)
    typer.echo("Workspace TM index updated.")


@reference_app.command("build")
def reference_build(
    paths: Optional[list[Path]] = typer.Argument(None),
    label: Optional[str] = typer.Option(None, "--label"),
) -> None:
    ctx = click.get_current_context()
    project_root = _project_root(ctx)
    try:
        project = _load_project(project_root)
        config = kdeconfig.load_config_from_root(project_root)
        snapshot = kderef.build_reference_snapshot(
            project_root,
            project_id=str(project["project_id"]),
            path_casefold=bool(project.get("path_casefold")),
            config=config.data,
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
        project = _load_project(project_root)
        config = kdeconfig.load_config_from_root(project_root)
    except Exception as exc:
        typer.secho(f"Examples build failed: {exc}", err=True)
        raise typer.Exit(1)

    if lang == "all":
        targets = config.data.get("languages", {}).get("targets", [])
        if not isinstance(targets, list) or not targets:
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
                        project_id=str(project["project_id"]),
                        config_hash=config.config_hash,
                        embed_policy_hash=config.embed_policy_hash,
                    )
                    db.conn.close()
                    typer.echo(f"Examples cache already current for {target_lang}.")
                    continue
            except Exception:
                pass

        try:
            embedder = _require_embedder(_examples_embed_policy(config.data))
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
                project_id=str(project["project_id"]),
                config_hash=config.config_hash,
                config_data=config.data,
            )
            try:
                kdeexamples.build_examples_db_from_workspace(
                    conn,
                    output_path=output_path,
                    lang=target_lang,
                    config=config.data,
                    project_id=str(project["project_id"]),
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
                    config=config.data,
                    project_id=str(project["project_id"]),
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
            project_id=str(project["project_id"]),
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
        project = _load_project(project_root)
        config = kdeconfig.load_config_from_root(project_root)
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
                expected_project_id=str(project["project_id"]),
                expected_config_hash=config.config_hash,
                expected_kind="glossary",
                expected_normalization_id=_glossary_normalization_id(config.data),
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
            config=config.data,
            project_id=str(project["project_id"]),
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
        project = _load_project(project_root)
        config = kdeconfig.load_config_from_root(project_root)
        report = kdegc.gc_workspace_tm(
            project_root,
            project_id=str(project["project_id"]),
            config_hash=config.config_hash,
            config_data=config.data,
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
