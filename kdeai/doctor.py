from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import portalocker

from kdeai import config as kdeconfig
from kdeai import db as kdedb
from kdeai import examples as kdeexamples
from kdeai import glossary as kdeglo
from kdeai import locks


@dataclass(frozen=True)
class DoctorReport:
    errors: list[str]
    warnings: list[str]
    notes: list[str]


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


def _require_keys(payload: dict, keys: list[str], label: str) -> None:
    missing = [key for key in keys if key not in payload or payload[key] in ("", None)]
    if missing:
        raise ValueError(f"{label} missing keys: {', '.join(missing)}")


def _check_run_lock(project_root: Path, report: DoctorReport) -> None:
    lock_path = locks.run_lock_path(project_root)
    try:
        lock = portalocker.Lock(str(lock_path), mode="a+", timeout=0)
        lock.acquire()
    except portalocker.exceptions.LockException:
        report.notes.append("run lock held (expected for active process)")
        return
    except Exception as exc:
        report.errors.append(f"run lock: {exc}")
        return
    try:
        report.notes.append("run lock acquired successfully")
    finally:
        lock.release()


def _safe_remove(path: Path, *, report: DoctorReport, label: str) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        report.errors.append(f"{label} cleanup failed: {exc}")


def _check_workspace_tm(
    project_root: Path,
    *,
    project_id: str,
    config_hash: str,
    repair_cache: bool,
    report: DoctorReport,
) -> None:
    db_path = project_root / ".kdeai" / "cache" / "workspace.tm.sqlite"
    if not db_path.exists():
        report.warnings.append("workspace tm: missing")
        return
    try:
        conn = kdedb.connect_readonly(db_path)
        try:
            kdedb.validate_meta_table(
                conn,
                expected_project_id=project_id,
                expected_config_hash=config_hash,
                expected_kind="workspace_tm",
            )
            files = int(conn.execute("SELECT COUNT(*) FROM files").fetchone()[0])
            translations = int(conn.execute("SELECT COUNT(*) FROM translations").fetchone()[0])
            report.notes.append(
                f"workspace tm: {files} files, {translations} translations"
            )
        finally:
            conn.close()
    except Exception as exc:
        if repair_cache:
            _safe_remove(db_path, report=report, label="workspace tm")
            report.warnings.append(f"workspace tm removed: {exc}")
        else:
            report.errors.append(f"workspace tm: {exc}")


def _check_reference_pointer(
    project_root: Path,
    *,
    project_id: str,
    config_hash: str,
    repair_cache: bool,
    report: DoctorReport,
) -> None:
    pointer_path = project_root / ".kdeai" / "cache" / "reference" / "reference.current.json"
    if not pointer_path.exists():
        report.warnings.append("reference pointer: missing")
        return
    try:
        pointer = _read_json(pointer_path, "reference.current.json")
        _require_keys(pointer, ["snapshot_id", "db_file", "created_at"], "reference pointer")
        db_path = pointer_path.parent / str(pointer["db_file"])
    except Exception as exc:
        if repair_cache:
            _safe_remove(pointer_path, report=report, label="reference pointer")
            report.warnings.append(f"reference pointer removed: {exc}")
        else:
            report.errors.append(f"reference pointer: {exc}")
        return

    if not db_path.exists():
        if repair_cache:
            _safe_remove(pointer_path, report=report, label="reference pointer")
            report.warnings.append("reference pointer removed: db missing")
        else:
            report.errors.append(f"reference db missing: {db_path}")
        return

    try:
        conn = kdedb.connect_readonly(db_path)
        try:
            kdedb.validate_meta_table(
                conn,
                expected_project_id=project_id,
                expected_config_hash=config_hash,
                expected_kind="reference_tm",
            )
        finally:
            conn.close()
    except Exception as exc:
        if repair_cache:
            _safe_remove(pointer_path, report=report, label="reference pointer")
            _safe_remove(db_path, report=report, label="reference db")
            report.warnings.append(f"reference cache removed: {exc}")
        else:
            report.errors.append(f"reference db: {exc}")


def _validate_examples_pointer(
    pointer: dict,
    *,
    expected_scope: str,
    expected_lang: str,
    expected_embed_policy_hash: str,
) -> None:
    required = [
        "ex_id",
        "scope",
        "lang",
        "db_file",
        "created_at",
        "embed_policy_hash",
        "embedding_model_id",
        "embedding_dim",
        "embedding_distance",
        "vector_encoding",
        "embedding_normalization",
        "source_snapshot",
    ]
    _require_keys(pointer, required, "examples pointer")
    if str(pointer["scope"]) != expected_scope:
        raise ValueError("examples pointer scope mismatch")
    if str(pointer["lang"]) != expected_lang:
        raise ValueError("examples pointer lang mismatch")
    if str(pointer["embed_policy_hash"]) != expected_embed_policy_hash:
        raise ValueError("examples pointer embed_policy_hash mismatch")
    source_snapshot = pointer.get("source_snapshot")
    if not isinstance(source_snapshot, dict):
        raise ValueError("examples pointer source_snapshot must be an object")
    _require_keys(source_snapshot, ["kind"], "examples pointer source_snapshot")
    if str(source_snapshot["kind"]) == "reference_tm":
        _require_keys(source_snapshot, ["snapshot_id"], "examples pointer source_snapshot")


def _check_examples_pointers(
    project_root: Path,
    *,
    project_id: str,
    config_hash: str,
    embed_policy_hash: str,
    repair_cache: bool,
    report: DoctorReport,
    valid_example_paths: list[Path],
) -> None:
    for scope in ("workspace", "reference"):
        examples_dir = project_root / ".kdeai" / "cache" / "examples" / scope
        if not examples_dir.exists():
            continue
        for pointer_path in sorted(examples_dir.glob("examples.*.*.current.json")):
            parts = pointer_path.name.split(".")
            file_scope = parts[1] if len(parts) > 2 else ""
            file_lang = parts[2] if len(parts) > 2 else ""
            try:
                pointer = _read_json(pointer_path, f"examples {scope} pointer")
                _validate_examples_pointer(
                    pointer,
                    expected_scope=file_scope or scope,
                    expected_lang=file_lang or str(pointer.get("lang") or ""),
                    expected_embed_policy_hash=embed_policy_hash,
                )
                db_path = pointer_path.parent / str(pointer["db_file"])
            except Exception as exc:
                if repair_cache:
                    _safe_remove(pointer_path, report=report, label="examples pointer")
                    report.warnings.append(f"examples pointer removed: {exc}")
                else:
                    report.errors.append(f"examples pointer {pointer_path}: {exc}")
                continue

            if not db_path.exists():
                if repair_cache:
                    _safe_remove(pointer_path, report=report, label="examples pointer")
                    report.warnings.append(f"examples pointer removed: db missing ({db_path})")
                else:
                    report.errors.append(f"examples db missing: {db_path}")
                continue

            try:
                db = kdeexamples.open_examples_db(
                    db_path,
                    project_id=project_id,
                    config_hash=config_hash,
                    embed_policy_hash=embed_policy_hash,
                )
            except Exception as exc:
                if repair_cache:
                    _safe_remove(pointer_path, report=report, label="examples pointer")
                    _safe_remove(db_path, report=report, label="examples db")
                    report.warnings.append(f"examples cache removed: {exc}")
                else:
                    report.errors.append(f"examples db {db_path}: {exc}")
                continue
            else:
                db.conn.close()
                valid_example_paths.append(db_path)


def _validate_glossary_pointer(pointer: dict) -> None:
    _require_keys(pointer, ["snapshot_id", "db_file", "created_at", "source_snapshot"], "glossary pointer")
    source_snapshot = pointer.get("source_snapshot")
    if not isinstance(source_snapshot, dict):
        raise ValueError("glossary pointer source_snapshot must be an object")
    _require_keys(source_snapshot, ["kind", "snapshot_id"], "glossary pointer source_snapshot")


def _check_glossary_pointer(
    project_root: Path,
    *,
    project_id: str,
    config_hash: str,
    normalization_id: str,
    repair_cache: bool,
    report: DoctorReport,
) -> None:
    pointer_path = project_root / ".kdeai" / "cache" / "glossary" / "glossary.current.json"
    if not pointer_path.exists():
        report.warnings.append("glossary pointer: missing")
        return
    try:
        pointer = _read_json(pointer_path, "glossary.current.json")
        _validate_glossary_pointer(pointer)
        db_path = pointer_path.parent / str(pointer["db_file"])
    except Exception as exc:
        if repair_cache:
            _safe_remove(pointer_path, report=report, label="glossary pointer")
            report.warnings.append(f"glossary pointer removed: {exc}")
        else:
            report.errors.append(f"glossary pointer: {exc}")
        return

    if not db_path.exists():
        if repair_cache:
            _safe_remove(pointer_path, report=report, label="glossary pointer")
            report.warnings.append(f"glossary pointer removed: db missing ({db_path})")
        else:
            report.errors.append(f"glossary db missing: {db_path}")
        return

    try:
        conn = kdedb.connect_readonly(db_path)
        try:
            kdedb.validate_meta_table(
                conn,
                expected_project_id=project_id,
                expected_config_hash=config_hash,
                expected_kind="glossary",
                expected_normalization_id=normalization_id,
            )
        finally:
            conn.close()
    except Exception as exc:
        if repair_cache:
            _safe_remove(pointer_path, report=report, label="glossary pointer")
            _safe_remove(db_path, report=report, label="glossary db")
            report.warnings.append(f"glossary cache removed: {exc}")
        else:
            report.errors.append(f"glossary db: {exc}")


def _run_examples_smoke_test(
    example_paths: list[Path],
    *,
    project_id: str,
    config_hash: str,
    embed_policy_hash: str,
    sqlite_vector_path: str,
    report: DoctorReport,
) -> None:
    for path in example_paths:
        try:
            db = kdeexamples.open_examples_db(
                path,
                project_id=project_id,
                config_hash=config_hash,
                embed_policy_hash=embed_policy_hash,
            )
            try:
                kdedb.enable_sqlite_vector(db.conn, extension_path=sqlite_vector_path)
                row = db.conn.execute("SELECT embedding FROM examples LIMIT 1").fetchone()
                if row is None:
                    report.warnings.append(f"examples smoke test skipped (empty): {path}")
                    continue
                embedding = row[0]
                matches = kdeexamples.query_examples(
                    db,
                    query_embedding=embedding,
                    top_n=1,
                )
                if not matches:
                    report.errors.append(f"examples smoke test failed: {path}")
            finally:
                db.conn.close()
        except Exception as exc:
            report.errors.append(f"examples smoke test error for {path}: {exc}")


def run_doctor(
    project_root: Path,
    *,
    repair_cache: bool = False,
    sqlite_vector_path: str | None = None,
) -> DoctorReport:
    report = DoctorReport(errors=[], warnings=[], notes=[])
    _check_run_lock(project_root, report)

    project = None
    config = None
    try:
        project = _read_json(project_root / ".kdeai" / "project.json", "project.json")
        _require_keys(project, ["project_id"], "project.json")
    except Exception as exc:
        report.errors.append(str(exc))

    try:
        config = kdeconfig.load_config_from_root(project_root)
    except Exception as exc:
        report.errors.append(str(exc))

    if project is None or config is None:
        return report

    project_id = str(project["project_id"])
    config_hash = config.config_hash

    _check_workspace_tm(
        project_root,
        project_id=project_id,
        config_hash=config_hash,
        repair_cache=repair_cache,
        report=report,
    )
    _check_reference_pointer(
        project_root,
        project_id=project_id,
        config_hash=config_hash,
        repair_cache=repair_cache,
        report=report,
    )

    valid_example_paths: list[Path] = []
    _check_examples_pointers(
        project_root,
        project_id=project_id,
        config_hash=config_hash,
        embed_policy_hash=config.embed_policy_hash,
        repair_cache=repair_cache,
        report=report,
        valid_example_paths=valid_example_paths,
    )

    normalization_id = str(config.prompt.glossary.normalization_id or kdeglo.NORMALIZATION_ID)
    _check_glossary_pointer(
        project_root,
        project_id=project_id,
        config_hash=config_hash,
        normalization_id=normalization_id,
        repair_cache=repair_cache,
        report=report,
    )

    if sqlite_vector_path and valid_example_paths:
        _run_examples_smoke_test(
            valid_example_paths,
            project_id=project_id,
            config_hash=config_hash,
            embed_policy_hash=config.embed_policy_hash,
            sqlite_vector_path=sqlite_vector_path,
            report=report,
        )

    return report
