from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

from kdeai import db as kdedb


@dataclass(frozen=True)
class GcReport:
    ttl_days: int
    cutoff_iso: str
    files_deleted: int
    sources_deleted: int
    best_translations_deleted: int


def _workspace_tm_settings(config: Mapping[str, object]) -> tuple[int, str]:
    sqlite_cfg = config.get("sqlite") if isinstance(config, Mapping) else None
    workspace_cfg = sqlite_cfg.get("workspace_tm") if isinstance(sqlite_cfg, Mapping) else None
    busy_timeout_ms = 50
    synchronous = "NORMAL"
    if isinstance(workspace_cfg, Mapping):
        synchronous = str(workspace_cfg.get("synchronous", synchronous)).upper()
        timeout_cfg = workspace_cfg.get("busy_timeout_ms")
        if isinstance(timeout_cfg, dict):
            busy_timeout_ms = int(timeout_cfg.get("write", busy_timeout_ms))
        elif isinstance(timeout_cfg, int):
            busy_timeout_ms = int(timeout_cfg)
    return busy_timeout_ms, synchronous


def gc_workspace_tm(
    project_root: Path,
    *,
    project_id: str,
    config_hash: str,
    config_data: Mapping[str, object],
    ttl_days: int,
) -> GcReport:
    if ttl_days <= 0:
        raise ValueError("ttl_days must be a positive integer")
    db_path = project_root / ".kdeai" / "cache" / "workspace.tm.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"workspace tm not found: {db_path}")

    busy_timeout_ms, synchronous = _workspace_tm_settings(config_data)
    conn = kdedb.connect_workspace_tm(
        db_path,
        busy_timeout_ms=busy_timeout_ms,
        synchronous=synchronous,
    )
    try:
        meta = kdedb.read_meta(conn)
        kdedb.validate_meta(
            meta,
            expected_project_id=project_id,
            expected_config_hash=config_hash,
            expected_kind="workspace_tm",
        )

        cutoff = datetime.now(timezone.utc) - timedelta(days=int(ttl_days))
        cutoff_iso = cutoff.isoformat()

        cursor = conn.cursor()
        cursor.execute("BEGIN")
        try:
            files_deleted = int(
                cursor.execute(
                    "SELECT COUNT(*) FROM files WHERE indexed_at < ?",
                    (cutoff_iso,),
                ).fetchone()[0]
            )
            cursor.execute(
                "DELETE FROM files WHERE indexed_at < ?",
                (cutoff_iso,),
            )
            best_deleted = cursor.execute(
                "DELETE FROM best_translations "
                "WHERE NOT EXISTS ("
                "SELECT 1 FROM translations t "
                "WHERE t.source_key = best_translations.source_key "
                "AND t.lang = best_translations.lang"
                ")"
            ).rowcount
            sources_deleted = cursor.execute(
                "DELETE FROM sources WHERE NOT EXISTS ("
                "SELECT 1 FROM translations t WHERE t.source_key = sources.source_key"
                ")"
            ).rowcount
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    finally:
        conn.close()

    return GcReport(
        ttl_days=int(ttl_days),
        cutoff_iso=cutoff_iso,
        files_deleted=files_deleted,
        sources_deleted=int(sources_deleted or 0),
        best_translations_deleted=int(best_deleted or 0),
    )
