from __future__ import annotations

from pathlib import Path
import sqlite3

from kdeai import db as kdedb


def resolve_sqlite_vector_path(project_root: Path) -> str | None:
    candidate = project_root / "vector.so"
    if candidate.exists():
        return str(candidate)
    return None


def enable_sqlite_vector(conn: sqlite3.Connection, *, extension_path: str) -> bool:
    return kdedb.try_enable_sqlite_vector(conn, extension_path=extension_path)
