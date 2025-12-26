from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import hashlib

import portalocker


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def lock_id(project_id: str, relpath_key: str) -> str:
    return _sha256_hex(f"{project_id}\n{relpath_key}")


def run_lock_path(project_root: Path) -> Path:
    return project_root / ".kdeai" / "run.lock"


def per_file_lock_path(project_root: Path, file_lock_id: str) -> Path:
    return project_root / ".kdeai" / "locks" / f"{file_lock_id}.lock"


@contextmanager
def acquire_run_lock(project_root: Path):
    lock_path = run_lock_path(project_root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = portalocker.Lock(str(lock_path), mode="a+", timeout=0)
    lock.acquire()
    try:
        yield lock
    finally:
        lock.release()


@contextmanager
def acquire_file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = portalocker.Lock(str(lock_path), mode="a+", timeout=0)
    lock.acquire()
    try:
        yield lock
    finally:
        lock.release()
