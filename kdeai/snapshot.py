from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

from kdeai import locks


@dataclass(frozen=True)
class LockedRead:
    file_path: str
    bytes: bytes
    sha256: str
    mtime_ns: int
    size: int


def _read_file_snapshot_unlocked(file_path: Path, *, relpath: str) -> LockedRead:
    data = file_path.read_bytes()
    sha256_hex = hashlib.sha256(data).hexdigest()
    stat = file_path.stat()

    return LockedRead(
        file_path=relpath,
        bytes=data,
        sha256=sha256_hex,
        mtime_ns=stat.st_mtime_ns,
        size=stat.st_size,
    )


def locked_read_file(file_path: Path, lock_path: Path, *, relpath: str) -> LockedRead:
    with locks.acquire_file_lock(lock_path):
        return _read_file_snapshot_unlocked(file_path, relpath=relpath)
