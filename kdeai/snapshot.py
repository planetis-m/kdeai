from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

from kdeai import locks


@dataclass(frozen=True)
class LockedRead:
    bytes: bytes
    sha256: str
    mtime_ns: int
    size: int


def _read_file_snapshot_unlocked(file_path: Path) -> LockedRead:
    data = Path(file_path).read_bytes()
    sha256_hex = hashlib.sha256(data).hexdigest()
    stat = Path(file_path).stat()

    return LockedRead(
        bytes=data,
        sha256=sha256_hex,
        mtime_ns=stat.st_mtime_ns,
        size=stat.st_size,
    )


def locked_read_file(file_path: Path, lock_path: Path) -> LockedRead:
    with locks.acquire_file_lock(lock_path):
        return _read_file_snapshot_unlocked(file_path)
