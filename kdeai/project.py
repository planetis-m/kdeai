from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
import json
import os

from kdeai import hash as kdehash
from kdeai.config import Config, load_config


def _project_dir(root: Path) -> Path:
    return root / ".kdeai"


def _project_path(root: Path) -> Path:
    return _project_dir(root) / "project.json"


def _path_casefold() -> bool:
    return os.name == "nt"


def _read_project(project_path: Path) -> dict:
    payload = json.loads(project_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"project.json must be a JSON object: {project_path}")
    if "project_id" not in payload:
        raise ValueError(f"project.json missing project_id: {project_path}")
    return payload


def _windows_file_id(path: Path) -> tuple[int, int] | None:
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return None

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _BY_HANDLE_FILE_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", wintypes.DWORD),
            ("ftCreationTime", wintypes.FILETIME),
            ("ftLastAccessTime", wintypes.FILETIME),
            ("ftLastWriteTime", wintypes.FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD),
            ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD),
            ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD),
            ("nFileIndexLow", wintypes.DWORD),
        ]

    FILE_SHARE_READ = 0x00000001
    FILE_SHARE_WRITE = 0x00000002
    FILE_SHARE_DELETE = 0x00000004
    OPEN_EXISTING = 3
    FILE_FLAG_BACKUP_SEMANTICS = 0x02000000

    handle = kernel32.CreateFileW(
        str(path),
        0,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        None,
        OPEN_EXISTING,
        FILE_FLAG_BACKUP_SEMANTICS,
        None,
    )
    if handle == wintypes.HANDLE(-1).value:
        return None
    try:
        info = _BY_HANDLE_FILE_INFORMATION()
        if not kernel32.GetFileInformationByHandle(handle, ctypes.byref(info)):
            return None
        file_index = (int(info.nFileIndexHigh) << 32) | int(info.nFileIndexLow)
        volume_serial = int(info.dwVolumeSerialNumber)
        return volume_serial, file_index
    finally:
        kernel32.CloseHandle(handle)


def _generate_project_id(project_dir: Path) -> tuple[str, str]:
    if os.name == "nt":
        file_id = _windows_file_id(project_dir)
        if file_id is not None:
            volume_serial, file_index = file_id
            payload = f"{volume_serial}\n{file_index}"
            return kdehash.sha256_hex_text(payload), "win_file_id"
    try:
        stat = project_dir.stat()
        if os.name != "nt" and getattr(stat, "st_ino", None) is not None:
            payload = f"{stat.st_dev}\n{stat.st_ino}"
            return kdehash.sha256_hex_text(payload), "posix_dev_ino"
    except OSError:
        pass

    payload = str(project_dir.resolve())
    return kdehash.sha256_hex_text(payload), "realpath_fallback"


@dataclass(frozen=True)
class Project:
    root: Path
    project_data: dict
    config: Config

    @classmethod
    def load(cls, root: Path) -> "Project":
        project_path = _project_path(root)
        config_path = _project_dir(root) / "config.json"
        project_data = _read_project(project_path)
        config = load_config(config_path)
        return cls(root=root, project_data=project_data, config=config)

    @classmethod
    def load_or_init(cls, root: Path) -> "Project":
        project_path = _project_path(root)
        if project_path.exists():
            return cls.load(root)

        project_dir = _project_dir(root)
        project_dir.mkdir(parents=True, exist_ok=True)
        project_id, method = _generate_project_id(project_dir)
        payload = {
            "format": 1,
            "project_id": project_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "path_casefold": _path_casefold(),
            "method": method,
        }
        project_path.write_text(
            kdehash.canonical_json(payload),
            encoding="utf-8",
        )
        return cls.load(root)
