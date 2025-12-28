from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
import json
import os

from kdeai import hash as kdehash
from kdeai.config import Config, load_config
from kdeai.utils.win32 import windows_file_id


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


def _generate_project_id(project_dir: Path) -> tuple[str, str]:
    if os.name == "nt":
        file_id = windows_file_id(project_dir)
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
    def load_project_data(cls, root: Path) -> dict:
        project_path = _project_path(root)
        return _read_project(project_path)

    @classmethod
    def ensure_project_data(cls, root: Path) -> dict:
        project_path = _project_path(root)
        if project_path.exists():
            return cls.load_project_data(root)

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
        return payload

    @classmethod
    def load(cls, root: Path) -> "Project":
        config_path = _project_dir(root) / "config.json"
        project_data = cls.load_project_data(root)
        config = load_config(config_path)
        return cls(root=root, project_data=project_data, config=config)

    @classmethod
    def load_or_init(cls, root: Path) -> "Project":
        cls.ensure_project_data(root)
        return cls.load(root)
