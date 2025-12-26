from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from kdeai.config import Config, load_config


@dataclass(frozen=True)
class Project:
    root: Path
    project_data: dict
    config: Config

    @classmethod
    def load(cls, root: Path) -> "Project":
        project_path = root / ".kdeai" / "project.json"
        config_path = root / ".kdeai" / "config.json"
        project_data = json.loads(project_path.read_text(encoding="utf-8"))
        config = load_config(config_path)
        return cls(root=root, project_data=project_data, config=config)
