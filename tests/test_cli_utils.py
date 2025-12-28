from __future__ import annotations

import json
from pathlib import Path

from kdeai.cli import _next_pointer_id


def _write_pointer(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_next_pointer_id_missing_file(tmp_path: Path) -> None:
    pointer_path = tmp_path / "pointer.json"
    assert _next_pointer_id(pointer_path, "snapshot_id") == 1


def test_next_pointer_id_increments(tmp_path: Path) -> None:
    pointer_path = tmp_path / "pointer.json"
    _write_pointer(pointer_path, {"snapshot_id": 4})
    assert _next_pointer_id(pointer_path, "snapshot_id") == 5
