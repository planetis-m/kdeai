from __future__ import annotations

from pathlib import Path

import sqlite3
import array

from kdeai import db as kdedb


def test_sqlite_vector_extension_loads(tmp_path: Path, monkeypatch) -> None:
    project_root = Path(__file__).resolve().parents[1]
    vector_path = project_root / "vector.so"
    assert vector_path.exists(), "vector.so must exist in project root"

    monkeypatch.chdir(project_root)

    db_path = tmp_path / "vector-test.sqlite"
    conn = kdedb.get_db_connection(db_path)
    try:
        conn.execute(
            "CREATE TABLE images (id INTEGER PRIMARY KEY, embedding BLOB, label TEXT)"
        )
        blob = array.array("f", [0.1, 0.2, 0.3]).tobytes()
        conn.execute("INSERT INTO images (embedding, label) VALUES (?, 'cat')", (blob,))
        conn.execute(
            "INSERT INTO images (embedding, label) "
            "VALUES (vector_as_f32('[0.2, 0.1, 0.0]'), 'dog')"
        )
        conn.execute(
            "SELECT vector_init('images', 'embedding', 'type=FLOAT32,dimension=3')"
        )
        conn.execute("SELECT vector_quantize('images', 'embedding')")
        rows = conn.execute(
            "SELECT e.id "
            "FROM images AS e "
            "JOIN vector_quantize_scan('images', 'embedding', vector_as_f32(?), 1) AS v "
            "ON e.id = v.rowid",
            ("[0.1, 0.2, 0.3]",),
        ).fetchall()
        assert rows, "vector_quantize_scan should return at least one match"
    finally:
        conn.close()
