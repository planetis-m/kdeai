import sqlite3
import tempfile
import unittest
from pathlib import Path

import pytest

from kdeai import db as kdedb


class TestConnectReadonly(unittest.TestCase):
    def test_connect_readonly_enforces_query_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            conn = sqlite3.connect(str(db_path))
            conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
            conn.execute("INSERT INTO items (name) VALUES ('one')")
            conn.commit()
            conn.close()

            ro_conn = kdedb.connect_readonly(db_path)
            try:
                query_only = ro_conn.execute("PRAGMA query_only").fetchone()[0]
                self.assertEqual(query_only, 1)
                with self.assertRaises(sqlite3.OperationalError):
                    ro_conn.execute("INSERT INTO items (name) VALUES ('two')")
                with self.assertRaises(sqlite3.OperationalError):
                    ro_conn.execute("CREATE TABLE other (id INTEGER PRIMARY KEY)")
            finally:
                ro_conn.close()


if __name__ == "__main__":
    unittest.main()


def test_try_enable_sqlite_vector_returns_false_for_missing_extension() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        assert kdedb.try_enable_sqlite_vector(conn, extension_path="missing_vector.so") is False
    finally:
        conn.close()


def test_validate_meta_rejects_unsupported_schema_version() -> None:
    meta = {
        "schema_version": "2",
        "kind": "workspace_tm",
        "project_id": "proj",
        "config_hash": "hash",
        "created_at": "2024-01-01T00:00:00Z",
    }
    with pytest.raises(ValueError):
        kdedb.validate_meta(
            meta,
            expected_project_id="proj",
            expected_config_hash="hash",
            expected_kind="workspace_tm",
        )
