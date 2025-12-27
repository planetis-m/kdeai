import sqlite3
import tempfile
import unittest
from pathlib import Path

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
            finally:
                ro_conn.close()


if __name__ == "__main__":
    unittest.main()
