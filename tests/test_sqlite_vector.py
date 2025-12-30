import tempfile
import unittest
from pathlib import Path

from kdeai import sqlite_vector


class TestResolveSqliteVectorPath(unittest.TestCase):
    def test_returns_none_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            self.assertIsNone(sqlite_vector.resolve_sqlite_vector_path(project_root))

    def test_returns_path_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            vector_path = project_root / "vector.so"
            vector_path.write_bytes(b"")
            self.assertEqual(
                sqlite_vector.resolve_sqlite_vector_path(project_root),
                str(vector_path),
            )
