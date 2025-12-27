import math
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from conftest import build_config
from kdeai import examples


def _base_config(*, allow_ai_generated=False, min_review_status="reviewed"):
    return build_config(
        {
            "tm": {"selection": {"review_status_order": ["reviewed", "draft"]}},
            "prompt": {
                "examples": {
                    "embedding_policy": {
                        "dim": 2,
                    },
                    "eligibility": {
                        "min_review_status": min_review_status,
                        "allow_ai_generated": allow_ai_generated,
                    },
                }
            },
        }
    )


def _write_examples_meta_db(
    path: Path,
    *,
    project_id: str = "proj",
    config_hash: str = "config",
    embed_policy_hash: str = "embed-hash",
) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    meta = _example_meta(
        project_id=project_id,
        config_hash=config_hash,
        embed_policy_hash=embed_policy_hash,
    )
    conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta.items())
    conn.commit()
    conn.close()


def _example_meta(
    *,
    project_id: str = "proj",
    config_hash: str = "config",
    embed_policy_hash: str = "embed-hash",
) -> dict[str, str]:
    return {
        "schema_version": "1",
        "kind": "examples",
        "project_id": project_id,
        "config_hash": config_hash,
        "created_at": "2024-01-01T00:00:00Z",
        "embed_policy_hash": embed_policy_hash,
        "embedding_model_id": "test-model",
        "embedding_dim": "2",
        "embedding_distance": "cosine",
        "vector_encoding": "float32_le",
        "embedding_normalization": "none",
        "require_finite": "1",
        "examples_scope": "workspace",
        "examples_lang": "de",
        "source_snapshot_kind": "workspace_tm",
        "source_snapshot_id": "",
    }


class TestExamplesEligibility(unittest.TestCase):
    def test_singular_requires_non_empty(self):
        config = _base_config()
        rows = [
            (
                "key1",
                "ctx:\nid:Hello\npl:",
                "",
                "de",
                "",
                "{}",
                "reviewed",
                0,
                "hash",
                "file.po",
            )
        ]

        def embedder(texts):
            return [[0.1, 0.2] for _ in texts]

        payload = examples._build_examples_rows(
            rows,
            lang="de",
            config=config,
            embedder=embedder,
            embedding_dim=2,
            embedding_normalization="none",
            require_finite=True,
            include_file_sha256=False,
        )
        self.assertEqual(payload, [])


class TestQueryExamples(unittest.TestCase):
    def test_query_examples_top_n_zero_skips_sql(self):
        db = examples.ExamplesDb(
            conn=mock.Mock(),
            meta={},
            embedding_dim=2,
            embedding_distance="cosine",
            vector_encoding="float32_le",
            embedding_normalization="none",
            require_finite=True,
        )
        result = examples.query_examples(
            db,
            query_embedding=[0.1, 0.2],
            top_n=0,
        )
        self.assertEqual(result, [])
        db.conn.execute.assert_not_called()

    def test_plural_accepts_any_non_empty_form(self):
        config = _base_config()
        rows = [
            (
                "key1",
                "ctx:\nid:Item\npl:Items",
                "Items",
                "de",
                "",
                '{"0":"","1":"Dinge"}',
                "reviewed",
                0,
                "hash",
                "file.po",
                "sha",
            )
        ]

        def embedder(texts):
            return [[0.1, 0.2] for _ in texts]

        payload = examples._build_examples_rows(
            rows,
            lang="de",
            config=config,
            embedder=embedder,
            embedding_dim=2,
            embedding_normalization="none",
            require_finite=True,
            include_file_sha256=True,
        )
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0].file_sha256, "sha")

    def test_review_status_threshold(self):
        config = _base_config(min_review_status="reviewed")
        rows = [
            (
                "key1",
                "ctx:\nid:Hello\npl:",
                "",
                "de",
                "Hallo",
                "{}",
                "draft",
                0,
                "hash",
                "file.po",
            )
        ]

        def embedder(texts):
            return [[0.1, 0.2] for _ in texts]

        payload = examples._build_examples_rows(
            rows,
            lang="de",
            config=config,
            embedder=embedder,
            embedding_dim=2,
            embedding_normalization="none",
            require_finite=True,
            include_file_sha256=False,
        )
        self.assertEqual(payload, [])

    def test_ai_generated_filtering(self):
        config = _base_config(allow_ai_generated=False)
        rows = [
            (
                "key1",
                "ctx:\nid:Hello\npl:",
                "",
                "de",
                "Hallo",
                "{}",
                "reviewed",
                1,
                "hash",
                "file.po",
            )
        ]

        def embedder(texts):
            return [[0.1, 0.2] for _ in texts]

        payload = examples._build_examples_rows(
            rows,
            lang="de",
            config=config,
            embedder=embedder,
            embedding_dim=2,
            embedding_normalization="none",
            require_finite=True,
            include_file_sha256=False,
        )
        self.assertEqual(payload, [])

        config = _base_config(allow_ai_generated=True)
        payload = examples._build_examples_rows(
            rows,
            lang="de",
            config=config,
            embedder=embedder,
            embedding_dim=2,
            embedding_normalization="none",
            require_finite=True,
            include_file_sha256=False,
        )
        self.assertEqual(len(payload), 1)

    def test_review_status_order_requires_min_present(self):
        config = _base_config(min_review_status="approved")
        rows = [
            (
                "key1",
                "ctx:\nid:Hello\npl:",
                "",
                "de",
                "Hallo",
                "{}",
                "reviewed",
                0,
                "hash",
                "file.po",
            )
        ]

        def embedder(texts):
            return [[0.1, 0.2] for _ in texts]

        with self.assertRaises(ValueError):
            examples._build_examples_rows(
                rows,
                lang="de",
                config=config,
                embedder=embedder,
                embedding_dim=2,
                embedding_normalization="none",
                require_finite=True,
                include_file_sha256=False,
            )

    def test_build_examples_rows_snapshot(self):
        config = _base_config()
        rows = [
            (
                "key1",
                "ctx:\nid:Hello\npl:",
                "",
                "de",
                "Hallo",
                "{}",
                "reviewed",
                0,
                "hash",
                "file.po",
                "",
            ),
            (
                "key2",
                "ctx:\nid:Items\npl:Items",
                "Items",
                "de",
                "",
                '{"0":"","1":"Dinge"}',
                "reviewed",
                0,
                "hash2",
                "file.po",
                "sha",
            ),
        ]

        def embedder(texts):
            return [[1.0, 2.0] for _ in texts]

        payload = examples._build_examples_rows(
            rows,
            lang="de",
            config=config,
            embedder=embedder,
            embedding_dim=2,
            embedding_normalization="none",
            require_finite=True,
            include_file_sha256=True,
        )
        self.assertEqual(len(payload), 2)
        self.assertEqual(
            payload[0],
            examples.ExampleRow(
                source_key="key1",
                source_text="ctx:\nid:Hello\npl:",
                lang="de",
                msgstr="Hallo",
                msgstr_plural="{}",
                review_status="reviewed",
                is_ai_generated=0,
                translation_hash="hash",
                file_path="file.po",
                file_sha256="",
                embedding=b"\x00\x00\x80?\x00\x00\x00@",
            ),
        )
        self.assertEqual(
            payload[1],
            examples.ExampleRow(
                source_key="key2",
                source_text="ctx:\nid:Items\npl:Items",
                lang="de",
                msgstr="",
                msgstr_plural='{"0":"","1":"Dinge"}',
                review_status="reviewed",
                is_ai_generated=0,
                translation_hash="hash2",
                file_path="file.po",
                file_sha256="sha",
                embedding=b"\x00\x00\x80?\x00\x00\x00@",
            ),
        )


class TestEmbeddingPacking(unittest.TestCase):
    def test_pack_embedding_rejects_non_finite(self):
        with self.assertRaises(ValueError):
            examples._pack_embedding(
                [1.0, math.nan],
                embedding_dim=2,
                require_finite=True,
            )

    def test_pack_embedding_length_mismatch(self):
        with self.assertRaises(ValueError):
            examples._pack_embedding(
                [1.0],
                embedding_dim=2,
                require_finite=True,
            )

    def test_normalize_embedding_l2(self):
        normalized = examples._normalize_embedding([3.0, 4.0], "l2_normalize")
        self.assertAlmostEqual(normalized[0], 0.6, places=6)
        self.assertAlmostEqual(normalized[1], 0.8, places=6)


class TestQueryExamples(unittest.TestCase):
    def test_query_blob_length_mismatch(self):
        conn = sqlite3.connect(":memory:")
        db = examples.ExamplesDb(
            conn=conn,
            meta={},
            embedding_dim=2,
            embedding_distance="cosine",
            vector_encoding="float32_le",
            embedding_normalization="none",
            require_finite=True,
        )
        with self.assertRaises(ValueError):
            examples.query_examples(db, query_embedding=b"\x00", top_n=1)
        conn.close()

    def test_query_examples_raises_when_vector_missing(self):
        conn = mock.Mock()
        db = examples.ExamplesDb(
            conn=conn,
            meta={},
            embedding_dim=2,
            embedding_distance="cosine",
            vector_encoding="float32_le",
            embedding_normalization="none",
            require_finite=True,
        )
        conn.execute.side_effect = RuntimeError("no vector")
        with self.assertRaises(RuntimeError) as ctx:
            examples.query_examples(db, query_embedding=[0.1, 0.2], top_n=1)
        self.assertEqual(
            str(ctx.exception), "sqlite-vector unavailable for examples"
        )
        self.assertEqual(conn.execute.call_count, 1)
        sql = conn.execute.call_args[0][0]
        self.assertIn("vector_quantize_scan", sql)
        self.assertNotIn("vector_init", sql)


class TestExamplesDb(unittest.TestCase):
    def test_build_examples_db_raises_when_output_exists(self):
        config = _base_config()
        rows = [
            (
                "key1",
                "ctx:\nid:Hello\npl:",
                "",
                "de",
                "Hallo",
                "{}",
                "reviewed",
                0,
                "hash",
                "file.po",
            )
        ]

        def embedder(texts):
            return [[0.1, 0.2] for _ in texts]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "examples.sqlite"
            output_path.write_text("placeholder", encoding="ascii")
            with self.assertRaises(FileExistsError):
                examples._build_examples_db(
                    rows,
                    output_path=output_path,
                    scope="workspace",
                    source_snapshot_kind="workspace_tm",
                    source_snapshot_id=None,
                    lang="de",
                    config=config,
                    project_id="proj",
                    config_hash=config.config_hash,
                    embed_policy_hash="hash",
                    embedder=embedder,
                    sqlite_vector_path="/tmp/vector.so",
                )

    def test_open_examples_db_rejects_bad_vector_encoding(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "examples.sqlite"
            _write_examples_meta_db(db_path)
            meta = _example_meta()
            meta["vector_encoding"] = "float32"
            with (
                mock.patch("kdeai.examples.kdedb.validate_meta_table", return_value=meta),
                mock.patch("kdeai.examples.kdedb.enable_sqlite_vector"),
            ):
                with self.assertRaises(ValueError):
                    examples.open_examples_db(
                        db_path,
                        project_id="proj",
                        config_hash="config",
                        embed_policy_hash="embed-hash",
                        sqlite_vector_path="/tmp/vector.so",
                    )

    def test_open_examples_db_rejects_bad_examples_scope(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "examples.sqlite"
            _write_examples_meta_db(db_path)
            meta = _example_meta()
            meta["examples_scope"] = "foo"
            with (
                mock.patch("kdeai.examples.kdedb.validate_meta_table", return_value=meta),
                mock.patch("kdeai.examples.kdedb.enable_sqlite_vector"),
            ):
                with self.assertRaises(ValueError):
                    examples.open_examples_db(
                        db_path,
                        project_id="proj",
                        config_hash="config",
                        embed_policy_hash="embed-hash",
                        sqlite_vector_path="/tmp/vector.so",
                    )

    def test_open_examples_db_rejects_bad_source_snapshot_kind(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "examples.sqlite"
            _write_examples_meta_db(db_path)
            meta = _example_meta()
            meta["source_snapshot_kind"] = "bar"
            with (
                mock.patch("kdeai.examples.kdedb.validate_meta_table", return_value=meta),
                mock.patch("kdeai.examples.kdedb.enable_sqlite_vector"),
            ):
                with self.assertRaises(ValueError):
                    examples.open_examples_db(
                        db_path,
                        project_id="proj",
                        config_hash="config",
                        embed_policy_hash="embed-hash",
                        sqlite_vector_path="/tmp/vector.so",
                    )

    def test_open_examples_db_accepts_valid_meta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "examples.sqlite"
            _write_examples_meta_db(db_path)
            meta = _example_meta()
            with (
                mock.patch("kdeai.examples.kdedb.validate_meta_table", return_value=meta),
                mock.patch("kdeai.examples.kdedb.enable_sqlite_vector"),
            ):
                db = examples.open_examples_db(
                    db_path,
                    project_id="proj",
                    config_hash="config",
                    embed_policy_hash="embed-hash",
                    sqlite_vector_path="/tmp/vector.so",
                )
                db.conn.close()

    def test_open_examples_db_requires_reference_snapshot_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "examples.sqlite"
            _write_examples_meta_db(db_path)
            meta = _example_meta()
            meta["source_snapshot_kind"] = "reference_tm"
            meta["source_snapshot_id"] = ""
            with (
                mock.patch("kdeai.examples.kdedb.validate_meta_table", return_value=meta),
                mock.patch("kdeai.examples.kdedb.enable_sqlite_vector"),
            ):
                with self.assertRaises(ValueError):
                    examples.open_examples_db(
                        db_path,
                        project_id="proj",
                        config_hash="config",
                        embed_policy_hash="embed-hash",
                        sqlite_vector_path="/tmp/vector.so",
                    )

    def test_open_examples_db_enables_sqlite_vector(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "examples.sqlite"
            _write_examples_meta_db(db_path)
            with (
                mock.patch("kdeai.examples.kdedb.enable_sqlite_vector") as enable_vector,
            ):
                db = examples.open_examples_db(
                    db_path,
                    project_id="proj",
                    config_hash="config",
                    embed_policy_hash="embed-hash",
                    sqlite_vector_path="/tmp/vector.so",
                )
                try:
                    enable_vector.assert_called_once()
                finally:
                    db.conn.close()

    def test_open_examples_db_requires_sqlite_vector(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "examples.sqlite"
            _write_examples_meta_db(db_path)
            with self.assertRaises(RuntimeError):
                examples.open_examples_db(
                    db_path,
                    project_id="proj",
                    config_hash="config",
                    embed_policy_hash="embed-hash",
                    sqlite_vector_path=None,
                )

    def test_open_examples_db_raises_on_extension_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "examples.sqlite"
            _write_examples_meta_db(db_path)
            with mock.patch(
                "kdeai.examples.kdedb.enable_sqlite_vector",
                side_effect=RuntimeError("boom"),
            ):
                with self.assertRaises(RuntimeError):
                    examples.open_examples_db(
                        db_path,
                        project_id="proj",
                        config_hash="config",
                        embed_policy_hash="embed-hash",
                        sqlite_vector_path="/tmp/vector.so",
                    )

    def test_build_examples_db_enables_sqlite_vector(self):
        config = _base_config()
        rows = [
            (
                "key1",
                "ctx:\nid:Hello\npl:",
                "",
                "de",
                "Hallo",
                "{}",
                "reviewed",
                0,
                "hash",
                "file.po",
            )
        ]

        def embedder(texts):
            return [[0.1, 0.2] for _ in texts]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "examples.sqlite"
            with (
                mock.patch("kdeai.examples.kdedb.enable_sqlite_vector") as enable_vector,
                mock.patch("kdeai.examples._create_vector_index") as create_index,
            ):
                create_index.return_value = None
                examples._build_examples_db(
                    rows,
                    output_path=output_path,
                    scope="workspace",
                    source_snapshot_kind="workspace_tm",
                    source_snapshot_id=None,
                    lang="de",
                    config=config,
                    project_id="proj",
                    config_hash=config.config_hash,
                    embed_policy_hash="hash",
                    embedder=embedder,
                    sqlite_vector_path="/tmp/vector.so",
                )

                enable_vector.assert_called_once()
                _, kwargs = enable_vector.call_args
                self.assertEqual(kwargs.get("extension_path"), "/tmp/vector.so")

    def test_build_examples_db_rolls_back_on_vector_failure(self):
        config = _base_config()
        rows = [
            (
                "key1",
                "ctx:\nid:Hello\npl:",
                "",
                "de",
                "Hallo",
                "{}",
                "reviewed",
                0,
                "hash",
                "file.po",
            )
        ]

        def embedder(texts):
            return [[0.1, 0.2] for _ in texts]

        mock_conn = mock.Mock()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "examples.sqlite"
            with (
                mock.patch("kdeai.examples.kdedb.connect_writable", return_value=mock_conn),
                mock.patch("kdeai.examples.kdedb.enable_sqlite_vector"),
                mock.patch(
                    "kdeai.examples._create_vector_index",
                    side_effect=RuntimeError("boom"),
                ),
            ):
                with self.assertRaises(RuntimeError):
                    examples._build_examples_db(
                        rows,
                        output_path=output_path,
                        scope="workspace",
                        source_snapshot_kind="workspace_tm",
                        source_snapshot_id=None,
                        lang="de",
                        config=config,
                        project_id="proj",
                        config_hash=config.config_hash,
                        embed_policy_hash="hash",
                        embedder=embedder,
                        sqlite_vector_path="/tmp/vector.so",
                    )

        mock_conn.rollback.assert_called_once()
        mock_conn.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
