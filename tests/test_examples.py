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


class TestExamplesDb(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
