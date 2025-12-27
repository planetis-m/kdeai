import unittest

from conftest import build_config
from kdeai import examples


class TestExamplesBuildRows(unittest.TestCase):
    def test_workspace_row_uses_source_text_column(self):
        config = build_config()
        source_text = "ctx:x\nid:y\npl:z"
        row = (
            "key1",
            source_text,
            "",
            "de",
            "Hallo",
            "{}",
            "reviewed",
            0,
            "hash",
            "file.po",
        )
        captured: list[str] = []

        def embedder(texts):
            captured.extend(texts)
            return [[0.1, 0.2] for _ in texts]

        payload = examples._build_examples_rows(
            [row],
            lang="de",
            config=config,
            embedder=embedder,
            embedding_dim=2,
            embedding_normalization="none",
            require_finite=True,
            include_file_sha256=False,
        )

        self.assertEqual(len(payload), 1)
        self.assertEqual(captured, [source_text])
        self.assertEqual(payload[0].source_text, source_text)
        self.assertEqual(payload[0].source_key, "key1")


if __name__ == "__main__":
    unittest.main()
