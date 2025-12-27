import unittest

from conftest import build_config
from kdeai import examples
from kdeai import po_model


class TestExamplesBuildRows(unittest.TestCase):
    def test_workspace_row_uses_msgctxt_column_even_if_looks_canonical(self):
        config = build_config()
        msgctxt = "ctx:x\nid:y\npl:z"
        msgid = "Hello"
        msgid_plural = ""
        row = (
            "key1",
            msgctxt,
            msgid,
            msgid_plural,
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
        expected_source_text = po_model.source_text_v1(msgctxt, msgid, msgid_plural)
        self.assertEqual(captured, [expected_source_text])
        self.assertEqual(payload[0].source_text, expected_source_text)
        self.assertEqual(payload[0].source_key, "key1")


if __name__ == "__main__":
    unittest.main()
