import types
import unittest

from kdeai import embed_client


class TestEmbedClientHelpers(unittest.TestCase):
    def test_embedding_response_data_accepts_dict_and_object(self):
        self.assertEqual(embed_client._embedding_response_data({"data": [1]}), [1])
        response_obj = types.SimpleNamespace(data=[{"embedding": [1.0, 2.0]}])
        self.assertEqual(
            embed_client._embedding_response_data(response_obj),
            [{"embedding": [1.0, 2.0]}],
        )

    def test_extract_embedding_accepts_dict_and_object(self):
        self.assertEqual(
            embed_client._extract_embedding({"embedding": [1, 2]}),
            [1.0, 2.0],
        )
        item_obj = types.SimpleNamespace(embedding=[3.0, 4.5])
        self.assertEqual(embed_client._extract_embedding(item_obj), [3.0, 4.5])
