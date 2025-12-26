import hashlib
import unittest

from kdeai import po_model


class TestPoModelSourceKey(unittest.TestCase):
    def test_source_key_spec(self):
        msgctxt = "menu"
        msgid = "File"
        msgid_plural = "Files"
        expected = hashlib.sha256(
            (msgctxt + "\u0004" + msgid + "\u0000" + msgid_plural).encode("utf-8")
        ).hexdigest()
        self.assertEqual(po_model.source_key_for(msgctxt, msgid, msgid_plural), expected)

    def test_source_key_empty_context_plural(self):
        msgctxt = ""
        msgid = "Hello"
        msgid_plural = ""
        expected = hashlib.sha256(("\u0004Hello\u0000").encode("utf-8")).hexdigest()
        self.assertEqual(po_model.source_key_for(msgctxt, msgid, msgid_plural), expected)

    def test_source_key_none_context_plural(self):
        msgctxt = None
        msgid = "Hello"
        msgid_plural = None
        expected = hashlib.sha256(("\u0004Hello\u0000").encode("utf-8")).hexdigest()
        self.assertEqual(po_model.source_key_for(msgctxt, msgid, msgid_plural), expected)

    def test_source_key_empty_context_with_plural(self):
        msgctxt = ""
        msgid = "One file"
        msgid_plural = "Many files"
        expected = hashlib.sha256(
            ("\u0004One file\u0000Many files").encode("utf-8")
        ).hexdigest()
        self.assertEqual(po_model.source_key_for(msgctxt, msgid, msgid_plural), expected)

    def test_source_key_none_context_with_plural(self):
        msgctxt = None
        msgid = "One file"
        msgid_plural = "Many files"
        expected = hashlib.sha256(
            ("\u0004One file\u0000Many files").encode("utf-8")
        ).hexdigest()
        self.assertEqual(po_model.source_key_for(msgctxt, msgid, msgid_plural), expected)

    def test_source_text_v1_spec(self):
        msgctxt = "menu"
        msgid = "File"
        msgid_plural = "Files"
        expected = "ctx:menu\nid:File\npl:Files"
        self.assertEqual(po_model.source_text_v1(msgctxt, msgid, msgid_plural), expected)


class TestPoModelParsing(unittest.TestCase):
    def test_parse_po_bytes_source_key_spec(self):
        po_text = (
            'msgid ""\n'
            'msgstr ""\n'
            '"Content-Type: text/plain; charset=UTF-8\\n"\n'
            "\n"
            'msgctxt "menu"\n'
            'msgid "File"\n'
            'msgid_plural "Files"\n'
            'msgstr[0] "Datei"\n'
            'msgstr[1] "Dateien"\n'
        )
        units = po_model.parse_po_bytes(po_text.encode("utf-8"))
        self.assertEqual(len(units), 1)
        expected = hashlib.sha256(
            ("menu\u0004File\u0000Files").encode("utf-8")
        ).hexdigest()
        self.assertEqual(units[0].source_key, expected)

if __name__ == "__main__":
    unittest.main()
