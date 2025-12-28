import unittest

from kdeai import validate


class TestValidators(unittest.TestCase):
    def test_non_empty_singular(self):
        errors = validate.validate_entry(
            validate.ValidationRequest(
                msgid="Hello",
                msgid_plural="",
                msgstr="",
                msgstr_plural={},
                plural_forms=None,
                placeholder_patterns=[],
            )
        )
        self.assertIn("non-empty translation", errors)

        errors = validate.validate_entry(
            validate.ValidationRequest(
                msgid="Hello",
                msgid_plural="",
                msgstr="Hallo",
                msgstr_plural={},
                plural_forms=None,
                placeholder_patterns=[],
            )
        )
        self.assertNotIn("non-empty translation", errors)

    def test_non_empty_plural_any_form(self):
        errors = validate.validate_entry(
            validate.ValidationRequest(
                msgid="File",
                msgid_plural="Files",
                msgstr="",
                msgstr_plural={"0": "", "1": ""},
                plural_forms="nplurals=2; plural=(n != 1);",
                placeholder_patterns=[],
            )
        )
        self.assertIn("non-empty translation", errors)

        errors = validate.validate_entry(
            validate.ValidationRequest(
                msgid="File",
                msgid_plural="Files",
                msgstr="",
                msgstr_plural={"0": "", "1": "Dateien"},
                plural_forms="nplurals=2; plural=(n != 1);",
                placeholder_patterns=[],
            )
        )
        self.assertNotIn("non-empty translation", errors)

    def test_plural_consistency_matches_header(self):
        errors = validate.validate_entry(
            validate.ValidationRequest(
                msgid="File",
                msgid_plural="Files",
                msgstr="",
                msgstr_plural={"0": "Datei"},
                plural_forms="nplurals=2; plural=(n != 1);",
                placeholder_patterns=[],
            )
        )
        self.assertIn("plural key consistency", errors)

        errors = validate.validate_entry(
            validate.ValidationRequest(
                msgid="File",
                msgid_plural="Files",
                msgstr="",
                msgstr_plural={"0": "Datei", "1": "Dateien"},
                plural_forms="nplurals=2; plural=(n != 1);",
                placeholder_patterns=[],
            )
        )
        self.assertNotIn("plural key consistency", errors)

    def test_plural_consistency_allows_missing_header(self):
        errors = validate.validate_entry(
            validate.ValidationRequest(
                msgid="File",
                msgid_plural="Files",
                msgstr="",
                msgstr_plural={"0": "Datei"},
                plural_forms=None,
                placeholder_patterns=[],
            )
        )
        self.assertNotIn("plural key consistency", errors)

    def test_tag_integrity_singular(self):
        errors = validate.validate_entry(
            validate.ValidationRequest(
                msgid="Hello %s",
                msgid_plural="",
                msgstr="Hallo %s",
                msgstr_plural={},
                plural_forms=None,
                placeholder_patterns=[r"%\w"],
            )
        )
        self.assertNotIn("tag/placeholder integrity", errors)

        errors = validate.validate_entry(
            validate.ValidationRequest(
                msgid="Hello %s",
                msgid_plural="",
                msgstr="Hallo",
                msgstr_plural={},
                plural_forms=None,
                placeholder_patterns=[r"%\w"],
            )
        )
        self.assertIn("tag/placeholder integrity", errors)

    def test_tag_integrity_plural_uses_index(self):
        errors = validate.validate_entry(
            validate.ValidationRequest(
                msgid="One %s",
                msgid_plural="%s items",
                msgstr="",
                msgstr_plural={"0": "Ein %s", "1": "%s Elemente"},
                plural_forms="nplurals=2; plural=(n != 1);",
                placeholder_patterns=[r"%\w"],
            )
        )
        self.assertNotIn("tag/placeholder integrity", errors)

        errors = validate.validate_entry(
            validate.ValidationRequest(
                msgid="One %s",
                msgid_plural="%s items",
                msgstr="",
                msgstr_plural={"0": "Ein %s", "1": "Elemente"},
                plural_forms="nplurals=2; plural=(n != 1);",
                placeholder_patterns=[r"%\w"],
            )
        )
        self.assertIn("tag/placeholder integrity", errors)

    def test_tag_integrity_allows_reordered_placeholders(self):
        errors = validate.validate_entry(
            validate.ValidationRequest(
                msgid="Value %1 %2",
                msgid_plural="",
                msgstr="Wert %2 %1",
                msgstr_plural={},
                plural_forms=None,
                placeholder_patterns=[r"%\d"],
            )
        )
        self.assertNotIn("tag/placeholder integrity", errors)

        errors = validate.validate_entry(
            validate.ValidationRequest(
                msgid="Value %1 %2",
                msgid_plural="",
                msgstr="Wert %1",
                msgstr_plural={},
                plural_forms=None,
                placeholder_patterns=[r"%\d"],
            )
        )
        self.assertIn("tag/placeholder integrity", errors)


if __name__ == "__main__":
    unittest.main()
