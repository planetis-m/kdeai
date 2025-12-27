import tempfile
import unittest
from pathlib import Path

import polib

from conftest import build_config
from kdeai import plan
from kdeai import state as kdestate
from kdeai import po_utils
from kdeai.constants import PlanAction


class TestPlanSkipAction(unittest.TestCase):
    def test_non_empty_entry_plans_llm_with_base_hash(self):
        config = build_config()
        entry = polib.POEntry(msgctxt="menu", msgid="File", msgstr="Datei")
        marker_flags, comment_prefixes, _review_prefix, _ai_prefix, ai_flag = (
            po_utils.marker_settings_from_config(config)
        )
        marker_flags = po_utils.ensure_ai_flag_in_markers(marker_flags, ai_flag)
        expected_hash = kdestate.entry_state_hash(
            entry,
            lang="de",
            marker_flags=marker_flags,
            comment_prefixes=comment_prefixes,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = plan.PlanBuilder(
                project_root=Path(tmpdir),
                project_id="proj",
                config=config,
                lang="de",
                cache="off",
                examples_mode="off",
                glossary_mode="off",
            )
            try:
                draft = builder.build_draft("locale/de.po", [entry])
            finally:
                builder.close()

        self.assertEqual(len(draft["entries"]), 1)
        item = draft["entries"][0]
        self.assertEqual(item["action"], PlanAction.LLM)
        self.assertEqual(item["msgctxt"], "menu")
        self.assertEqual(item["msgid"], "File")
        self.assertEqual(item["msgid_plural"], "")
        self.assertEqual(item["base_state_hash"], expected_hash)


if __name__ == "__main__":
    unittest.main()
