import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import dspy
import polib

from conftest import build_config
from kdeai import plan as kdeplan
from kdeai import retrieve_examples
from kdeai import apply as kdeapply


class TestBuildAssetsExamples(unittest.TestCase):
    def test_auto_examples_skips_open_without_embedder(self):
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("kdeai.plan.retrieve_examples.open_examples_best_effort") as open_mock:
                assets, effective_examples_mode, _ = kdeplan._build_assets(
                    project_root=Path(tmpdir),
                    project_id="proj",
                    config=config,
                    lang="de",
                    cache="on",
                    examples_mode="auto",
                    glossary_mode="off",
                    embedder=None,
                    sqlite_vector_path=None,
                )
        open_mock.assert_not_called()
        self.assertEqual(effective_examples_mode, "auto")
        self.assertIsNone(assets.examples_db)


class TestCollectExamples(unittest.TestCase):
    def test_embedder_exception_auto_returns_empty(self):
        def embedder(_texts):
            raise RuntimeError("boom")

        result = retrieve_examples.collect_examples(
            examples_db=mock.Mock(),
            embedder=embedder,
            source_text="Hello",
            top_n=1,
            lang="de",
            eligibility=mock.Mock(),
            review_status_order=["reviewed"],
            required=False,
        )
        self.assertEqual(result, [])

    def test_embedder_exception_required_raises(self):
        def embedder(_texts):
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            retrieve_examples.collect_examples(
                examples_db=mock.Mock(),
                embedder=embedder,
                source_text="Hello",
                top_n=1,
                lang="de",
                eligibility=mock.Mock(),
                review_status_order=["reviewed"],
                required=True,
            )

    def test_embedder_empty_auto_returns_empty(self):
        def embedder(_texts):
            return []

        result = retrieve_examples.collect_examples(
            examples_db=mock.Mock(),
            embedder=embedder,
            source_text="Hello",
            top_n=1,
            lang="de",
            eligibility=mock.Mock(),
            review_status_order=["reviewed"],
            required=False,
        )
        self.assertEqual(result, [])

    def test_embedder_empty_required_raises(self):
        def embedder(_texts):
            return []

        with self.assertRaises(RuntimeError):
            retrieve_examples.collect_examples(
                examples_db=mock.Mock(),
                embedder=embedder,
                source_text="Hello",
                top_n=1,
                lang="de",
                eligibility=mock.Mock(),
                review_status_order=["reviewed"],
                required=True,
            )

    def test_query_exception_auto_returns_empty(self):
        def embedder(_texts):
            return [[0.1, 0.2]]

        with mock.patch("kdeai.retrieve_examples.kdeexamples.query_examples") as query_mock:
            query_mock.side_effect = RuntimeError("boom")
            result = retrieve_examples.collect_examples(
                examples_db=mock.Mock(),
                embedder=embedder,
                source_text="Hello",
                top_n=1,
                lang="de",
                eligibility=mock.Mock(),
                review_status_order=["reviewed"],
                required=False,
            )
        self.assertEqual(result, [])

    def test_query_exception_required_raises(self):
        def embedder(_texts):
            return [[0.1, 0.2]]

        with mock.patch("kdeai.retrieve_examples.kdeexamples.query_examples") as query_mock:
            query_mock.side_effect = RuntimeError("boom")
            with self.assertRaises(RuntimeError):
                retrieve_examples.collect_examples(
                    examples_db=mock.Mock(),
                    embedder=embedder,
                    source_text="Hello",
                    top_n=1,
                    lang="de",
                    eligibility=mock.Mock(),
                    review_status_order=["reviewed"],
                    required=True,
                )


class TestOpenExamplesDb(unittest.TestCase):
    def test_open_examples_db_auto_returns_none_on_failure(self):
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pointer_dir = root / ".kdeai" / "cache" / "examples" / "workspace"
            pointer_dir.mkdir(parents=True)
            pointer_path = pointer_dir / "examples.workspace.de.current.json"
            pointer_path.write_text('{"db_file":"examples.sqlite"}', encoding="utf-8")
            with mock.patch(
                "kdeai.retrieve_examples.kdeexamples.open_examples_db",
                side_effect=RuntimeError("boom"),
            ):
                result = retrieve_examples.open_examples_best_effort(
                    root,
                    scope="workspace",
                    lang="de",
                    project_id="proj",
                    config_hash=config.config_hash,
                    embed_policy_hash=config.embed_policy_hash,
                    sqlite_vector_path="vector.so",
                    required=False,
                )
        self.assertIsNone(result)

    def test_open_examples_db_required_raises_on_failure(self):
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pointer_dir = root / ".kdeai" / "cache" / "examples" / "workspace"
            pointer_dir.mkdir(parents=True)
            pointer_path = pointer_dir / "examples.workspace.de.current.json"
            pointer_path.write_text('{"db_file":"examples.sqlite"}', encoding="utf-8")
            (pointer_dir / "examples.sqlite").write_text("", encoding="utf-8")
            with mock.patch(
                "kdeai.retrieve_examples.kdeexamples.open_examples_db",
                side_effect=RuntimeError("boom"),
            ):
                with self.assertRaises(RuntimeError):
                    retrieve_examples.open_examples_best_effort(
                        root,
                        scope="workspace",
                        lang="de",
                        project_id="proj",
                        config_hash=config.config_hash,
                        embed_policy_hash=config.embed_policy_hash,
                        sqlite_vector_path="vector.so",
                        required=True,
                    )


class TestPlanBuilderBuildDraft(unittest.TestCase):
    def _entry(self):
        return polib.POEntry(msgid="Hello", msgstr="")

    def test_build_draft_auto_examples_without_sqlite_vector(self):
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = kdeplan.PlanBuilder(
                project_root=Path(tmpdir),
                project_id="proj",
                config=config,
                lang="de",
                cache="on",
                examples_mode="auto",
                glossary_mode="off",
                embedder=None,
                sqlite_vector_path=None,
            )
            try:
                plan = builder.build_draft("file.po", [self._entry()])
            finally:
                builder.close()
        entries = plan["entries"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["action"], "llm")
        self.assertEqual(entries[0]["examples"], "")

    def test_build_draft_auto_examples_with_embedder_without_sqlite_vector(self):
        config = build_config()

        def embedder(_texts):
            return [[0.0]]

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = kdeplan.PlanBuilder(
                project_root=Path(tmpdir),
                project_id="proj",
                config=config,
                lang="de",
                cache="on",
                examples_mode="auto",
                glossary_mode="off",
                embedder=embedder,
                sqlite_vector_path=None,
            )
            try:
                plan = builder.build_draft("file.po", [self._entry()])
            finally:
                builder.close()
        entries = plan["entries"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["action"], "llm")
        self.assertEqual(entries[0]["examples"], "")

    def test_build_draft_required_examples_without_sqlite_vector_fails(self):
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                kdeplan.PlanBuilder(
                    project_root=Path(tmpdir),
                    project_id="proj",
                    config=config,
                    lang="de",
                    cache="on",
                    examples_mode="required",
                    glossary_mode="off",
                    embedder=lambda texts: [[0.0] for _ in texts],
                    sqlite_vector_path=None,
                )

    def test_build_draft_required_examples_query_failure(self):
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch(
                "kdeai.plan.retrieve_examples.open_examples_best_effort",
                return_value=mock.Mock(),
            ):
                with mock.patch(
                    "kdeai.retrieve_examples.kdeexamples.query_examples",
                    side_effect=RuntimeError("boom"),
                ):
                    builder = kdeplan.PlanBuilder(
                        project_root=Path(tmpdir),
                        project_id="proj",
                        config=config,
                        lang="de",
                        cache="on",
                        examples_mode="required",
                        glossary_mode="off",
                        embedder=lambda texts: [[0.0] for _ in texts],
                        sqlite_vector_path="vector.so",
                    )
                    try:
                        with self.assertRaises(RuntimeError):
                            builder.build_draft("file.po", [self._entry()])
                    finally:
                        builder.close()

    def test_required_examples_raises_without_embedder(self):
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                kdeplan._build_assets(
                    project_root=Path(tmpdir),
                    project_id="proj",
                    config=config,
                    lang="de",
                    cache="on",
                    examples_mode="required",
                    glossary_mode="off",
                    embedder=None,
                    sqlite_vector_path="vector.so",
                )

    def test_required_examples_raises_without_sqlite_vector(self):
        config = build_config()

        def embedder(texts):
            return [[0.0] for _ in texts]

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                kdeplan._build_assets(
                    project_root=Path(tmpdir),
                    project_id="proj",
                    config=config,
                    lang="de",
                    cache="on",
                    examples_mode="required",
                    glossary_mode="off",
                    embedder=embedder,
                    sqlite_vector_path=None,
                )

    def test_auto_examples_ignores_open_failures(self):
        config = build_config()

        def embedder(texts):
            return [[0.0] for _ in texts]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pointer_dir = root / ".kdeai" / "cache" / "examples" / "workspace"
            pointer_dir.mkdir(parents=True)
            pointer_path = pointer_dir / "examples.workspace.de.current.json"
            pointer_path.write_text('{"db_file":"examples.sqlite"}', encoding="utf-8")
            (pointer_dir / "examples.sqlite").write_text("", encoding="utf-8")
            with mock.patch(
                "kdeai.plan.kdeexamples.open_examples_db",
                side_effect=RuntimeError("boom"),
            ) as open_mock:
                assets, effective_examples_mode, _ = kdeplan._build_assets(
                    project_root=root,
                    project_id="proj",
                    config=config,
                    lang="de",
                    cache="on",
                    examples_mode="auto",
                    glossary_mode="off",
                    embedder=embedder,
                    sqlite_vector_path="vector.so",
                )
        self.assertEqual(open_mock.call_count, 1)
        self.assertEqual(effective_examples_mode, "auto")
        self.assertIsNone(assets.examples_db)


class TestGeneratePlanWithRunLlm(unittest.TestCase):
    def test_generate_plan_for_file_run_llm_applyable(self):
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            locale_dir = root / "locale"
            locale_dir.mkdir(parents=True, exist_ok=True)
            po_path = locale_dir / "de.po"
            po_text = (
                'msgid ""\n'
                'msgstr ""\n'
                '"Content-Type: text/plain; charset=UTF-8\\n"\n'
                '"Plural-Forms: nplurals=2; plural=(n != 1);\\n"\n'
                "\n"
                'msgctxt "menu"\n'
                'msgid "File"\n'
                'msgstr ""\n'
            )
            po_path.write_text(po_text, encoding="utf-8")

            builder = kdeplan.PlanBuilder(
                project_root=root,
                project_id="proj",
                config=config,
                lang="de",
                cache="off",
                examples_mode="off",
                glossary_mode="off",
                embedder=None,
                sqlite_vector_path=None,
            )
            from kdeai import llm as kdellm
            original_forward = kdellm.KDEAITranslator.forward
            original_lm = dspy.settings.lm
            try:
                def _fake_forward(self, _prompt):
                    return types.SimpleNamespace(translated_text="Datei", translated_plural="")

                dspy.settings.lm = object()
                kdellm.KDEAITranslator.forward = _fake_forward
                file_draft = kdeplan.generate_plan_for_file(
                    project_root=root,
                    project_id="proj",
                    path=po_path,
                    path_casefold=False,
                    builder=builder,
                    config=config,
                    run_llm=True,
                )
            finally:
                kdellm.KDEAITranslator.forward = original_forward
                dspy.settings.lm = original_lm
                builder.close()

            entries = file_draft["entries"]
            self.assertEqual(len(entries), 1)
            entry = entries[0]
            self.assertEqual(entry["action"], "llm")
            self.assertTrue(entry["translation"]["msgstr"])
            self.assertEqual(entry["tag_profile"], "llm")

            plan_payload = {
                "format": 1,
                "project_id": "proj",
                "config_hash": config.config_hash,
                "lang": "de",
                "apply_defaults": {
                    "mode": "strict",
                    "overwrite": "conservative",
                    "post_index": "off",
                },
                "files": [file_draft],
            }
            finalized_plan = kdeplan.finalize_plan(plan_payload)
            before = po_path.read_text(encoding="utf-8")
            result = kdeapply.apply_plan(
                finalized_plan,
                project_root=root,
                project_id="proj",
                path_casefold=False,
                config=config,
                apply_mode="strict",
                overwrite="conservative",
            )
            after = po_path.read_text(encoding="utf-8")

            self.assertEqual(result.files_written, ["locale/de.po"])
            self.assertNotEqual(before, after)
            updated = polib.pofile(str(po_path))
            updated_entry = updated.find("File", msgctxt="menu")
            self.assertEqual(updated_entry.msgstr, "Datei")
