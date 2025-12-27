from __future__ import annotations

from typing import Iterable, Mapping, MutableMapping, Sequence, TypedDict

import dspy

from kdeai.config import Config
from kdeai.constants import PlanAction
from kdeai import prompt as kdeprompt
from kdeai.prompt import PromptData


class TranslationSignature(dspy.Signature):
    source_context: str = dspy.InputField(desc="msgctxt")
    source_text: str = dspy.InputField(desc="msgid")
    plural_text: str = dspy.InputField(desc="msgid_plural")
    target_lang: str = dspy.InputField(desc="Target language code")
    glossary_context: str = dspy.InputField(desc="Comma-separated glossary terms")
    few_shot_examples: str = dspy.InputField(desc="Retrieved TM examples")
    translated_text: str = dspy.OutputField(desc="Translated text")
    translated_plural: str = dspy.OutputField(desc="Translated plural text")


class KDEAITranslator(dspy.Module):
    def __init__(self, *, use_cot: bool = False) -> None:
        super().__init__()
        if use_cot:
            self._predictor = dspy.ChainOfThought(TranslationSignature)
        else:
            self._predictor = dspy.Predict(TranslationSignature)

    def forward(self, prompt: PromptData) -> dspy.Prediction:
        fields = _normalize_prompt_data(prompt)
        return self._predictor(**fields)


class PlanEntry(TypedDict, total=False):
    msgctxt: str
    msgid: str
    msgid_plural: str
    action: str
    examples: Sequence[Mapping[str, object]]
    glossary_terms: Sequence[Mapping[str, object]]
    translation: Mapping[str, object]
    tag_profile: str


Plan = dict[str, object]


def _normalize_prompt_data(prompt: Mapping[str, object]) -> dict[str, str]:
    def _text(value: object) -> str:
        return "" if value is None else str(value)

    return {
        "source_context": _text(prompt.get("source_context")),
        "source_text": _text(prompt.get("source_text")),
        "plural_text": _text(prompt.get("plural_text")),
        "target_lang": _text(prompt.get("target_lang")),
        "glossary_context": _text(prompt.get("glossary_context")),
        "few_shot_examples": _text(prompt.get("few_shot_examples")),
    }


def build_prompt_payload(
    entry: Mapping[str, object],
    *,
    target_lang: str,
) -> dict[str, object]:
    def _text(value: object) -> str:
        return "" if value is None else str(value)

    def _examples_context(value: object) -> str:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            examples = [item for item in value if isinstance(item, Mapping)]
            return kdeprompt.examples_context_from_dicts(examples)
        return _text(value)

    def _glossary_context(value: object) -> str:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            matches = [item for item in value if isinstance(item, Mapping)]
            return kdeprompt.glossary_context_from_dicts(matches)
        return _text(value)

    return {
        "source_context": _text(entry.get("msgctxt", "")),
        "source_text": _text(entry.get("msgid", "")),
        "plural_text": _text(entry.get("msgid_plural", "")),
        "target_lang": _text(target_lang),
        "glossary_context": _glossary_context(entry.get("glossary_terms", "")),
        "few_shot_examples": _examples_context(entry.get("examples", "")),
    }


def _translation_payload(
    *,
    msgid_plural: str,
    translated_text: str,
    translated_plural: str,
) -> dict[str, object]:
    if msgid_plural:
        plural_value = translated_plural.strip() or translated_text
        return {
            "msgstr": "",
            "msgstr_plural": {"0": translated_text, "1": plural_value},
        }
    return {"msgstr": translated_text, "msgstr_plural": {}}


def _has_translation_payload(translation: object) -> bool:
    if not isinstance(translation, Mapping):
        return False
    msgstr = str(translation.get("msgstr", "")).strip()
    msgstr_plural = translation.get("msgstr_plural")
    if msgstr:
        return True
    if isinstance(msgstr_plural, Mapping):
        return any(str(value).strip() for value in msgstr_plural.values())
    return False


def batch_translate(
    entries: Iterable[MutableMapping[str, object]],
    config: Config,
    *,
    target_lang: str,
) -> list[MutableMapping[str, object]]:
    if dspy.settings.lm is None:
        raise RuntimeError("DSPy not configured. Call configure_dspy() before batch_translate().")
    translator = KDEAITranslator()
    normalized_lang = str(target_lang or "")

    updated: list[MutableMapping[str, object]] = []
    for entry in entries:
        if not isinstance(entry, MutableMapping):
            continue
        if str(entry.get("action", "")) != PlanAction.LLM:
            continue
        if _has_translation_payload(entry.get("translation")):
            continue
        prompt_payload = build_prompt_payload(entry, target_lang=normalized_lang)
        prediction = translator(prompt_payload)
        translated_text = str(getattr(prediction, "translated_text", ""))
        translated_plural = str(getattr(prediction, "translated_plural", ""))
        msgid_plural = str(entry.get("msgid_plural", ""))

        entry["translation"] = _translation_payload(
            msgid_plural=msgid_plural,
            translated_text=translated_text,
            translated_plural=translated_plural,
        )
        entry.setdefault("action", PlanAction.LLM)
        entry.setdefault("tag_profile", "llm")
        updated.append(entry)

    return updated
