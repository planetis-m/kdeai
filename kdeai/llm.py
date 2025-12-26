from __future__ import annotations

from typing import Iterable, Mapping, MutableMapping, TypedDict

import dspy

from kdeai import apply as kdeapply
from kdeai.config import Config
from kdeai.llm_provider import configure_dspy
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
    prompt: Mapping[str, object]
    translation: Mapping[str, object]
    flags: Mapping[str, object]
    comments: Mapping[str, object]


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


def _resolve_model_id(config: Config) -> str:
    data = config.data
    prompt_cfg = data.get("prompt") if isinstance(data, Mapping) else None
    if isinstance(prompt_cfg, Mapping):
        generation_model_id = prompt_cfg.get("generation_model_id")
        if isinstance(generation_model_id, str) and generation_model_id.strip():
            return generation_model_id
        embedding_policy = prompt_cfg.get("embedding_policy")
        if isinstance(embedding_policy, Mapping):
            model_id = embedding_policy.get("model_id")
            if isinstance(model_id, str) and model_id.strip():
                return model_id
        examples_cfg = prompt_cfg.get("examples")
        if isinstance(examples_cfg, Mapping):
            examples_policy = examples_cfg.get("embedding_policy")
            if isinstance(examples_policy, Mapping):
                model_id = examples_policy.get("model_id")
                if isinstance(model_id, str) and model_id.strip():
                    return model_id
    return "gpt-4o"


def _comment_prefix(config: Mapping[str, object], key: str) -> str:
    defaults = {
        "tool": "KDEAI:",
        "ai": "KDEAI-AI:",
        "tm": "KDEAI-TM:",
        "review": "KDEAI-REVIEW:",
    }
    markers = config.get("markers") if isinstance(config, Mapping) else None
    if isinstance(markers, Mapping):
        comment_prefixes = markers.get("comment_prefixes")
        if isinstance(comment_prefixes, Mapping):
            value = comment_prefixes.get(key)
            if value:
                return str(value)
    return defaults.get(key, "KDEAI-AI:")


def _ai_flag(config: Mapping[str, object]) -> str:
    markers = config.get("markers") if isinstance(config, Mapping) else None
    if isinstance(markers, Mapping):
        ai_flag = markers.get("ai_flag")
        if ai_flag:
            return str(ai_flag)
    return kdeapply.DEFAULT_AI_FLAG


def _llm_tagging_settings(config: Mapping[str, object]) -> tuple[list[str], bool, str, str]:
    apply_cfg = config.get("apply") if isinstance(config, Mapping) else None
    tagging = apply_cfg.get("tagging") if isinstance(apply_cfg, Mapping) else None
    llm_cfg = tagging.get("llm") if isinstance(tagging, Mapping) else None

    if isinstance(llm_cfg, Mapping):
        add_flags_raw = llm_cfg.get("add_flags", ["fuzzy"])
        add_flags = (
            [str(flag) for flag in add_flags_raw]
            if isinstance(add_flags_raw, Iterable) and not isinstance(add_flags_raw, (str, bytes))
            else ["fuzzy"]
        )
        add_ai_flag = bool(llm_cfg.get("add_ai_flag", True))
        comment_prefix_key = str(llm_cfg.get("comment_prefix_key") or "ai")
    else:
        add_flags = ["fuzzy"]
        add_ai_flag = True
        comment_prefix_key = "ai"

    prefix = _comment_prefix(config, comment_prefix_key)
    ai_flag = _ai_flag(config)
    return add_flags, add_ai_flag, ai_flag, prefix


def _merge_unique(values: Iterable[str], extra: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for value in list(values) + list(extra):
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        merged.append(text)
    return merged


def _apply_llm_tagging(
    entry: MutableMapping[str, object],
    *,
    config: Mapping[str, object],
    model_id: str,
) -> None:
    add_flags, add_ai_flag, ai_flag, comment_prefix = _llm_tagging_settings(config)
    ensured_line = f"{comment_prefix} model={model_id}"

    flags = entry.get("flags")
    if isinstance(flags, Mapping):
        add_list = list(flags.get("add", []))
        remove_list = list(flags.get("remove", []))
    else:
        add_list = []
        remove_list = []
    extra_flags = list(add_flags)
    if add_ai_flag:
        extra_flags.append(ai_flag)
    entry["flags"] = {
        "add": _merge_unique(add_list, extra_flags),
        "remove": remove_list,
    }

    comments = entry.get("comments")
    if isinstance(comments, Mapping):
        remove_prefixes = list(comments.get("remove_prefixes", []))
        ensure_lines = list(comments.get("ensure_lines", []))
        append = str(comments.get("append", ""))
    else:
        remove_prefixes = []
        ensure_lines = []
        append = ""

    entry["comments"] = {
        "remove_prefixes": _merge_unique(remove_prefixes, [comment_prefix]),
        "ensure_lines": _merge_unique(ensure_lines, [ensured_line]),
        "append": append,
    }


def _prompt_for_entry(
    entry: Mapping[str, object],
    prompt_data: Mapping[str, object] | None,
    *,
    target_lang: str,
) -> dict[str, object]:
    prompt_payload: dict[str, object] = {}
    if isinstance(prompt_data, Mapping):
        for key in (
            "source_context",
            "source_text",
            "plural_text",
            "target_lang",
            "glossary_context",
            "few_shot_examples",
        ):
            if key in prompt_data:
                prompt_payload[key] = prompt_data.get(key)

    prompt_payload.setdefault("source_context", entry.get("msgctxt", ""))
    prompt_payload.setdefault("source_text", entry.get("msgid", ""))
    prompt_payload.setdefault("plural_text", entry.get("msgid_plural", ""))
    prompt_payload.setdefault("target_lang", target_lang)
    prompt_payload.setdefault("glossary_context", "")
    prompt_payload.setdefault("few_shot_examples", "")
    return prompt_payload


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


def batch_translate_plan(plan: Plan, config: Config) -> Plan:
    if dspy.settings.lm is None:
        configure_dspy(config)
    translator = KDEAITranslator()
    model_id = _resolve_model_id(config)
    target_lang = str(plan.get("lang", ""))

    files = plan.get("files")
    if not isinstance(files, Iterable):
        return plan

    for file_item in files:
        if not isinstance(file_item, MutableMapping):
            continue
        entries = file_item.get("entries")
        if not isinstance(entries, Iterable):
            continue
        for entry in entries:
            if not isinstance(entry, MutableMapping):
                continue
            if str(entry.get("action", "")) != "llm":
                continue
            prompt_raw = entry.get("prompt")
            prompt_payload = _prompt_for_entry(
                entry,
                prompt_raw if isinstance(prompt_raw, Mapping) else None,
                target_lang=target_lang,
            )
            prediction = translator(prompt_payload)
            translated_text = str(getattr(prediction, "translated_text", ""))
            translated_plural = str(getattr(prediction, "translated_plural", ""))
            msgid_plural = str(entry.get("msgid_plural", ""))

            entry["translation"] = _translation_payload(
                msgid_plural=msgid_plural,
                translated_text=translated_text,
                translated_plural=translated_plural,
            )
            _apply_llm_tagging(entry, config=config.data, model_id=model_id)

    return plan
