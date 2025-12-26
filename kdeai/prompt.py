from __future__ import annotations

from typing import Mapping, Sequence, TYPE_CHECKING, TypedDict

from kdeai import hash as kdehash
from kdeai.config import Config

class PromptData(TypedDict):
    source_context: str
    source_text: str
    plural_text: str
    target_lang: str
    glossary_context: str
    few_shot_examples: str
    messages: list[dict[str, str]]


if TYPE_CHECKING:
    from kdeai.examples import ExampleMatch
    from kdeai.glossary import GlossaryMatch
else:  # pragma: no cover - avoid heavy imports at runtime
    ExampleMatch = object  # type: ignore
    GlossaryMatch = object  # type: ignore


def _normalize_field(value: str | None) -> str:
    return value if value is not None else ""


def source_text_v1(msgctxt: str | None, msgid: str, msgid_plural: str | None) -> str:
    """Canonical source text used for prompts and embeddings."""
    norm_ctxt = _normalize_field(msgctxt)
    norm_plural = _normalize_field(msgid_plural)
    return f"ctx:{norm_ctxt}\nid:{msgid}\npl:{norm_plural}"


def _example_translation(example: ExampleMatch) -> str:
    msgstr = str(getattr(example, "msgstr", ""))
    if msgstr.strip():
        return msgstr
    msgstr_plural = getattr(example, "msgstr_plural", {})
    if isinstance(msgstr_plural, Mapping):
        normalized = {str(k): str(v) for k, v in msgstr_plural.items()}
        if any(value.strip() for value in normalized.values()):
            return kdehash.canonical_json(normalized)
    return ""


def _format_examples(examples: Sequence[ExampleMatch]) -> str:
    blocks: list[str] = []
    for idx, example in enumerate(examples, start=1):
        source_text = str(getattr(example, "source_text", ""))
        translation = _example_translation(example)
        blocks.append(
            "\n".join(
                [
                    f"{idx}. Source:",
                    source_text,
                    "Translation:",
                    translation,
                ]
            )
        )
    return "\n\n".join(blocks)


def _format_glossary(glossary: Sequence[GlossaryMatch]) -> str:
    lines: list[str] = []
    for match in glossary:
        term = getattr(match, "term", None)
        if term is None:
            continue
        src_surface = str(getattr(term, "src_surface", ""))
        tgt_primary = str(getattr(term, "tgt_primary", ""))
        alternates = getattr(term, "tgt_alternates", [])
        line = f"- {src_surface} -> {tgt_primary}"
        if isinstance(alternates, Sequence) and not isinstance(alternates, (str, bytes)):
            alt_values = [str(value) for value in alternates if str(value).strip()]
            if alt_values:
                line += f" (alternates: {', '.join(alt_values)})"
        lines.append(line)
    return "\n".join(lines)


def _glossary_context(glossary: Sequence[GlossaryMatch]) -> str:
    terms: list[str] = []
    for match in glossary:
        term = getattr(match, "term", None)
        if term is None:
            continue
        src_surface = str(getattr(term, "src_surface", ""))
        tgt_primary = str(getattr(term, "tgt_primary", ""))
        if not src_surface and not tgt_primary:
            continue
        if src_surface and tgt_primary:
            terms.append(f"{src_surface} -> {tgt_primary}")
        elif src_surface:
            terms.append(src_surface)
        else:
            terms.append(tgt_primary)
    return ", ".join(terms)


def examples_context(examples: Sequence[ExampleMatch]) -> str:
    return _format_examples(examples)


def glossary_context(glossary: Sequence[GlossaryMatch]) -> str:
    return _glossary_context(glossary)


def _system_prompt(source_lang: str, target_lang: str) -> str:
    return "\n".join(
        [
            "You are a KDEAI translation assistant.",
            f"Translate from {source_lang} to {target_lang}.",
            "Use examples and glossary terms as guidance only.",
            "Return only the translation text.",
        ]
    )


def _user_prompt(
    source_text: str,
    *,
    examples: Sequence[ExampleMatch],
    glossary: Sequence[GlossaryMatch],
    target_lang: str,
) -> str:
    sections: list[str] = ["Source (canonical):", source_text]
    if examples:
        sections.append("\nFew-shot examples:")
        sections.append(_format_examples(examples))
    if glossary:
        sections.append("\nGlossary:")
        sections.append(_format_glossary(glossary))
    sections.append(f"\nTranslate the source text into {target_lang}.")
    return "\n".join(sections).strip()


def build_prompt_payload(
    *,
    config: Config,
    msgctxt: str | None,
    msgid: str,
    msgid_plural: str | None,
    target_lang: str,
    examples: Sequence[ExampleMatch] | None = None,
    glossary: Sequence[GlossaryMatch] | None = None,
) -> PromptData:
    if not target_lang:
        raise ValueError("target_lang missing")
    source_lang = config.languages.source
    if not source_lang:
        raise ValueError("languages.source missing")

    examples = list(examples or [])
    glossary = list(glossary or [])

    source_text = source_text_v1(msgctxt, msgid, msgid_plural)
    system = _system_prompt(source_lang, target_lang)
    user = _user_prompt(
        source_text,
        examples=examples,
        glossary=glossary,
        target_lang=target_lang,
    )

    return {
        "source_context": _normalize_field(msgctxt),
        "source_text": msgid,
        "plural_text": _normalize_field(msgid_plural),
        "target_lang": target_lang,
        "glossary_context": _glossary_context(glossary),
        "few_shot_examples": _format_examples(examples),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    }
