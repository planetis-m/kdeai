from __future__ import annotations

import dspy

from kdeai.config import Config


def _resolve_model_id(config: Config) -> str:
    generation_model_id = config.prompt.generation_model_id
    if generation_model_id:
        return str(generation_model_id)
    raise ValueError("prompt.generation_model_id missing")


def configure_dspy(config: Config) -> None:
    model_id = _resolve_model_id(config)
    lm_class = getattr(dspy, "LM", None) or getattr(dspy, "OpenAI", None)
    if lm_class is None:
        raise RuntimeError("dspy.LM or dspy.OpenAI is required to configure the LM")
    dspy.configure(lm=lm_class(model_id))
