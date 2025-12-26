from __future__ import annotations

from typing import Mapping

import dspy

from kdeai.config import Config


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


def configure_dspy(config: Config) -> None:
    model_id = _resolve_model_id(config)
    lm_class = getattr(dspy, "LM", None) or getattr(dspy, "OpenAI", None)
    if lm_class is None:
        raise RuntimeError("dspy.LM or dspy.OpenAI is required to configure the LM")
    dspy.configure(lm=lm_class(model_id))
