from __future__ import annotations

from typing import Sequence
import math


def normalize_embedding(values: Sequence[float], normalization: str) -> list[float]:
    if normalization == "none":
        return [float(value) for value in values]
    if normalization == "l2_normalize":
        floats = [float(value) for value in values]
        norm = math.sqrt(sum(value * value for value in floats))
        if norm == 0.0:
            return floats
        return [value / norm for value in floats]
    raise ValueError(f"unsupported embedding normalization: {normalization}")
