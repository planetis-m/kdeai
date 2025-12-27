from __future__ import annotations

from typing import Mapping, MutableMapping, TypeAlias

TmKey: TypeAlias = tuple[str, str]
TmValue: TypeAlias = dict[str, str | int | dict[str, str]]
SessionTm: TypeAlias = MutableMapping[TmKey, TmValue]
SessionTmView: TypeAlias = Mapping[TmKey, TmValue]
