import pytest
import polib

from kdeai import state as kdestate


def test_entry_state_hash_requires_explicit_flags_and_prefixes() -> None:
    entry = polib.POEntry(msgid="Hello")
    with pytest.raises(TypeError):
        kdestate.entry_state_hash(entry, lang="de")
