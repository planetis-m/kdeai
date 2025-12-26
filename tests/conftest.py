from __future__ import annotations

import asyncio
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_EVENT_LOOP: asyncio.AbstractEventLoop | None = None


def pytest_sessionstart() -> None:
    global _EVENT_LOOP
    try:
        asyncio.get_running_loop()
        return
    except RuntimeError:
        pass
    _EVENT_LOOP = asyncio.new_event_loop()
    asyncio.set_event_loop(_EVENT_LOOP)


def pytest_sessionfinish() -> None:
    global _EVENT_LOOP
    if _EVENT_LOOP is None:
        return
    _EVENT_LOOP.close()
    _EVENT_LOOP = None
    asyncio.set_event_loop(None)
