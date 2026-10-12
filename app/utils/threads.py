"""Blocking work run off the event loop that must not be lost to cancellation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


async def run_to_completion(func: Callable[..., T], *args: Any) -> T:
    """Run blocking work in a thread, to completion even if the caller is cancelled.

    ``asyncio.to_thread`` is not enough for cleanup: cancelling it cancels the
    executor job too when no thread has picked it up yet, and then the job
    never runs. Under anyio's level-triggered cancellation — a client
    disconnecting from a streamed response — every await in a ``finally`` is
    cancelled at once, and the job was dropped in roughly nine runs of ten.

    The caller still sees its ``CancelledError``; only the work is protected.
    """
    return await asyncio.shield(asyncio.get_running_loop().run_in_executor(None, func, *args))
