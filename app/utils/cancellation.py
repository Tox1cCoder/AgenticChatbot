"""Work that must finish even when the task awaiting it is cancelled.

A client disconnecting from a streamed response cancels the producer through
anyio, whose cancellation is level-triggered: every await on the way out — the
cleanup included — is cancelled again at once. A Stop on the owning worker
cancels twice, once locally and once through the control bus. Durable writes
that close out a turn therefore cannot be plain awaits.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)

# Tasks still running after their caller was cancelled. The event loop keeps
# only a weak reference to a task, so one nobody holds can be collected mid-run.
_detached: set[asyncio.Future[Any]] = set()


def keep_until_done(future: asyncio.Future[Any]) -> None:
    """Hold a strong reference to ``future`` until it finishes."""
    _detached.add(future)
    future.add_done_callback(_detached.discard)


async def run_to_completion(func: Callable[..., T], *args: Any) -> T:
    """Run blocking work in a thread, to completion even if the caller is cancelled.

    ``asyncio.to_thread`` is not enough for cleanup: cancelling it cancels the
    executor job too when no thread has picked it up yet, and then the job
    never runs. Under level-triggered cancellation that was nine runs in ten.

    The caller still sees its ``CancelledError``; only the work is protected.
    """
    return await asyncio.shield(asyncio.get_running_loop().run_in_executor(None, func, *args))


async def await_to_completion(
    awaitable: Awaitable[T],
    *,
    on_abandoned: Callable[[T], Awaitable[Any]] | None = None,
) -> T:
    """Await ``awaitable`` to completion even if the caller is cancelled.

    The pattern ``AsyncSession.__aexit__`` uses for its close: the work runs as
    its own task, shielded, and held so it cannot be collected. The caller
    still sees its ``CancelledError``.

    ``on_abandoned`` handles a result nobody is left to receive. Work that
    *opens* something — a lifecycle row, a lease — finishes after its caller
    is gone, and whatever it opened must then be closed; ``on_abandoned`` is
    called with the result and its awaitable is run the same way.
    """
    task = asyncio.ensure_future(awaitable)
    keep_until_done(task)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        task.add_done_callback(partial(_finish_abandoned, on_abandoned))
        raise


def _finish_abandoned(
    on_abandoned: Callable[[Any], Awaitable[Any]] | None, task: asyncio.Future[Any]
) -> None:
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        # Nobody awaits this task any more, so this is the only report of it.
        logger.warning(
            "Work that outlived its cancelled caller failed: %s",
            type(error).__name__,
            exc_info=error,
        )
        return
    if on_abandoned is not None:
        keep_until_done(asyncio.ensure_future(on_abandoned(task.result())))
