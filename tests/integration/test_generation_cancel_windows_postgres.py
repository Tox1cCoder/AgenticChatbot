"""A cancel between a turn's durable writes must not leave its row active.

A turn writes durable state in steps: allocate the row, enter ``running``,
persist the answer, close the row. A Stop, or a client disconnect, cancels the
producer task wherever it happens to be awaiting. Each test here parks the turn
at one await between two of those steps, cancels it there, and requires the row
to leave the active statuses. An active row refuses every later message in the
conversation until the startup reaper fails it, so a wedge here lasts until the
next deploy.

Same harness as ``test_generation_turn_release_postgres``: a real
``MessageService`` and a real ``GenerationControlService``, over PostgreSQL and
over a double that enforces the same partial unique index.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from app.models.generation import ACTIVE_STATUSES, GenerationStatus
from app.services.event_streaming.events import make_event
from tests.integration.test_generation_turn_release_postgres import (
    _approval_pause,
    _Backend,
    _budget_pause,
    _drain,
    _interrupt_event,
    _pause_event,
    _second_turn_is_accepted,
    _Turns,
    _types,
)
from tests.integration.test_generation_turn_release_postgres import (
    _clean_registry as _clean_registry,
)
from tests.integration.test_generation_turn_release_postgres import (
    backend as backend,
)
from tests.integration.test_generation_turn_release_postgres import (
    pg_engines as pg_engines,
)

pytestmark = pytest.mark.selector_event_loop


class _Gate:
    """An await the turn parks at until the test has cancelled it there."""

    def __init__(self) -> None:
        self.reached = asyncio.Event()
        self.opened = asyncio.Event()
        self.passes = 0

    async def wait(self) -> None:
        self.passes += 1
        if self.passes > 1:
            return
        self.reached.set()
        await self.opened.wait()

    def wrap(self, func: Any) -> Any:
        async def gated(*args: Any, **kwargs: Any) -> Any:
            await self.wait()
            return await func(*args, **kwargs)

        return gated


async def _cancel_at(gate: _Gate, stream: Any) -> None:
    """Consume ``stream``; cancel the consumer once the turn reaches ``gate``.

    The gate opens only after the cancelled consumer has finished, so its
    cleanup runs first, before whatever was parked at the gate can settle the
    row; otherwise a cleanup that wrongly touches the row could lose the race
    and go unnoticed.
    """

    async def consume() -> None:
        async for _event in stream:
            pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(gate.reached.wait(), timeout=5)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)
    gate.opened.set()


async def _settled(backend: _Backend) -> list[GenerationStatus]:
    """The conversation's statuses once any shielded settlement has finished."""
    for _ in range(150):
        statuses = backend.statuses()
        if not [status for status in statuses if status in ACTIVE_STATUSES]:
            return statuses
        await asyncio.sleep(0.02)
    return backend.statuses()


def _active(statuses: list[GenerationStatus]) -> list[GenerationStatus]:
    return [status for status in statuses if status in ACTIVE_STATUSES]


async def _answers(_request):
    yield make_event("message_delta", sequence=1, data={"text": "the answer"})
    yield make_event("complete", sequence=2, data={"response": None})


# ----------------------------------------------------------------------
# the first stream
# ----------------------------------------------------------------------


async def test_a_cancel_while_entering_running_closes_the_row(backend):
    """The row is committed ``starting`` before ``mark_running`` is awaited."""
    turns = _Turns(backend)
    gate = _Gate()
    backend.control.mark_running = gate.wrap(backend.control.mark_running)
    turns.turn_sources.append(_answers)

    await _cancel_at(gate, turns.turn())

    assert await _settled(backend) == [GenerationStatus.STOPPED]
    await _second_turn_is_accepted(turns)


async def test_a_cancel_after_the_answer_is_saved_closes_the_row(backend):
    """The answer is persisted; closing the row was still to come."""
    turns = _Turns(backend)
    gate = _Gate()
    compact = turns.service._compact_checkpoint_after_persist
    turns.service._compact_checkpoint_after_persist = gate.wrap(compact)
    turns.turn_sources.append(_answers)

    await _cancel_at(gate, turns.turn())

    statuses = await _settled(backend)
    assert _active(statuses) == []
    assert statuses == [GenerationStatus.COMPLETED], "a saved answer is a completed turn"
    await _second_turn_is_accepted(turns)


async def test_a_cancel_while_offering_a_continuation_closes_the_row(backend):
    """The paused partial is saved; the cleanup must not insert it a second time."""
    turns = _Turns(backend)
    gate = _Gate()
    mark_images = turns.service._mark_persisted_web_images
    turns.service._mark_persisted_web_images = gate.wrap(mark_images)

    async def runs_out(request):
        yield make_event("message_delta", sequence=1, data={"text": "Partial"})
        yield _pause_event(logical_turn_id=request.user_message_id)

    turns.turn_sources.append(runs_out)

    await _cancel_at(gate, turns.turn())

    # The pause finishes as a pause: the partial is saved and Continue offered.
    assert await _settled(backend) == [GenerationStatus.CONTINUABLE]
    await _second_turn_is_accepted(turns)


async def test_closing_the_stream_at_an_approval_keeps_the_approval(backend):
    """The SSE consumer stops reading at an interrupt; that is not a disconnect.

    The approval message is already saved under the reserved id, so a cleanup
    that saves the partial under it again collides with it.
    """
    turns = _Turns(backend)
    saved = turns.service._create_bot_response_message
    stopped_partials: list[dict] = []

    def recording(**kwargs):
        if kwargs.get("metadata", {}).get("stopped"):
            stopped_partials.append(kwargs)
        return saved(**kwargs)

    turns.service._create_bot_response_message = recording

    async def asks_for_approval(request):
        yield make_event("message_delta", sequence=1, data={"text": "Let me search."})
        yield _interrupt_event(request)

    turns.turn_sources.append(asks_for_approval)
    stream = turns.turn()
    async for event in stream:
        if event.type == "interrupt":
            break
    await stream.aclose()

    assert stopped_partials == [], "the cleanup saved the approval's partial a second time"
    assert backend.statuses() == [GenerationStatus.CONTINUABLE]
    await _second_turn_is_accepted(turns)


async def test_a_second_cancel_while_failing_the_turn_closes_the_row(backend):
    """A Stop on the owning worker cancels twice: locally and through the bus."""
    turns = _Turns(backend)
    gate = _Gate()
    backend.control.mark_failed = gate.wrap(backend.control.mark_failed)

    async def breaks(_request):
        yield make_event("message_delta", sequence=1, data={"text": "partial"})
        raise RuntimeError("provider went away")

    turns.turn_sources.append(breaks)

    await _cancel_at(gate, turns.turn())

    assert await _settled(backend) == [GenerationStatus.FAILED]
    await _second_turn_is_accepted(turns)


async def test_a_disconnect_whose_partial_cannot_be_saved_still_closes_the_row(backend):
    """Keeping the partial is best effort; closing the row is not."""
    turns = _Turns(backend)
    delta_seen = asyncio.Event()

    def cannot_save(**_kwargs):
        raise RuntimeError("the database refused the partial")

    turns.service._create_bot_response_message = cannot_save

    async def streams_then_hangs(_request):
        yield make_event("message_delta", sequence=1, data={"text": "partial"})
        delta_seen.set()
        await asyncio.Event().wait()

    turns.turn_sources.append(streams_then_hangs)

    async def consume() -> None:
        async for _event in turns.turn():
            pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(delta_seen.wait(), timeout=5)
    task.cancel()
    # The cancel is what propagates: a cleanup failure raised over it would
    # replace the CancelledError anyio and Starlette are waiting for.
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert await _settled(backend) == [GenerationStatus.STOPPED]
    await _second_turn_is_accepted(turns)


# ----------------------------------------------------------------------
# resume after approval
# ----------------------------------------------------------------------


async def test_a_cancel_after_a_resumed_answer_is_saved_closes_the_row(backend):
    turns = _Turns(backend)
    thread_id = await _approval_pause(turns)
    gate = _Gate()
    compact = turns.service._compact_checkpoint_after_persist
    turns.service._compact_checkpoint_after_persist = gate.wrap(compact)

    async def finishes(**_kwargs):
        yield make_event("message_delta", sequence=1, data={"text": "found it"})
        yield make_event("complete", sequence=2, data={"response": None})

    turns.resume_sources.append(finishes)

    await _cancel_at(gate, turns.resume(thread_id))

    statuses = await _settled(backend)
    assert _active(statuses) == []
    assert statuses == [GenerationStatus.COMPLETED]
    await _second_turn_is_accepted(turns)


# ----------------------------------------------------------------------
# continue after a budget pause
# ----------------------------------------------------------------------


async def test_a_cancel_while_leasing_a_continuation_closes_the_row(backend):
    """``lease_continuation`` commits ``continuing`` before it records the claim."""
    turns = _Turns(backend)
    offer = await _budget_pause(turns)
    gate = _Gate()
    backend.control._record = gate.wrap(backend.control._record)  # noqa: SLF001

    async def finishes(**_kwargs):
        yield make_event("complete", sequence=1, data={"response": None})

    turns.continue_sources.append(finishes)

    await _cancel_at(gate, turns.continue_(offer))

    assert await _settled(backend) == [GenerationStatus.STOPPED]
    await _second_turn_is_accepted(turns)


async def test_the_harness_still_completes_an_uncancelled_turn(backend):
    """The control: gating nothing changes nothing."""
    turns = _Turns(backend)
    turns.turn_sources.append(_answers)
    assert _types(await _drain(turns.turn()))[-1] == "complete"
    assert backend.statuses() == [GenerationStatus.COMPLETED]
