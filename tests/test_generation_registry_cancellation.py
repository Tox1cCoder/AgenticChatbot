"""The registry accelerates cancellation; it never decides the outcome.

Before the generation row existed this was the only record a turn was running,
so a Stop that missed it reported "not in flight" and a turn kept going. Now the
durable row is the authority and this is a local shortcut: it holds the
cooperative event and the producer task so a Stop landing on the owning worker
does not have to wait for the next durable check.

Which is why the entry outlives an HTTP wait timeout. Removing it there is how
a pending Stop became an unstoppable turn: the request gave up, the entry went
away, and the retry found nothing to cancel.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from app.services.generation_registry import GenerationRegistry


def _registry() -> GenerationRegistry:
    return GenerationRegistry()


async def test_an_entry_is_addressable_by_generation_id():
    registry = _registry()
    generation_id = uuid4()

    registry.register(generation_id, uuid4(), uuid4())

    assert registry.get(generation_id) is not None


async def test_requesting_cancel_sets_the_cooperative_event():
    registry = _registry()
    generation_id = uuid4()
    entry = registry.register(generation_id, uuid4(), uuid4())

    assert registry.request_cancel(generation_id) is True
    assert entry.is_cancelled is True


async def test_requesting_cancel_also_cancels_the_producer_task():
    """The cooperative event only helps between awaits.

    A worker blocked in a provider call reaches no check point, so the task
    itself has to be cancelled or Stop waits for the provider to finish.
    """
    registry = _registry()
    generation_id = uuid4()
    started = asyncio.Event()

    async def producer() -> None:
        started.set()
        await asyncio.sleep(30)

    task = asyncio.create_task(producer())
    await started.wait()
    entry = registry.register(generation_id, uuid4(), uuid4())
    entry.task = task

    registry.request_cancel(generation_id)
    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled()


async def test_requesting_cancel_for_an_unknown_generation_reports_false():
    """The owner is in another process. Not an error, just not here."""
    assert _registry().request_cancel(uuid4()) is False


async def test_a_finished_task_is_not_cancelled_again():
    registry = _registry()
    generation_id = uuid4()

    async def producer() -> str:
        return "done"

    task = asyncio.create_task(producer())
    await task
    entry = registry.register(generation_id, uuid4(), uuid4())
    entry.task = task

    assert registry.request_cancel(generation_id) is True
    assert task.cancelled() is False


async def test_cancel_is_idempotent():
    registry = _registry()
    generation_id = uuid4()
    registry.register(generation_id, uuid4(), uuid4())

    assert registry.request_cancel(generation_id) is True
    assert registry.request_cancel(generation_id) is True


async def test_the_entry_survives_a_wait_timeout():
    """Explicitly not removed on timeout.

    The HTTP request gives up; the worker has not. Dropping the entry here is
    what made a retried Stop find nothing to cancel.
    """
    registry = _registry()
    generation_id = uuid4()
    registry.register(generation_id, uuid4(), uuid4())
    registry.request_cancel(generation_id)

    assert registry.get(generation_id) is not None


async def test_the_worker_removes_the_entry_when_it_finishes():
    registry = _registry()
    generation_id = uuid4()
    registry.register(generation_id, uuid4(), uuid4())

    assert registry.remove(generation_id) is not None
    assert registry.get(generation_id) is None
    assert registry.remove(generation_id) is None


async def test_the_custom_agent_lock_queries_still_work():
    """These gate custom-agent edit/delete while a turn is using the agent.

    They read owner, conversation and active agent — none of which changed when
    the key did, and breaking them would let a user edit an agent mid-turn.
    """
    registry = _registry()
    owner_id, conversation_id = uuid4(), uuid4()
    registry.register(uuid4(), conversation_id, owner_id, active_agent_id="custom:abc")

    assert registry.is_runtime_agent_in_use(str(owner_id), "custom:abc") is True
    assert len(registry.find_by_conversation(conversation_id)) == 1
    assert len(registry.find_by_user(owner_id)) == 1
