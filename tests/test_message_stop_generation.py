"""Stop must not claim more than it knows.

Two defects this locks down, both of which shipped untested:

* **The entry was removed when the HTTP wait timed out.** The request gave up;
  the producer had not. Dropping the registry entry there left a retried Stop
  with nothing to cancel while the turn was still running -- the exact failure
  mode that made Stop feel unreliable.
* **A timeout was reported as ``cancelled``.** The producer may be mid-provider
  call in another process. Telling the user their generation stopped when
  nobody has confirmed it is the one thing a Stop control must never do.

The durable lifecycle in ``generations`` replaces this method wholesale (Task 5
of the generation-controls plan). Until it does, the honest answer for a
timeout is a distinct pending status, which is also the vocabulary that plan
uses.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.services.generation_registry import get_generation_registry
from app.services.message_service import MessageService


@pytest.fixture(autouse=True)
def _clean_registry():
    registry = get_generation_registry()
    for key in list(registry._store):  # noqa: SLF001 - test isolation
        registry._store.pop(key, None)  # noqa: SLF001
    yield
    for key in list(registry._store):  # noqa: SLF001
        registry._store.pop(key, None)  # noqa: SLF001


def _service() -> MessageService:
    """The two collaborators this method actually touches, and nothing else."""
    service = MessageService.__new__(MessageService)
    service.conversation_validation_utils = SimpleNamespace(
        validate_conversation_access=lambda user_id, conversation_id: None
    )
    return service


async def test_a_generation_that_was_never_running_is_reported_as_such():
    service = _service()

    result = await service.stop_message_generation(
        conversation_id=uuid4(), user_id=uuid4(), user_message_id=uuid4()
    )

    assert result["status"] == "not_inflight"


async def test_another_users_generation_is_not_stoppable():
    """Nor distinguishable from one that does not exist."""
    service = _service()
    generation_id, conversation_id = uuid4(), uuid4()
    get_generation_registry().register(generation_id, conversation_id, uuid4())

    result = await service.stop_message_generation(
        conversation_id=conversation_id, user_id=uuid4(), user_message_id=generation_id
    )

    assert result["status"] == "not_inflight"


async def test_a_producer_that_finishes_reports_stopped_with_its_partial():
    service = _service()
    generation_id, conversation_id, user_id = uuid4(), uuid4(), uuid4()
    entry = get_generation_registry().register(generation_id, conversation_id, user_id)
    entry.resolve({"id": "partial-message"})

    result = await service.stop_message_generation(
        conversation_id=conversation_id, user_id=user_id, user_message_id=generation_id
    )

    assert result["status"] == "cancelled"
    assert result["message"] == {"id": "partial-message"}


async def test_a_finished_producer_releases_its_registry_entry():
    service = _service()
    generation_id, conversation_id, user_id = uuid4(), uuid4(), uuid4()
    entry = get_generation_registry().register(generation_id, conversation_id, user_id)
    entry.resolve(None)

    await service.stop_message_generation(
        conversation_id=conversation_id, user_id=user_id, user_message_id=generation_id
    )

    assert get_generation_registry().get(generation_id) is None


async def test_a_wait_timeout_is_reported_as_pending_not_as_stopped():
    """The producer never answered. Saying it stopped would be a guess."""
    service = _service()
    generation_id, conversation_id, user_id = uuid4(), uuid4(), uuid4()
    get_generation_registry().register(generation_id, conversation_id, user_id)

    result = await service.stop_message_generation(
        conversation_id=conversation_id,
        user_id=user_id,
        user_message_id=generation_id,
        wait_seconds=0.01,
    )

    assert result["status"] == "stop_requested"
    assert result["message"] is None


async def test_a_wait_timeout_keeps_the_entry_so_a_retry_can_still_cancel():
    """The request gave up; the worker has not."""
    service = _service()
    generation_id, conversation_id, user_id = uuid4(), uuid4(), uuid4()
    get_generation_registry().register(generation_id, conversation_id, user_id)

    await service.stop_message_generation(
        conversation_id=conversation_id,
        user_id=user_id,
        user_message_id=generation_id,
        wait_seconds=0.01,
    )

    assert get_generation_registry().get(generation_id) is not None


async def test_the_cancel_signal_reaches_the_producer_even_on_timeout():
    service = _service()
    generation_id, conversation_id, user_id = uuid4(), uuid4(), uuid4()
    entry = get_generation_registry().register(generation_id, conversation_id, user_id)

    await service.stop_message_generation(
        conversation_id=conversation_id,
        user_id=user_id,
        user_message_id=generation_id,
        wait_seconds=0.01,
    )

    assert entry.is_cancelled is True


async def test_a_retried_stop_after_a_timeout_still_finds_the_generation():
    """The regression the removal-on-timeout caused, stated end to end."""
    service = _service()
    generation_id, conversation_id, user_id = uuid4(), uuid4(), uuid4()
    entry = get_generation_registry().register(generation_id, conversation_id, user_id)

    first = await service.stop_message_generation(
        conversation_id=conversation_id,
        user_id=user_id,
        user_message_id=generation_id,
        wait_seconds=0.01,
    )
    entry.resolve({"id": "partial-message"})
    second = await service.stop_message_generation(
        conversation_id=conversation_id,
        user_id=user_id,
        user_message_id=generation_id,
        wait_seconds=0.01,
    )

    assert first["status"] == "stop_requested"
    assert second["status"] == "cancelled"


async def test_stopping_a_running_producer_interrupts_its_task():
    service = _service()
    generation_id, conversation_id, user_id = uuid4(), uuid4(), uuid4()
    started = asyncio.Event()

    async def producer() -> None:
        started.set()
        await asyncio.sleep(30)

    task = asyncio.create_task(producer())
    await started.wait()
    entry = get_generation_registry().register(generation_id, conversation_id, user_id)
    entry.task = task

    await service.stop_message_generation(
        conversation_id=conversation_id,
        user_id=user_id,
        user_message_id=generation_id,
        wait_seconds=0.01,
    )
    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled()
