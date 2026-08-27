"""Same-conversation turns are serialized; different conversations are not.

Two turns in one conversation that overlap would each snapshot history the
other is about to change, so the second turn answers from a view that no longer
exists by the time it writes. The coordinator holds a cross-process lock from
context snapshot through response persistence.

Different conversations share nothing, so serializing them would be pure
latency. The lock is per conversation and nothing wider.
"""

from __future__ import annotations

import asyncio

import pytest

from app.ai.workflow.contracts import WorkflowRoutingException
from app.services.conversation_turn_coordinator import (
    ConversationTurnCoordinator,
    InProcessTurnLockBackend,
    PostgresAdvisoryLockBackend,
)


class RecordingBackend(InProcessTurnLockBackend):
    """In-process backend that also records acquisition order, for tests."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.acquired: list[str] = []
        self.released: list[str] = []
        self.concurrent = 0
        self.max_concurrent = 0

    async def acquire(self, key: str, *, timeout_seconds: float) -> bool:
        acquired = await super().acquire(key, timeout_seconds=timeout_seconds)
        if acquired:
            self.acquired.append(key)
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        return acquired

    async def release(self, key: str) -> None:
        await super().release(key)
        self.released.append(key)
        self.concurrent -= 1


def _coordinator(backend=None, timeout_seconds: float = 1.0) -> ConversationTurnCoordinator:
    return ConversationTurnCoordinator(
        backend=backend or RecordingBackend(), timeout_seconds=timeout_seconds
    )


# ----------------------------------------------------------------------
# serialization
# ----------------------------------------------------------------------


async def test_same_conversation_turns_never_overlap():
    backend = RecordingBackend()
    coordinator = _coordinator(backend)
    observed: list[str] = []

    async def turn(name: str) -> None:
        async with coordinator.hold("conversation-1", request_id=name):
            observed.append(f"{name}:start")
            await asyncio.sleep(0.01)
            observed.append(f"{name}:end")

    await asyncio.gather(turn("first"), turn("second"))

    assert backend.max_concurrent == 1
    # Each turn's start is immediately followed by its own end.
    assert observed[0].endswith(":start")
    assert observed[1] == observed[0].replace(":start", ":end")


async def test_different_conversations_remain_concurrent():
    backend = RecordingBackend()
    coordinator = _coordinator(backend)

    async def turn(conversation_id: str) -> None:
        async with coordinator.hold(conversation_id, request_id="request-1"):
            await asyncio.sleep(0.02)

    await asyncio.gather(turn("conversation-a"), turn("conversation-b"))

    assert backend.max_concurrent == 2


async def test_the_lock_is_released_after_a_failure():
    backend = RecordingBackend()
    coordinator = _coordinator(backend)

    with pytest.raises(RuntimeError):
        async with coordinator.hold("conversation-1", request_id="request-1"):
            raise RuntimeError("turn blew up")

    assert backend.released == ["conversation-1"]

    # The next turn can still acquire it.
    async with coordinator.hold("conversation-1", request_id="request-2"):
        pass
    assert backend.acquired == ["conversation-1", "conversation-1"]


async def test_the_lock_is_released_on_cancellation():
    backend = RecordingBackend()
    coordinator = _coordinator(backend)

    async def turn() -> None:
        async with coordinator.hold("conversation-1", request_id="request-1"):
            await asyncio.sleep(10)

    task = asyncio.create_task(turn())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert backend.released == ["conversation-1"]


# ----------------------------------------------------------------------
# bounded acquisition
# ----------------------------------------------------------------------


async def test_acquisition_timeout_raises_the_typed_retriable_conflict():
    backend = RecordingBackend()
    coordinator = _coordinator(backend, timeout_seconds=0.02)

    async with coordinator.hold("conversation-1", request_id="holder"):
        with pytest.raises(WorkflowRoutingException) as exc:
            async with coordinator.hold("conversation-1", request_id="waiter"):
                pass

    assert exc.value.error.code == "conversation_turn_conflict"
    assert exc.value.error.retriable is True
    assert exc.value.error.request_id == "waiter"


async def test_a_conflict_does_not_leave_the_lock_held_by_the_loser():
    backend = RecordingBackend()
    coordinator = _coordinator(backend, timeout_seconds=0.02)

    async with coordinator.hold("conversation-1", request_id="holder"):
        with pytest.raises(WorkflowRoutingException):
            async with coordinator.hold("conversation-1", request_id="waiter"):
                pass

    # Only the holder ever released; the waiter never acquired.
    assert backend.released == ["conversation-1"]


async def test_a_missing_conversation_id_does_not_take_a_global_lock():
    """A turn with no conversation must not serialize against every other turn."""
    backend = RecordingBackend()
    coordinator = _coordinator(backend)

    async with (
        coordinator.hold(None, request_id="request-1"),
        coordinator.hold(None, request_id="request-2"),
    ):
        pass

    assert backend.acquired == []


# ----------------------------------------------------------------------
# production backend
# ----------------------------------------------------------------------


def test_the_postgres_backend_derives_a_stable_bounded_lock_key():
    """Advisory locks take a bigint, so the key must hash deterministically."""
    first = PostgresAdvisoryLockBackend.lock_key("conversation-1")
    again = PostgresAdvisoryLockBackend.lock_key("conversation-1")
    other = PostgresAdvisoryLockBackend.lock_key("conversation-2")

    assert first == again
    assert first != other
    assert -(2**63) <= first < 2**63


def test_production_refuses_an_in_process_only_coordinator():
    """An in-process lock is not durable across workers, so it cannot ship."""
    with pytest.raises(ValueError) as exc:
        ConversationTurnCoordinator(
            backend=InProcessTurnLockBackend(), timeout_seconds=1.0, production=True
        )
    assert "in-process" in str(exc.value).lower()


def test_the_postgres_backend_is_accepted_in_production():
    coordinator = ConversationTurnCoordinator(
        backend=PostgresAdvisoryLockBackend(session_factory=lambda: None),
        timeout_seconds=1.0,
        production=True,
    )
    assert coordinator is not None


def test_timeout_must_be_positive():
    with pytest.raises(ValueError):
        ConversationTurnCoordinator(backend=InProcessTurnLockBackend(), timeout_seconds=0.0)
