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


# ----------------------------------------------------------------------
# the turn path actually takes the lock
# ----------------------------------------------------------------------
#
# The coordinator existing and being unit-tested proves nothing about
# production: a lock nobody acquires serializes nothing. These assert that the
# real turn entrypoints enter it, and that a conflict reaches the caller as a
# typed event rather than an unhandled exception.

from datetime import datetime, timezone  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import AsyncMock  # noqa: E402
from uuid import uuid4  # noqa: E402

from app.models.enums import MessageRole  # noqa: E402
from app.schemas.message import MessageCreate, MessageRead  # noqa: E402
from app.schemas.workflow import (  # noqa: E402
    WorkflowExecutionRequest,
    WorkflowPlanningContext,
    WorkflowResponse,
    WorkflowResponseMessage,
)
from app.services.event_streaming.events import make_event  # noqa: E402
from app.services.message_service import MessageService  # noqa: E402

from .conftest import async_double  # noqa: E402


def _row(*, conversation_id, sender: int, content: str) -> SimpleNamespace:
    now = datetime.now(timezone.utc)
    return SimpleNamespace(
        id=uuid4(),
        conversation_id=conversation_id,
        sender=sender,
        content=content,
        message_metadata={},
        feedback=None,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )


def _streaming_service(conversation_id, user_id, coordinator, *, observed=None):
    """A MessageService wired with doubles for everything but the lock."""
    service = MessageService.__new__(MessageService)
    service._turn_coordinator = coordinator

    def _create(entity):
        return _row(
            conversation_id=conversation_id,
            sender=MessageRole.user.value,
            content=entity["content"],
        )

    def _get_by_id(_conversation_id):
        return SimpleNamespace(title="Existing chat")

    service.repository = SimpleNamespace(create=_create, acreate=async_double(_create))
    service.conversation_validation_utils = SimpleNamespace(
        validate_conversation_access=lambda *_a: None,
        avalidate_conversation_access=async_double(lambda *_a: None),
        conversation_repository=SimpleNamespace(
            get_by_id=_get_by_id, aget_by_id=async_double(_get_by_id)
        ),
    )
    workflow_request = WorkflowExecutionRequest(
        message="hello",
        conversation_id=str(conversation_id),
        user_id=str(user_id),
        planning=WorkflowPlanningContext(),
    )

    async def _build(**_kwargs):
        if observed is not None:
            observed.append("snapshot")
        return (user_id, None, workflow_request)

    service._build_user_message_workflow_request = _build

    async def source(_request):
        await asyncio.sleep(0.01)
        yield make_event("message_delta", sequence=1, data={"text": "hello"})
        yield make_event(
            "complete",
            sequence=2,
            data={
                "response": WorkflowResponse(
                    message=WorkflowResponseMessage(content="hello"),
                    metadata={},
                )
            },
        )

    service.ai_service = SimpleNamespace(
        invalidate_history_cache=lambda *_a: None,
        execute_request_stream=source,
    )

    async def _persist(**_kwargs):
        if observed is not None:
            observed.append("persist")
        return MessageRead.model_validate(
            _row(
                conversation_id=conversation_id,
                sender=MessageRole.assistant.value,
                content="hello",
            )
        )

    service._persist_completed_workflow_response = _persist
    service._compact_checkpoint_after_persist = AsyncMock()
    return service


async def test_the_streaming_turn_path_holds_the_conversation_lock():
    """The lock must span snapshot through persistence, not just generation."""
    conversation_id = uuid4()
    backend = RecordingBackend()
    coordinator = _coordinator(backend)
    observed: list[str] = []
    service = _streaming_service(conversation_id, uuid4(), coordinator, observed=observed)

    events = [
        event
        async for event in service.create_message_stream(
            MessageCreate(conversation_id=conversation_id, content="hello"), uuid4()
        )
    ]

    assert events[-1].type == "complete"
    assert backend.acquired == [str(conversation_id)]
    assert backend.released == [str(conversation_id)]
    # Snapshot and persistence both happened inside the held lock.
    assert observed == ["snapshot", "persist"]


async def test_two_streams_in_one_conversation_do_not_overlap():
    conversation_id = uuid4()
    backend = RecordingBackend()
    coordinator = _coordinator(backend)

    async def run() -> None:
        service = _streaming_service(conversation_id, uuid4(), coordinator)
        async for _event in service.create_message_stream(
            MessageCreate(conversation_id=conversation_id, content="hello"), uuid4()
        ):
            pass

    await asyncio.gather(run(), run())

    assert backend.max_concurrent == 1


async def test_streams_in_different_conversations_stay_concurrent():
    backend = RecordingBackend()
    coordinator = _coordinator(backend)

    async def run(conversation_id) -> None:
        service = _streaming_service(conversation_id, uuid4(), coordinator)
        async for _event in service.create_message_stream(
            MessageCreate(conversation_id=conversation_id, content="hello"), uuid4()
        ):
            pass

    await asyncio.gather(run(uuid4()), run(uuid4()))

    assert backend.max_concurrent == 2


async def test_a_turn_conflict_reaches_the_client_as_a_typed_error_event():
    """A contended conversation must not surface as an unhandled exception."""
    conversation_id = uuid4()
    backend = RecordingBackend()
    coordinator = _coordinator(backend, timeout_seconds=0.01)
    service = _streaming_service(conversation_id, uuid4(), coordinator)

    async with coordinator.hold(str(conversation_id), request_id="holder"):
        events = [
            event
            async for event in service.create_message_stream(
                MessageCreate(conversation_id=conversation_id, content="hello"), uuid4()
            )
        ]

    error = next(event for event in events if event.type == "error")
    assert error.data["code"] == "conversation_turn_conflict"
    assert error.data["retriable"] is True
    assert not [event for event in events if event.type == "complete"]


def test_the_container_wires_a_durable_turn_coordinator_into_message_service():
    """A coordinator nobody injects is a coordinator nobody acquires."""
    import inspect

    from app.core import container as container_module

    assert "turn_coordinator" in inspect.signature(MessageService.__init__).parameters

    source = inspect.getsource(container_module)
    message_service_block = source.split("message_service: providers.Provider")[1].split(
        "feedback_service"
    )[0]
    assert "turn_coordinator=conversation_turn_coordinator" in message_service_block


# ----------------------------------------------------------------------
# the production backend against a real session factory
# ----------------------------------------------------------------------
#
# The backend's other tests pass ``lambda: None`` and never call acquire, so
# they cannot see what the container actually hands it. The container's
# ``db.provided.session`` is a ``@contextmanager``, which returns a context
# manager rather than a Session — and a context-managed session would be closed
# on exit, releasing the very lock it was taken to hold.


class FakeSession:
    """A sync Session stand-in that records the statements it ran."""

    def __init__(self, *, acquired: bool = True) -> None:
        self._acquired = acquired
        self.statements: list[str] = []
        self.closed = False

    def execute(self, statement, _params=None):
        self.statements.append(str(statement))
        return SimpleNamespace(scalar=lambda: self._acquired)

    def close(self) -> None:
        self.closed = True


async def test_the_postgres_backend_holds_one_session_from_acquire_to_release():
    sessions: list[FakeSession] = []

    def factory() -> FakeSession:
        session = FakeSession()
        sessions.append(session)
        return session

    backend = PostgresAdvisoryLockBackend(session_factory=factory)

    assert await backend.acquire("conversation-1", timeout_seconds=1.0) is True
    assert len(sessions) == 1
    assert sessions[0].closed is False, "the lock's session must stay open while held"

    await backend.release("conversation-1")
    assert sessions[0].closed is True
    assert any("pg_try_advisory_lock" in s for s in sessions[0].statements)
    assert any("pg_advisory_unlock" in s for s in sessions[0].statements)


async def test_the_postgres_backend_rejects_a_context_manager_session_factory():
    """``db.provided.session`` is a @contextmanager — passing it is a wiring bug.

    It must fail with a message naming the problem, not an AttributeError on
    the first query, which is what happens when the wrong factory reaches
    production.
    """
    import contextlib

    @contextlib.contextmanager
    def factory():
        yield FakeSession()

    backend = PostgresAdvisoryLockBackend(session_factory=factory)

    with pytest.raises(TypeError) as exc:
        await backend.acquire("conversation-1", timeout_seconds=1.0)

    assert "session factory" in str(exc.value).lower()


def test_the_container_gives_the_lock_backend_a_raw_session_factory():
    import inspect

    from app.core import container as container_module

    source = inspect.getsource(container_module)
    coordinator_block = source.split("conversation_turn_coordinator = providers.Singleton(")[
        1
    ].split("\n    )")[0]
    assert "PostgresAdvisoryLockBackend" in coordinator_block
    assert "db.provided.session" not in coordinator_block, (
        "db.provided.session is a @contextmanager; the advisory lock needs a "
        "raw session factory it can hold open past the call"
    )
