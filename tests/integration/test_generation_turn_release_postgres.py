"""Every way a turn can end must release its conversation for the next one.

``uq_generations_active_per_conversation`` admits one *active* generation per
conversation. A branch of the stream that ends without moving its row out of
the active statuses therefore wedges the conversation: the next message is
refused with ``conversation_turn_conflict`` until the startup reaper fails the
row. The repository double in ``tests/generation_control_support.py`` does not
model that index, so no test noticed when the approval, disconnect, exception
and resume branches all left the row ``running``.

Each scenario here drives a real ``MessageService`` stream through one ending,
then starts a second turn in the same conversation and requires it to be
accepted. It runs twice: once over PostgreSQL, where the partial unique index
itself is what refuses (skipped without ``TEST_DATABASE_URL``), and once over a
double that enforces the same index, so the suite still catches a regression
on machines without a test database.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import os
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine, delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.ai.workflow.state import build_checkpoint_thread_id
from app.models.base import Base
from app.models.conversation import Conversation
from app.models.enums import MessageRole
from app.models.generation import (
    ACTIVE_STATUSES,
    Generation,
    GenerationCommand,
    GenerationStatus,
)
from app.models.message import Message
from app.models.user import User
from app.repositories.generation import GenerationRepository
from app.schemas.generation import StopGenerationCommand
from app.schemas.message import MessageCreate, MessageRead
from app.schemas.workflow import WorkflowExecutionRequest, WorkflowPlanningContext
from app.services.event_streaming.events import make_event
from app.services.generation_control_bus import InMemoryGenerationControlBus
from app.services.generation_control_service import GenerationControlService
from app.services.generation_registry import get_generation_registry
from app.services.message_service import MessageService
from tests.generation_control_support import FakeRepository

pytestmark = pytest.mark.selector_event_loop


# ----------------------------------------------------------------------
# backends
# ----------------------------------------------------------------------


class _IndexedFakeRepository(FakeRepository):
    """The double, plus the one index it was missing.

    Raises where PostgreSQL raises: on an INSERT, or an UPDATE, that would give
    a conversation a second active row.
    """

    def _another_active(self, conversation_id: UUID, *, except_id: UUID | None) -> bool:
        return any(
            row["conversation_id"] == conversation_id
            and row["status"] in ACTIVE_STATUSES
            and generation_id != except_id
            for generation_id, row in self.rows.items()
        )

    @staticmethod
    def _violation() -> IntegrityError:
        return IntegrityError(
            "generations", {}, Exception("uq_generations_active_per_conversation")
        )

    async def acreate(self, command):
        if self._another_active(command.conversation_id, except_id=None):
            raise self._violation()
        return await super().acreate(command)

    async def atransition(self, *, generation_id, conversation_id, values, **kwargs):
        if values.get("status") in ACTIVE_STATUSES and self._another_active(
            conversation_id, except_id=generation_id
        ):
            raise self._violation()
        return await super().atransition(
            generation_id=generation_id, conversation_id=conversation_id, values=values, **kwargs
        )


class _Backend:
    """What a scenario needs from the store, whichever store it is."""

    def __init__(self, *, repository: Any, user_id: UUID, conversation_id: UUID) -> None:
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.repository = repository
        self.control = GenerationControlService(
            repository=repository, bus=InMemoryGenerationControlBus(), stop_wait_seconds=0.5
        )

    def insert_message(self, message_id: UUID, content: str) -> None:
        """The fake keeps nothing; PostgreSQL needs the row for the foreign key."""

    def statuses(self) -> list[GenerationStatus]:
        return [
            row["status"]
            for row in self.repository.rows.values()
            if row["conversation_id"] == self.conversation_id
        ]


class _PostgresBackend(_Backend):
    def __init__(self, *, session_factory: Any, async_session_factory: Any) -> None:
        self._session_factory = session_factory
        self._sequence = itertools.count(1)
        super().__init__(
            repository=GenerationRepository(
                session_factory=session_factory, async_session_factory=async_session_factory
            ),
            user_id=uuid4(),
            conversation_id=uuid4(),
        )
        with session_factory.begin() as session:
            session.add(
                User(
                    id=self.user_id,
                    username=f"release-{self.user_id}",
                    email=f"{self.user_id}@example.test",
                    password_hash="test",
                )
            )
        with session_factory.begin() as session:
            session.add(
                Conversation(id=self.conversation_id, owner_id=self.user_id, title="release")
            )

    def insert_message(self, message_id: UUID, content: str) -> None:
        with self._session_factory.begin() as session:
            session.add(
                Message(
                    id=message_id,
                    conversation_id=self.conversation_id,
                    sender=MessageRole.assistant,
                    content=content or "-",
                    sequence=next(self._sequence),
                )
            )

    def statuses(self) -> list[GenerationStatus]:
        with self._session_factory() as session:
            return list(
                session.execute(
                    select(Generation.status).where(
                        Generation.conversation_id == self.conversation_id
                    )
                ).scalars()
            )

    def cleanup(self) -> None:
        with self._session_factory.begin() as session:
            generation_ids = select(Generation.id).where(Generation.user_id == self.user_id)
            session.execute(
                delete(GenerationCommand).where(GenerationCommand.generation_id.in_(generation_ids))
            )
            session.execute(delete(Generation).where(Generation.user_id == self.user_id))
            session.execute(delete(Message).where(Message.conversation_id == self.conversation_id))
            session.execute(delete(Conversation).where(Conversation.id == self.conversation_id))
            session.execute(delete(User).where(User.id == self.user_id))


def _async_url(database_url: str) -> str:
    for prefix in ("postgresql+psycopg2://", "postgresql://"):
        if database_url.startswith(prefix):
            return database_url.replace(prefix, "postgresql+psycopg://", 1)
    return database_url


@pytest.fixture(scope="module")
def pg_engines():
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is required for PostgreSQL integration tests")
    engine = create_engine(database_url)
    Base.metadata.create_all(
        engine,
        tables=[
            User.__table__,
            Conversation.__table__,
            Message.__table__,
            Generation.__table__,
            GenerationCommand.__table__,
        ],
    )
    async_engine = create_async_engine(_async_url(database_url))
    try:
        yield (
            sessionmaker(bind=engine, expire_on_commit=False),
            async_sessionmaker(bind=async_engine, expire_on_commit=False),
        )
    finally:
        engine.dispose()
        if sys.platform == "win32":
            asyncio.run(async_engine.dispose(), loop_factory=asyncio.SelectorEventLoop)
        else:
            asyncio.run(async_engine.dispose())


@pytest.fixture(params=["indexed_double", "postgres"])
def backend(request) -> Iterator[_Backend]:
    if request.param == "indexed_double":
        yield _Backend(
            repository=_IndexedFakeRepository(), user_id=uuid4(), conversation_id=uuid4()
        )
        return
    session_factory, async_session_factory = request.getfixturevalue("pg_engines")
    pg = _PostgresBackend(
        session_factory=session_factory, async_session_factory=async_session_factory
    )
    try:
        yield pg
    finally:
        pg.cleanup()


@pytest.fixture(autouse=True)
def _clean_registry():
    registry = get_generation_registry()
    registry._store.clear()  # noqa: SLF001 - test isolation
    yield
    registry._store.clear()  # noqa: SLF001


# ----------------------------------------------------------------------
# the service under test
# ----------------------------------------------------------------------


def _row(message_id: UUID, conversation_id: UUID, content: str, sender: int) -> SimpleNamespace:
    now = datetime.now(UTC)
    return SimpleNamespace(
        id=message_id,
        conversation_id=conversation_id,
        sender=sender,
        content=content,
        message_metadata={},
        feedback=None,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )


class _Turns:
    """A real ``MessageService`` whose model and storage are scripted.

    Everything the lifecycle touches is real: the control service, the
    repository, the registry, the stream branches. What is scripted is the
    graph (one queued source per call) and message persistence, which inserts
    the row the generation's foreign key points at and nothing else.
    """

    def __init__(self, backend: _Backend) -> None:
        self.backend = backend
        self.turn_sources: list[Any] = []
        self.resume_sources: list[Any] = []
        self.continue_sources: list[Any] = []
        self.service = self._build()

    def _bot_message(self, *, conversation_id, content, metadata=None, message_id=None):
        message_id = message_id or uuid4()
        self.backend.insert_message(message_id, content)
        return MessageRead.model_validate(
            _row(message_id, conversation_id, content, MessageRole.assistant.value)
        )

    def _build(self) -> MessageService:
        service = MessageService.__new__(MessageService)
        for name in (
            "_turn_coordinator",
            "custom_agent_service",
            "chat_image_service",
            "web_image_service",
            "tool_approval_setting_repository",
            "tool_approval_repository",
            "hitl_interrupt_repository",
            "task_plan_service",
            "project_context_service",
            "redis_client",
        ):
            setattr(service, name, None)
        service.generation_control_service = self.backend.control

        def create_user_message(entity):
            return _row(uuid4(), self.backend.conversation_id, "hi", MessageRole.user.value)

        async def acreate_user_message(entity):
            return create_user_message(entity)

        service.repository = SimpleNamespace(
            create=create_user_message, acreate=acreate_user_message
        )
        conversation = SimpleNamespace(
            title="Existing chat",
            persona_prompt=None,
            planning_mode_enabled=False,
            plan_lifecycle=None,
            owner_id=self.backend.user_id,
        )

        async def aconversation(_conversation_id):
            return conversation

        async def avalidate(*_args):
            return None

        service.conversation_validation_utils = SimpleNamespace(
            validate_conversation_access=lambda *_args: None,
            avalidate_conversation_access=avalidate,
            conversation_repository=SimpleNamespace(
                get_by_id=lambda _cid: conversation, aget_by_id=aconversation
            ),
        )

        async def build_request(*, message_create_data, user_message_id, **_kwargs):
            return (
                self.backend.user_id,
                None,
                WorkflowExecutionRequest(
                    message=message_create_data.content,
                    conversation_id=str(self.backend.conversation_id),
                    user_id=str(self.backend.user_id),
                    user_message_id=str(user_message_id),
                    planning=WorkflowPlanningContext(),
                ),
            )

        service._build_user_message_workflow_request = build_request
        service._create_bot_response_message = self._bot_message

        async def abot_message(**kwargs):
            return self._bot_message(**kwargs)

        service._acreate_bot_response_message = abot_message

        async def persist_completed(**kwargs):
            return self._bot_message(
                conversation_id=kwargs["conversation_id"],
                content="the answer",
                message_id=kwargs.get("message_id"),
            )

        service._persist_completed_workflow_response = persist_completed

        async def no_compaction(**_kwargs):
            return None

        service._compact_checkpoint_after_persist = no_compaction
        service.ai_service = SimpleNamespace(
            invalidate_history_cache=lambda *_args: None,
            execute_request_stream=lambda request: self.turn_sources.pop(0)(request),
            resume_interrupted_execution_stream=lambda **kwargs: self.resume_sources.pop(0)(
                **kwargs
            ),
            resume_generation_control_stream=lambda **kwargs: self.continue_sources.pop(0)(
                **kwargs
            ),
        )
        return service

    def turn(self, *, role: MessageRole = MessageRole.user):
        return self.service.create_message_stream(
            MessageCreate(conversation_id=self.backend.conversation_id, content="hi", role=role),
            self.backend.user_id,
        )

    def resume(self, thread_id: str, interrupt_id: str = "interrupt-1"):
        return self.service.resume_message_creation_stream(
            thread_id=thread_id,
            conversation_id=self.backend.conversation_id,
            user_id=self.backend.user_id,
            decisions=[],
            interrupt_id=interrupt_id,
        )

    def continue_(self, offer: dict[str, Any], *, key: str = "continue-key-0001"):
        return self.service.continue_message_generation_stream(
            generation_id=UUID(offer["generation_id"]),
            continuation_id=UUID(offer["continuation_id"]),
            conversation_id=self.backend.conversation_id,
            user_id=self.backend.user_id,
            idempotency_key=key,
            expected_version=offer["version"],
        )


async def _drain(stream) -> list:
    return [event async for event in stream]


def _types(events) -> list[str]:
    return [event.type for event in events]


async def _second_turn_is_accepted(turns: _Turns) -> None:
    async def answers(_request):
        yield make_event("message_delta", sequence=1, data={"text": "second"})
        yield make_event("complete", sequence=2, data={"response": None})

    turns.turn_sources.append(answers)
    events = await _drain(turns.turn())

    refused = [event.data for event in events if event.type == "error"]
    assert not refused, f"the next turn was refused: {refused}"
    assert "complete" in _types(events)


def _interrupt_event(request, *, interrupt_id: str = "interrupt-1"):
    thread_id = build_checkpoint_thread_id(request.conversation_id, request.user_message_id)
    return make_event(
        "interrupt",
        sequence=2,
        data={
            "thread_id": thread_id,
            "next": ["chat_agent"],
            "pending_tool_calls": [{"action": "web_search", "tool_call_id": "call-1"}],
            "interrupt": {
                "interrupt_id": interrupt_id,
                "thread_id": thread_id,
                "conversation_id": request.conversation_id,
                "action_requests": [
                    {"action": "web_search", "args": {}, "tool_call_id": "call-1"}
                ],
                "metadata": {},
            },
        },
    )


def _pause_event(*, logical_turn_id: str, epoch: int = 0):
    return make_event(
        "continuation_available",
        sequence=3,
        data={
            "type": "execution_budget_exhausted",
            "generation_id": "",
            "logical_turn_id": logical_turn_id,
            "execution_epoch": epoch,
            "active_agent_id": "search_agent",
            "validated_content": f"Partial findings, epoch {epoch}.",
            "budget": {"model_calls": 7, "tool_calls": 12, "exhausted_by": "tool_calls"},
        },
    )


async def _approval_pause(turns: _Turns) -> str:
    """Run a first turn to an approval interrupt; return its checkpoint thread."""
    threads: list[str] = []

    async def asks_for_approval(request):
        event = _interrupt_event(request)
        threads.append(event.data["thread_id"])
        yield make_event("message_delta", sequence=1, data={"text": "Let me search."})
        yield event

    turns.turn_sources.append(asks_for_approval)
    events = await _drain(turns.turn())
    assert _types(events)[-1] == "interrupt"
    return threads[0]


async def _budget_pause(turns: _Turns) -> dict[str, Any]:
    """Run a first turn to a budget pause; return the offer it published."""

    async def runs_out(request):
        yield make_event("message_delta", sequence=1, data={"text": "Partial"})
        yield _pause_event(logical_turn_id=request.user_message_id)

    turns.turn_sources.append(runs_out)
    events = await _drain(turns.turn())
    offers = [event.data for event in events if event.type == "continuation_available"]
    assert offers, f"the pause offered no continuation: {_types(events)}"
    return offers[0]


async def _cancel_after_first_delta(stream) -> None:
    """What the SSE producer does when the client goes away: cancel the task."""
    first = asyncio.Event()

    async def consume():
        async for event in stream:
            if event.type == "message_delta":
                first.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(first.wait(), timeout=5)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def _blocks_after_a_delta(*_args, **_kwargs):
    yield make_event("message_delta", sequence=1, data={"text": "partial answer"})
    await asyncio.Event().wait()


# ----------------------------------------------------------------------
# the first stream
# ----------------------------------------------------------------------


async def test_a_turn_that_completes_releases_the_conversation(backend):
    """The control: the one ending that already worked."""
    turns = _Turns(backend)
    await _second_turn_is_accepted(turns)
    await _second_turn_is_accepted(turns)


async def test_an_approval_interrupt_releases_the_conversation(backend):
    turns = _Turns(backend)
    await _approval_pause(turns)

    assert not [status for status in backend.statuses() if status in ACTIVE_STATUSES]
    await _second_turn_is_accepted(turns)


async def test_an_approval_pause_cannot_be_continued_like_a_budget_pause(backend):
    """Paused, not continuable: only an approval decision moves it on."""
    turns = _Turns(backend)
    await _approval_pause(turns)

    paused = await backend.control.find_by_logical_turn(
        logical_turn_id=str(next(iter(_logical_turns(backend)))),
        user_id=backend.user_id,
        conversation_id=backend.conversation_id,
    )
    assert paused.status is GenerationStatus.CONTINUABLE
    assert paused.continuation_available is False
    assert paused.continuation_id is None
    assert paused.continuation_block_reason == "tool_approval_required"


def _logical_turns(backend: _Backend) -> list[str]:
    if isinstance(backend, _PostgresBackend):
        with backend._session_factory() as session:  # noqa: SLF001 - test helper
            return list(
                session.execute(
                    select(Generation.logical_turn_id).where(
                        Generation.conversation_id == backend.conversation_id
                    )
                ).scalars()
            )
    return [row["logical_turn_id"] for row in backend.repository.rows.values()]


async def test_a_client_disconnect_releases_the_conversation(backend):
    turns = _Turns(backend)
    turns.turn_sources.append(_blocks_after_a_delta)

    await _cancel_after_first_delta(turns.turn())

    assert backend.statuses() == [GenerationStatus.STOPPED]
    await _second_turn_is_accepted(turns)


async def test_a_closed_stream_releases_the_conversation(backend):
    """``aclose`` raises ``GeneratorExit`` at the yield, not ``CancelledError``."""
    turns = _Turns(backend)
    turns.turn_sources.append(_blocks_after_a_delta)

    stream = turns.turn()
    async for event in stream:
        if event.type == "message_delta":
            break
    await stream.aclose()

    assert backend.statuses() == [GenerationStatus.STOPPED]
    await _second_turn_is_accepted(turns)


async def test_a_disconnect_at_the_first_event_releases_the_conversation(backend):
    """The row is already running when ``run_start`` goes out; so is the client's exit."""
    turns = _Turns(backend)

    stream = turns.turn()
    assert (await anext(stream)).type == "run_start"
    await stream.aclose()

    assert backend.statuses() == [GenerationStatus.STOPPED]
    await _second_turn_is_accepted(turns)


async def test_an_exception_releases_the_conversation(backend):
    turns = _Turns(backend)

    async def explodes(_request):
        yield make_event("message_delta", sequence=1, data={"text": "partial"})
        raise RuntimeError("provider fell over")

    turns.turn_sources.append(explodes)
    events = await _drain(turns.turn())

    assert _types(events)[-1] == "error"
    assert backend.statuses() == [GenerationStatus.FAILED]
    await _second_turn_is_accepted(turns)


async def test_a_stop_on_this_worker_releases_the_conversation(backend):
    """The common Stop: it cancels the producer task, which used to look like a disconnect."""
    turns = _Turns(backend)
    turns.turn_sources.append(_blocks_after_a_delta)
    started: list[dict] = []
    first = asyncio.Event()

    async def consume():
        async for event in turns.turn():
            if event.type == "run_start":
                started.append(event.data)
            if event.type == "message_delta":
                first.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(first.wait(), timeout=5)
    settled = await turns.service.stop_generation(
        generation_id=UUID(started[0]["generation_id"]),
        conversation_id=backend.conversation_id,
        user_id=backend.user_id,
        idempotency_key="stop-key-0001",
        expected_version=started[0]["version"],
    )
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert settled.status is GenerationStatus.STOPPED
    await _second_turn_is_accepted(turns)


async def test_a_stop_from_another_worker_releases_the_conversation(backend):
    """The worker's fence is stale after a Stop it did not see being issued."""
    turns = _Turns(backend)

    async def stopped_elsewhere(request):
        yield make_event("message_delta", sequence=1, data={"text": "partial"})
        running = await backend.control.find_by_logical_turn(
            logical_turn_id=request.user_message_id,
            user_id=backend.user_id,
            conversation_id=backend.conversation_id,
        )
        await backend.control.request_stop(
            StopGenerationCommand(
                generation_id=running.generation_id,
                conversation_id=backend.conversation_id,
                user_id=backend.user_id,
                idempotency_key="stop-key-0001",
                expected_version=running.version,
            )
        )
        yield make_event("tool_execution_end", sequence=2, tool_name="web_search", data={})
        yield make_event("complete", sequence=3, data={"response": None})

    turns.turn_sources.append(stopped_elsewhere)
    events = await _drain(turns.turn())

    assert "complete" not in _types(events)
    assert backend.statuses() == [GenerationStatus.STOPPED]
    await _second_turn_is_accepted(turns)


async def test_a_stop_that_loses_the_race_to_the_answer_releases_the_conversation(backend):
    """The answer finished before the worker looked again; the row still says stop."""
    turns = _Turns(backend)

    async def answers_anyway(request):
        running = await backend.control.find_by_logical_turn(
            logical_turn_id=request.user_message_id,
            user_id=backend.user_id,
            conversation_id=backend.conversation_id,
        )
        await backend.control.request_stop(
            StopGenerationCommand(
                generation_id=running.generation_id,
                conversation_id=backend.conversation_id,
                user_id=backend.user_id,
                idempotency_key="stop-key-0001",
                expected_version=running.version,
            )
        )
        yield make_event("message_delta", sequence=1, data={"text": "the whole answer"})
        yield make_event("complete", sequence=2, data={"response": None})

    turns.turn_sources.append(answers_anyway)
    await _drain(turns.turn())

    assert not [status for status in backend.statuses() if status in ACTIVE_STATUSES]
    await _second_turn_is_accepted(turns)


async def test_a_non_user_message_releases_the_conversation(backend):
    """Nothing is generated for it, so nothing may stay active for it."""
    turns = _Turns(backend)

    await _drain(turns.turn(role=MessageRole.assistant))

    assert not [status for status in backend.statuses() if status in ACTIVE_STATUSES]
    await _second_turn_is_accepted(turns)


# ----------------------------------------------------------------------
# the approval resume
# ----------------------------------------------------------------------


async def test_an_approval_resume_that_completes_releases_the_conversation(backend):
    turns = _Turns(backend)
    thread_id = await _approval_pause(turns)

    async def finishes(**_kwargs):
        yield make_event("message_delta", sequence=1, data={"text": "Found it."})
        yield make_event("complete", sequence=2, data={"response": None})

    turns.resume_sources.append(finishes)
    events = await _drain(turns.resume(thread_id))

    assert _types(events)[-1] == "complete"
    assert backend.statuses() == [GenerationStatus.COMPLETED]
    await _second_turn_is_accepted(turns)


async def test_an_approval_resume_runs_as_the_active_turn(backend):
    """While it streams, the resumed turn holds the conversation like any turn."""
    turns = _Turns(backend)
    thread_id = await _approval_pause(turns)
    during: list[list[GenerationStatus]] = []

    async def observes(**_kwargs):
        during.append(backend.statuses())
        yield make_event("complete", sequence=1, data={"response": None})

    turns.resume_sources.append(observes)
    await _drain(turns.resume(thread_id))

    assert during == [[GenerationStatus.RUNNING]]


async def test_an_approval_resume_that_fails_releases_the_conversation(backend):
    turns = _Turns(backend)
    thread_id = await _approval_pause(turns)

    async def explodes(**_kwargs):
        raise ValueError("Workflow is not waiting on a human decision")
        yield  # pragma: no cover

    turns.resume_sources.append(explodes)
    events = await _drain(turns.resume(thread_id))

    assert _types(events)[-1] == "error"
    assert backend.statuses() == [GenerationStatus.FAILED]
    await _second_turn_is_accepted(turns)


async def test_an_approval_resume_error_event_releases_the_conversation(backend):
    turns = _Turns(backend)
    thread_id = await _approval_pause(turns)

    async def reports_error(**_kwargs):
        yield make_event("error", sequence=1, data={"error": "tool failed"})

    turns.resume_sources.append(reports_error)
    await _drain(turns.resume(thread_id))

    assert backend.statuses() == [GenerationStatus.FAILED]
    await _second_turn_is_accepted(turns)


async def test_an_approval_resume_disconnect_releases_the_conversation(backend):
    turns = _Turns(backend)
    thread_id = await _approval_pause(turns)
    turns.resume_sources.append(_blocks_after_a_delta)

    await _cancel_after_first_delta(turns.resume(thread_id))

    assert backend.statuses() == [GenerationStatus.STOPPED]
    await _second_turn_is_accepted(turns)


async def test_a_second_approval_after_a_resume_pauses_again(backend):
    turns = _Turns(backend)
    thread_id = await _approval_pause(turns)

    async def asks_again(**_kwargs):
        request = SimpleNamespace(
            conversation_id=str(backend.conversation_id),
            user_message_id=thread_id.rsplit(":", 1)[-1],
        )
        yield _interrupt_event(request, interrupt_id="interrupt-2")

    turns.resume_sources.append(asks_again)
    events = await _drain(turns.resume(thread_id))

    assert _types(events)[-1] == "interrupt"
    assert backend.statuses() == [GenerationStatus.CONTINUABLE]
    await _second_turn_is_accepted(turns)


async def test_a_budget_pause_after_an_approval_is_persisted_and_offered(backend):
    turns = _Turns(backend)
    thread_id = await _approval_pause(turns)

    async def runs_out(**_kwargs):
        yield make_event("message_delta", sequence=1, data={"text": "Partial"})
        yield _pause_event(logical_turn_id=thread_id.rsplit(":", 1)[-1])

    turns.resume_sources.append(runs_out)
    events = await _drain(turns.resume(thread_id))

    assert _types(events)[-2:] == ["message_end", "continuation_available"]
    offer = events[-1].data
    assert offer["continuation_available"] is True
    assert offer["continuation_id"]
    assert backend.statuses() == [GenerationStatus.CONTINUABLE]
    await _second_turn_is_accepted(turns)


# ----------------------------------------------------------------------
# Continue
# ----------------------------------------------------------------------


async def test_a_continuation_error_releases_the_conversation(backend):
    turns = _Turns(backend)
    offer = await _budget_pause(turns)

    async def reports_error(**_kwargs):
        yield make_event("error", sequence=1, data={"error": "not paused at a budget"})

    turns.continue_sources.append(reports_error)
    await _drain(turns.continue_(offer))

    assert backend.statuses() == [GenerationStatus.FAILED]
    await _second_turn_is_accepted(turns)


async def test_a_continuation_exception_releases_the_conversation(backend):
    turns = _Turns(backend)
    offer = await _budget_pause(turns)

    async def explodes(**_kwargs):
        raise RuntimeError("checkpointer unavailable")
        yield  # pragma: no cover

    turns.continue_sources.append(explodes)
    events = await _drain(turns.continue_(offer))

    assert _types(events)[-1] == "error"
    assert backend.statuses() == [GenerationStatus.FAILED]
    await _second_turn_is_accepted(turns)


async def test_a_continuation_disconnect_releases_the_conversation(backend):
    turns = _Turns(backend)
    offer = await _budget_pause(turns)
    turns.continue_sources.append(_blocks_after_a_delta)

    await _cancel_after_first_delta(turns.continue_(offer))

    assert backend.statuses() == [GenerationStatus.STOPPED]
    await _second_turn_is_accepted(turns)


async def test_a_replayed_continue_does_not_resume_the_graph(backend):
    """The replay carries the first lease's epoch; the pause node refuses it by finalizing.

    So a replay that reaches the graph does not merely fail: it ends the live
    pause the *next* Continue was going to redeem.
    """
    turns = _Turns(backend)
    offer = await _budget_pause(turns)
    graph = {"epoch": 0, "finalized": False}
    calls: list[dict] = []

    async def pause_node(**kwargs):
        calls.append(kwargs)
        if kwargs["expected_epoch"] != graph["epoch"]:
            graph["finalized"] = True
            yield make_event("complete", sequence=1, data={"response": None})
            return
        graph["epoch"] += 1
        logical_turn_id = kwargs["thread_id"].rsplit(":", 1)[-1]
        yield _pause_event(logical_turn_id=logical_turn_id, epoch=graph["epoch"])

    turns.continue_sources.extend([pause_node, pause_node])
    first = await _drain(turns.continue_(offer))
    live_offer = next(e.data for e in first if e.type == "continuation_available")

    replay = await _drain(turns.continue_(offer))

    assert len(calls) == 1, "the replay resumed the graph a second time"
    assert not graph["finalized"], "the replay finalized the live pause"
    assert _types(replay) == ["error"]
    turns.continue_sources.append(pause_node)
    second = await _drain(turns.continue_(live_offer, key="continue-key-0002"))
    assert "continuation_available" in _types(second)
    assert len(calls) == 2
