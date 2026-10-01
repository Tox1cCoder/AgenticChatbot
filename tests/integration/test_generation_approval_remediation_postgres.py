"""Durable approval recovery and independent worker serialization regressions."""

from __future__ import annotations

import asyncio
import multiprocessing
import os
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, delete, update
from sqlalchemy.orm import sessionmaker

from app.ai.schemas import InterruptDecision
from app.models.hitl_interrupt import HITLInterrupt, HITLInterruptStatus
from app.models.message import Message
from app.models.user import User
from app.repositories.hitl_interrupt import HITLInterruptRepository
from app.schemas.generation import CreateGeneration, StopGenerationCommand
from app.services.conversation_turn_coordinator import (
    ConversationTurnCoordinator,
    PostgresAdvisoryLockBackend,
)
from app.services.event_streaming.events import make_event
from app.services.generation_registry import get_generation_registry
from tests.integration.test_generation_turn_release_postgres import (
    _approval_pause,
    _budget_pause,
    _drain,
    _interrupt_event,
    _PostgresBackend,
    _second_turn_is_accepted,
    _Turns,
)
from tests.integration.test_generation_turn_release_postgres import (
    pg_engines as pg_engines,
)

pytestmark = pytest.mark.selector_event_loop


@pytest.fixture
def turns(pg_engines):
    session_factory, async_session_factory = pg_engines
    backend = _PostgresBackend(
        session_factory=session_factory, async_session_factory=async_session_factory
    )
    value = _Turns(backend)
    hitl = HITLInterruptRepository(session_factory)
    value.service.hitl_interrupt_repository = hitl
    backend.control._hitl_interrupt_repository = hitl
    value.service._turn_coordinator = ConversationTurnCoordinator(
        backend=PostgresAdvisoryLockBackend(session_factory),
        timeout_seconds=0.05,
    )
    registry = get_generation_registry()
    registry._store.clear()
    persist = value.service._create_bot_response_message

    def persist_metadata(**kwargs):
        message = persist(**kwargs)
        with session_factory.begin() as session:
            session.execute(
                update(Message)
                .where(Message.id == message.id)
                .values(message_metadata=kwargs.get("metadata") or {})
            )
        return message

    value.service._create_bot_response_message = persist_metadata
    try:
        yield value
    finally:
        registry._store.clear()
        with session_factory.begin() as session:
            session.execute(
                delete(HITLInterrupt).where(
                    HITLInterrupt.conversation_id == backend.conversation_id
                )
            )
        backend.cleanup()


def _resume(turns, thread, interrupt_id="interrupt-1"):
    return turns.service.resume_message_creation_stream(
        thread_id=thread,
        conversation_id=turns.backend.conversation_id,
        user_id=turns.backend.user_id,
        interrupt_id=interrupt_id,
        decisions=[InterruptDecision(type="approve", action="web_search", tool_call_id="call-1")],
    )


async def test_continued_approval_is_durable_and_redeemable(turns):
    offer = await _budget_pause(turns)

    async def asks(**kwargs):
        yield _interrupt_event(
            SimpleNamespace(
                conversation_id=str(turns.backend.conversation_id),
                user_message_id=kwargs["thread_id"].rsplit(":", 1)[-1],
            )
        )

    turns.continue_sources.append(asks)
    events = await _drain(turns.continue_(offer))
    interrupt = next(e for e in events if e.type == "interrupt")
    record = turns.service.hitl_interrupt_repository.get_by_id("interrupt-1")
    assert record is not None
    assert record.status is HITLInterruptStatus.PENDING
    assert record.thread_id == interrupt.data["thread_id"]
    with turns.backend._session_factory() as session:
        message = session.get(Message, record.assistant_message_id)
        assert message.message_metadata["interrupt"]["interrupt_id"] == record.id
        assert message.message_metadata["paused"] is True

    async def finish(**kwargs):
        yield make_event("complete", sequence=1, data={"response": None})

    turns.resume_sources.append(finish)
    resumed = await _drain(_resume(turns, record.thread_id))
    assert resumed[-1].type == "complete"
    assert (
        turns.service.hitl_interrupt_repository.get_by_id(record.id).status
        is HITLInterruptStatus.RESOLVED
    )
    await _second_turn_is_accepted(turns)


async def test_active_row_conflict_preserves_pending_approval(turns):
    thread = await _approval_pause(turns)
    await turns.backend.control.start_generation(
        CreateGeneration(
            conversation_id=turns.backend.conversation_id,
            user_id=turns.backend.user_id,
            logical_turn_id=str(uuid4()),
            checkpoint_thread_id="separate-active-turn",
        )
    )
    events = await _drain(_resume(turns, thread))
    assert events[-1].data["code"] == "conversation_turn_conflict"
    assert (
        turns.service.hitl_interrupt_repository.get_by_id("interrupt-1").status
        is HITLInterruptStatus.PENDING
    )


async def test_stopping_approval_expires_only_its_owned_thread(turns):
    thread = await _approval_pause(turns)
    other_owner = uuid4()
    with turns.backend._session_factory.begin() as session:
        session.add(
            User(
                id=other_owner,
                username=f"other-{other_owner}",
                email=f"{other_owner}@example.test",
                password_hash="test",
            )
        )
    hitl = turns.service.hitl_interrupt_repository
    for key, owner, checkpoint in [
        ("other-thread", turns.backend.user_id, thread + ":other"),
        ("other-owner", other_owner, thread),
    ]:
        hitl.create(
            interrupt_id=key,
            conversation_id=turns.backend.conversation_id,
            user_id=owner,
            thread_id=checkpoint,
            expires_at=datetime.now(UTC) + timedelta(minutes=5),
            action_requests_json=[],
        )
    try:
        paused = await turns.backend.control.find_by_logical_turn(
            logical_turn_id=thread.rsplit(":", 1)[-1],
            user_id=turns.backend.user_id,
            conversation_id=turns.backend.conversation_id,
        )
        await turns.backend.control.request_stop(
            StopGenerationCommand(
                generation_id=paused.generation_id,
                user_id=turns.backend.user_id,
                conversation_id=turns.backend.conversation_id,
                expected_version=paused.version,
                idempotency_key="stop-owned-approval",
            )
        )
        assert hitl.get_by_id("interrupt-1").status is HITLInterruptStatus.EXPIRED
        assert hitl.get_by_id("interrupt-1").resolution_source == "generation_stopped"
        assert hitl.get_by_id("other-thread").status is HITLInterruptStatus.PENDING
        assert hitl.get_by_id("other-owner").status is HITLInterruptStatus.PENDING
    finally:
        with turns.backend._session_factory.begin() as session:
            session.execute(delete(HITLInterrupt).where(HITLInterrupt.user_id == other_owner))
            session.execute(delete(User).where(User.id == other_owner))


def _hold_worker_lock(url, conversation_id, ready, release):
    """A separate process with its own pool and advisory-lock connection."""
    engine = create_engine(url)

    async def hold():
        backend = PostgresAdvisoryLockBackend(sessionmaker(bind=engine))
        if not await backend.acquire(conversation_id, timeout_seconds=5):
            raise RuntimeError("worker failed to acquire lock")
        ready.set()
        try:
            await asyncio.to_thread(release.wait, 10)
        finally:
            await backend.release(conversation_id)

    try:
        asyncio.run(hold())
    finally:
        engine.dispose()


async def test_other_process_busy_lock_leaves_approval_pending(turns):
    thread = await _approval_pause(turns)
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    worker = context.Process(
        target=_hold_worker_lock,
        args=(os.environ["TEST_DATABASE_URL"], str(turns.backend.conversation_id), ready, release),
    )
    worker.start()
    try:
        assert await asyncio.to_thread(ready.wait, 8)
        events = await _drain(_resume(turns, thread))
        assert events[-1].data["code"] == "conversation_turn_conflict"
        assert (
            turns.service.hitl_interrupt_repository.get_by_id("interrupt-1").status
            is HITLInterruptStatus.PENDING
        )
    finally:
        release.set()
        await asyncio.to_thread(worker.join, 8)
        if worker.is_alive():
            worker.terminate()
            worker.join()
    assert worker.exitcode == 0


async def test_resume_checks_decisions_before_reclaiming_row(turns):
    thread = await _approval_pause(turns)
    from app.core.exceptions import CustomHTTPException

    with pytest.raises(CustomHTTPException):
        await _drain(turns.resume(thread))  # Missing decision for call-1.
    row = await turns.backend.control.find_by_logical_turn(
        logical_turn_id=thread.rsplit(":", 1)[-1],
        user_id=turns.backend.user_id,
        conversation_id=turns.backend.conversation_id,
    )
    from app.models.generation import GenerationStatus

    assert row.status is GenerationStatus.CONTINUABLE
    assert (
        turns.service.hitl_interrupt_repository.get_by_id("interrupt-1").status
        is HITLInterruptStatus.PENDING
    )


async def test_continue_approval_keeps_visible_text_and_tool_evidence(turns):
    offer = await _budget_pause(turns)

    async def asks(**kwargs):
        yield make_event("message_delta", sequence=1, data={"text": "Findings before approval."})
        yield make_event(
            "tool_execution_end",
            sequence=2,
            tool_name="web_search",
            tool_call_id="prior-search",
            data={"output": "source evidence"},
        )
        yield _interrupt_event(
            SimpleNamespace(
                conversation_id=str(turns.backend.conversation_id),
                user_message_id=kwargs["thread_id"].rsplit(":", 1)[-1],
            )
        )

    turns.continue_sources.append(asks)
    events = await _drain(turns.continue_(offer))
    paused = next(event for event in events if event.type == "interrupt")
    record = turns.service.hitl_interrupt_repository.get_by_id("interrupt-1")
    with turns.backend._session_factory() as session:
        message = session.get(Message, record.assistant_message_id)
        assert message.content == "Findings before approval."
        assert message.message_metadata["tool_artifacts"][0]["tool"] == "web_search"

    async def finish(**kwargs):
        yield make_event("complete", sequence=1, data={"response": None})

    turns.resume_sources.append(finish)
    resumed = await _drain(_resume(turns, paused.data["thread_id"]))
    assert resumed[-1].type == "complete"
    with turns.backend._session_factory() as session:
        original = session.get(Message, record.assistant_message_id)
        assert original.content == "Findings before approval."


async def test_detached_custom_agent_refuses_resume_and_fails_claim(turns):
    thread = await _approval_pause(turns)
    generation = await turns.backend.control.find_by_logical_turn(
        logical_turn_id=thread.rsplit(":", 1)[-1],
        user_id=turns.backend.user_id,
        conversation_id=turns.backend.conversation_id,
    )
    registry = get_generation_registry()
    custom = f"custom_agent:{uuid4()}"
    registry.get(generation.generation_id).active_agent_id = custom
    turns.service.custom_agent_service = SimpleNamespace(build_runtime_state=lambda *_args: {})
    from app.core.exceptions import CustomHTTPException

    with pytest.raises(CustomHTTPException) as error:
        await _drain(_resume(turns, thread))
    assert error.value.status_code == 409
    assert (
        turns.service.hitl_interrupt_repository.get_by_id("interrupt-1").status
        is HITLInterruptStatus.FAILED
    )
    assert not registry.is_runtime_agent_in_use(turns.backend.user_id, custom)
    await _second_turn_is_accepted(turns)


async def test_record_selected_agent_survives_resume_without_local_token(turns):
    thread = await _approval_pause(turns)
    generation = await turns.backend.control.find_by_logical_turn(
        logical_turn_id=thread.rsplit(":", 1)[-1],
        user_id=turns.backend.user_id,
        conversation_id=turns.backend.conversation_id,
    )
    custom = f"custom_agent:{uuid4()}"
    with turns.backend._session_factory.begin() as session:
        session.execute(
            update(HITLInterrupt)
            .where(HITLInterrupt.id == "interrupt-1")
            .values(interrupt_metadata_json={"active_agent_id": custom})
        )
    registry = get_generation_registry()
    registry.remove(generation.generation_id)
    turns.service.custom_agent_service = SimpleNamespace(
        build_runtime_state=lambda *_args: {custom: {"name": "Attached"}}
    )

    async def asks_again(**kwargs):
        yield _interrupt_event(
            SimpleNamespace(
                conversation_id=str(turns.backend.conversation_id),
                user_message_id=thread.rsplit(":", 1)[-1],
            ),
            interrupt_id="interrupt-next",
        )

    turns.resume_sources.append(asks_again)
    events = await _drain(_resume(turns, thread))
    assert events[-1].type == "interrupt"
    assert registry.is_runtime_agent_in_use(turns.backend.user_id, custom)
    assert registry.get(generation.generation_id).active_agent_id == custom
