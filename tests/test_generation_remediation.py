"""Generation lifecycle regression coverage over an index-enforcing repository."""

import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import UUID, uuid4

import pytest

from app.models.generation import ACTIVE_STATUSES, GenerationStatus
from app.schemas.generation import CreateGeneration, StopGenerationCommand
from app.services.event_streaming.events import make_event
from app.services.generation_registry import get_generation_registry
from tests.integration.test_generation_turn_release_postgres import (
    _approval_pause,
    _Backend,
    _budget_pause,
    _drain,
    _IndexedFakeRepository,
    _interrupt_event,
    _second_turn_is_accepted,
    _Turns,
)


@pytest.fixture
def turns():
    registry = get_generation_registry()
    registry._store.clear()
    value = _Turns(
        _Backend(repository=_IndexedFakeRepository(), user_id=uuid4(), conversation_id=uuid4())
    )
    yield value
    registry._store.clear()


async def test_disconnect_at_continue_run_start_releases_row(turns):
    offer = await _budget_pause(turns)
    stream = turns.continue_(offer)
    assert (await anext(stream)).type == "run_start"
    await stream.aclose()
    assert turns.backend.statuses()[0] not in ACTIVE_STATUSES
    await _second_turn_is_accepted(turns)


async def test_unreadable_research_accounting_releases_row(turns):
    offer = await _budget_pause(turns)
    turns.backend.repository.rows[UUID(offer["generation_id"])]["research_accounting"] = {
        "schema_version": -1
    }
    events = await _drain(turns.continue_(offer))
    assert events[-1].data["error_code"] == "research_accounting_unreadable"
    assert turns.backend.statuses()[0] not in ACTIVE_STATUSES
    await _second_turn_is_accepted(turns)


async def test_approval_during_continue_is_persisted(turns):
    offer = await _budget_pause(turns)
    turns.service.hitl_interrupt_repository = SimpleNamespace(create=Mock())

    async def asks(**kwargs):
        yield _interrupt_event(
            SimpleNamespace(
                conversation_id=str(turns.backend.conversation_id),
                user_message_id=kwargs["thread_id"].rsplit(":", 1)[-1],
            )
        )

    turns.continue_sources.append(asks)
    events = await _drain(turns.continue_(offer))
    assert "interrupt" in [e.type for e in events]
    assert turns.service.hitl_interrupt_repository.create.called


async def test_resume_refuses_if_another_turn_is_active(turns):
    thread_id = await _approval_pause(turns)
    await turns.backend.control.start_generation(
        CreateGeneration(
            conversation_id=turns.backend.conversation_id,
            user_id=turns.backend.user_id,
            logical_turn_id=str(uuid4()),
            checkpoint_thread_id="new-turn",
        )
    )
    calls = []

    async def resumed(**kwargs):
        calls.append(kwargs)
        yield make_event("message_delta", sequence=1, data={"text": "tool executed"})
        yield make_event("complete", sequence=2, data={"response": None})

    turns.resume_sources.append(resumed)
    await _drain(turns.resume(thread_id))
    assert not calls, "approval resume executed alongside another active turn"


async def test_stopped_approval_cannot_execute_later(turns):
    thread_id = await _approval_pause(turns)
    row = next(iter(turns.backend.repository.rows.values()))
    await turns.backend.control.request_stop(
        StopGenerationCommand(
            generation_id=row["id"],
            user_id=turns.backend.user_id,
            conversation_id=turns.backend.conversation_id,
            idempotency_key="stop-approval-probe",
            expected_version=row["version"],
        )
    )
    assert turns.backend.statuses() == [GenerationStatus.COMPLETED_PARTIAL]
    calls = []

    async def resumed(**kwargs):
        calls.append(kwargs)
        yield make_event("complete", sequence=2, data={"response": None})

    turns.resume_sources.append(resumed)
    await _drain(turns.resume(thread_id))
    assert not calls, "a stopped approval still resumed"


async def test_continued_epoch_observes_durable_stop_without_bus_signal(turns):
    offer = await _budget_pause(turns)
    stream = turns.continue_(offer)
    assert (await anext(stream)).type == "run_start"
    row = turns.backend.repository.rows[UUID(offer["generation_id"])]
    turns.backend.control._bus.publish_stop = AsyncMock()
    await turns.backend.control.request_stop(
        StopGenerationCommand(
            generation_id=row["id"],
            user_id=turns.backend.user_id,
            conversation_id=turns.backend.conversation_id,
            idempotency_key="stop-durable-probe",
            expected_version=row["version"],
        )
    )

    async def continues(**kwargs):
        yield make_event("tool_execution_start", sequence=1, data={"tool": "side-effect"})
        yield make_event("message_delta", sequence=2, data={"text": "kept running after Stop"})

    turns.continue_sources.append(continues)
    events = await _drain(stream)
    assert "message_delta" not in [e.type for e in events], "durable Stop was ignored"


async def test_continued_epoch_restamps_producer(turns):
    offer = await _budget_pause(turns)
    row = turns.backend.repository.rows[UUID(offer["generation_id"])]
    row["producer_token"] = "old-worker:123:456"
    stream = turns.continue_(offer)
    assert (await anext(stream)).type == "run_start"
    token_after_lease = row["producer_token"]
    await stream.aclose()
    assert token_after_lease != "old-worker:123:456"


async def test_continued_epoch_disconnect_persists_partial(turns):
    offer = await _budget_pause(turns)
    persist = turns.service._acreate_bot_response_message
    turns.service._acreate_bot_response_message = AsyncMock(wraps=persist)

    async def source(**kwargs):
        yield make_event("message_delta", sequence=1, data={"text": "new partial text"})

    turns.continue_sources.append(source)
    stream = turns.continue_(offer)
    assert (await anext(stream)).type == "run_start"
    assert (await anext(stream)).type == "message_delta"
    await stream.aclose()
    assert turns.service._acreate_bot_response_message.called, (
        "continued partial text was discarded"
    )


@pytest.mark.parametrize("kind", ["continue", "resume"])
async def test_nested_ai_stream_is_closed_at_disconnect(turns, kind):
    closed = []

    async def source(**kwargs):
        try:
            yield make_event("message_delta", sequence=1, data={"text": "saved partial"})
            await asyncio.Event().wait()
        finally:
            closed.append(True)

    if kind == "continue":
        offer = await _budget_pause(turns)
        turns.continue_sources.append(source)
        stream = turns.continue_(offer)
        await anext(stream)
    else:
        thread_id = await _approval_pause(turns)
        turns.resume_sources.append(source)
        stream = turns.resume(thread_id)
    assert (await anext(stream)).type == "message_delta"
    entry = next(
        e
        for e in get_generation_registry().find_by_conversation(turns.backend.conversation_id)
        if not e.paused
    )
    await stream.aclose()
    assert closed == [True]
    assert entry.done.done()
    assert entry.done.result()["content"] == "saved partial"
    assert turns.backend.statuses() == [GenerationStatus.STOPPED]
    await _second_turn_is_accepted(turns)


async def test_resume_observes_durable_stop_and_saves_partial(turns):
    thread_id = await _approval_pause(turns)

    async def source(**kwargs):
        yield make_event("message_delta", sequence=1, data={"text": "resume partial"})
        row = next(iter(turns.backend.repository.rows.values()))
        turns.backend.control._bus.publish_stop = AsyncMock()
        await turns.backend.control.request_stop(
            StopGenerationCommand(
                generation_id=row["id"],
                user_id=turns.backend.user_id,
                conversation_id=turns.backend.conversation_id,
                idempotency_key="durable-resume-stop",
                expected_version=row["version"],
            )
        )
        yield make_event("tool_execution_start", sequence=2, data={})
        yield make_event("message_delta", sequence=3, data={"text": "must not appear"})

    turns.resume_sources.append(source)
    stream = turns.resume(thread_id)
    assert (await anext(stream)).data["text"] == "resume partial"
    entry = next(
        e
        for e in get_generation_registry().find_by_conversation(turns.backend.conversation_id)
        if not e.paused
    )
    events = await _drain(stream)
    assert all(e.type != "message_delta" for e in events)
    assert entry.done.result()["content"] == "resume partial"
    assert turns.backend.statuses() == [GenerationStatus.STOPPED]


async def test_stop_winning_approval_persistence_race_suppresses_pending(turns):
    offer = await _budget_pause(turns)

    async def source(**kwargs):
        yield _interrupt_event(
            SimpleNamespace(
                conversation_id=str(turns.backend.conversation_id),
                user_message_id=kwargs["thread_id"].rsplit(":", 1)[-1],
            )
        )

    turns.continue_sources.append(source)
    persist = turns.service._persist_interrupt_bot_message

    def persist_after_stop(**kwargs):
        # Model Stop committing after durable approval persistence but before pause transition.
        message = persist(**kwargs)
        row = turns.backend.repository.rows[UUID(offer["generation_id"])]
        row["status"] = GenerationStatus.STOP_REQUESTED
        row["version"] += 1
        return message

    turns.service._persist_interrupt_bot_message = persist_after_stop
    events = await _drain(turns.continue_(offer))
    assert "interrupt" not in [e.type for e in events]
    assert turns.backend.statuses() == [GenerationStatus.STOPPED]


async def test_continue_error_publication_is_already_settled(turns):
    offer = await _budget_pause(turns)
    generation_id = UUID(offer["generation_id"])
    turns.backend.repository.rows[generation_id]["research_accounting"] = {"schema_version": -1}
    stream = turns.continue_(offer)
    assert (await anext(stream)).data["error_code"] == "research_accounting_unreadable"
    assert turns.backend.statuses() == [GenerationStatus.FAILED]
    await stream.aclose()
    assert get_generation_registry().get(generation_id) is None
    await _second_turn_is_accepted(turns)


@pytest.mark.parametrize("kind", ["continue", "resume"])
async def test_local_stop_returns_persisted_partial_to_registry_waiter(turns, kind):
    ready = asyncio.Event()

    async def source(**kwargs):
        yield make_event("message_delta", sequence=1, data={"text": "recoverable answer"})
        await asyncio.Event().wait()

    if kind == "continue":
        offer = await _budget_pause(turns)
        turns.continue_sources.append(source)
        stream = turns.continue_(offer)
        generation_id = UUID(offer["generation_id"])
    else:
        thread = await _approval_pause(turns)
        turns.resume_sources.append(source)
        stream = turns.resume(thread)
        generation_id = next(iter(turns.backend.repository.rows))

    async def consume():
        async for event in stream:
            if event.type == "message_delta":
                ready.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(ready.wait(), 3)
    entry = get_generation_registry().get(generation_id)
    row = turns.backend.repository.rows[generation_id]
    snapshot = await turns.service.stop_generation(
        generation_id=generation_id,
        conversation_id=turns.backend.conversation_id,
        user_id=turns.backend.user_id,
        expected_version=row["version"],
        idempotency_key="local-stop-regression",
    )
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert snapshot.status is GenerationStatus.STOPPED
    assert snapshot.assistant_message_id == UUID(entry.done.result()["id"])
    assert entry.done.result()["content"] == "recoverable answer"
    await _second_turn_is_accepted(turns)


@pytest.mark.parametrize("kind", ["continue", "resume"])
async def test_disconnect_releases_row_when_partial_persistence_fails(turns, kind):
    async def source(**kwargs):
        yield make_event("message_delta", sequence=1, data={"text": "unsaved partial"})
        await asyncio.Event().wait()

    if kind == "continue":
        offer = await _budget_pause(turns)
        turns.continue_sources.append(source)
        stream = turns.continue_(offer)
        await anext(stream)
    else:
        thread = await _approval_pause(turns)
        turns.resume_sources.append(source)
        stream = turns.resume(thread)
    await anext(stream)
    turns.service._acreate_bot_response_message = AsyncMock(
        side_effect=RuntimeError("storage unavailable")
    )
    with contextlib.suppress(RuntimeError):
        await stream.aclose()
    assert all(status not in ACTIVE_STATUSES for status in turns.backend.statuses())

    async def persist(**kwargs):
        return turns.service._create_bot_response_message(**kwargs)

    turns.service._acreate_bot_response_message = persist
    await _second_turn_is_accepted(turns)


async def test_resume_error_persistence_failure_still_releases_row(turns):
    thread = await _approval_pause(turns)

    async def source(**kwargs):
        yield make_event("error", sequence=1, data={"error": "provider failed"})

    turns.resume_sources.append(source)
    turns.service._acreate_bot_response_message = AsyncMock(
        side_effect=RuntimeError("storage unavailable")
    )
    turns.service._create_bot_response_message = Mock(
        side_effect=RuntimeError("storage unavailable")
    )
    with contextlib.suppress(RuntimeError):
        await _drain(turns.resume(thread))
    assert all(status not in ACTIVE_STATUSES for status in turns.backend.statuses())


@pytest.mark.parametrize("kind", ["continue", "resume"])
@pytest.mark.parametrize("ending", ["complete", "stop"])
async def test_custom_agent_remains_in_use_at_repeated_approval(turns, kind, ending):
    registry = get_generation_registry()
    custom = "custom:test-agent"
    if kind == "continue":
        offer = await _budget_pause(turns)
        generation_id = UUID(offer["generation_id"])
        registry.get(generation_id).active_agent_id = custom

        def stream_factory():
            return turns.continue_(offer)

        sources = turns.continue_sources
    else:
        thread = await _approval_pause(turns)
        generation_id = next(iter(turns.backend.repository.rows))
        registry.get(generation_id).active_agent_id = custom

        def stream_factory():
            return turns.resume(thread)

        sources = turns.resume_sources

    async def asks_again(**kwargs):
        yield _interrupt_event(
            SimpleNamespace(
                conversation_id=str(turns.backend.conversation_id),
                user_message_id=kwargs["thread_id"].rsplit(":", 1)[-1],
            ),
            interrupt_id="interrupt-2",
        )

    sources.append(asks_again)
    events = await _drain(stream_factory())
    assert events[-1].type == "interrupt"
    assert registry.is_runtime_agent_in_use(turns.backend.user_id, custom)
    entry = registry.get(generation_id)
    assert entry.paused
    assert entry.task is None
    if ending == "stop":
        row = turns.backend.repository.rows[generation_id]
        await turns.service.stop_generation(
            generation_id=generation_id,
            user_id=turns.backend.user_id,
            conversation_id=turns.backend.conversation_id,
            expected_version=row["version"],
            idempotency_key="stop-paused-custom",
        )
    else:

        async def finish(**kwargs):
            yield make_event("complete", sequence=1, data={"response": None})

        turns.resume_sources.append(finish)
        await _drain(turns.resume(events[-1].data["thread_id"], interrupt_id="interrupt-2"))
    assert not registry.is_runtime_agent_in_use(turns.backend.user_id, custom)


async def test_detached_custom_agent_cannot_resume_after_registry_replacement(turns):
    thread = await _approval_pause(turns)
    generation_id = next(iter(turns.backend.repository.rows))
    custom = f"custom_agent:{uuid4()}"
    registry = get_generation_registry()
    registry.get(generation_id).active_agent_id = custom
    turns.service.custom_agent_service = SimpleNamespace(build_runtime_state=lambda *_args: {})

    async def must_not_execute(**kwargs):
        pytest.fail("detached custom agent resumed")
        yield

    turns.resume_sources.append(must_not_execute)
    from app.core.exceptions import CustomHTTPException

    with pytest.raises(CustomHTTPException) as error:
        await _drain(turns.resume(thread))
    assert error.value.status_code == 409
    assert turns.backend.statuses() == [GenerationStatus.FAILED]
    assert not registry.is_runtime_agent_in_use(turns.backend.user_id, custom)


@pytest.mark.parametrize("ending", ["complete", "error", "disconnect", "budget_pause"])
async def test_resuming_one_approval_keeps_another_turn_gate(turns, ending):
    thread_a = await _approval_pause(turns)
    id_a = next(iter(turns.backend.repository.rows))
    thread_b = await _approval_pause(turns)
    id_b = next(key for key in turns.backend.repository.rows if key != id_a)
    registry = get_generation_registry()
    agent_a, agent_b = f"custom_agent:{uuid4()}", f"custom_agent:{uuid4()}"
    registry.get(id_a).active_agent_id = agent_a
    pending_b = registry.get(id_b)
    pending_b.active_agent_id = agent_b

    async def source(**kwargs):
        if ending == "complete":
            yield make_event("complete", sequence=1, data={"response": None})
        elif ending == "error":
            yield make_event("error", sequence=1, data={"error": "provider failed"})
        elif ending == "disconnect":
            yield make_event("message_delta", sequence=1, data={"text": "partial"})
            await asyncio.Event().wait()
        else:
            from tests.integration.test_generation_turn_release_postgres import _pause_event

            yield _pause_event(logical_turn_id=thread_a.rsplit(":", 1)[-1])

    turns.resume_sources.append(source)
    stream = turns.resume(thread_a)
    if ending == "disconnect":
        assert (await anext(stream)).type == "message_delta"
        await stream.aclose()
    else:
        await _drain(stream)
    assert registry.get(id_b) is pending_b
    assert registry.is_runtime_agent_in_use(turns.backend.user_id, agent_b)
    assert turns.backend.repository.rows[id_b]["status"] is GenerationStatus.CONTINUABLE

    async def finish_b(**kwargs):
        yield make_event("complete", sequence=1, data={"response": None})

    turns.resume_sources.append(finish_b)
    assert (await _drain(turns.resume(thread_b)))[-1].type == "complete"
    assert not registry.is_runtime_agent_in_use(turns.backend.user_id, agent_b)


async def test_unrelated_detached_agent_does_not_refuse_current_approval(turns):
    thread_a = await _approval_pause(turns)
    id_a = next(iter(turns.backend.repository.rows))
    await _approval_pause(turns)
    id_b = next(key for key in turns.backend.repository.rows if key != id_a)
    registry = get_generation_registry()
    agent_a, agent_b = f"custom_agent:{uuid4()}", f"custom_agent:{uuid4()}"
    registry.get(id_a).active_agent_id = agent_a
    registry.get(id_b).active_agent_id = agent_b
    turns.service.custom_agent_service = SimpleNamespace(
        build_runtime_state=lambda *_args: {agent_a: {"name": "Attached"}}
    )

    async def finish(**kwargs):
        yield make_event("complete", sequence=1, data={"response": None})

    turns.resume_sources.append(finish)
    events = await _drain(turns.resume(thread_a))
    assert events[-1].type == "complete"
    assert registry.is_runtime_agent_in_use(turns.backend.user_id, agent_b)


@pytest.mark.parametrize("scope", ["legacy_turn", "unknown_legacy_thread", "other_owner"])
async def test_legacy_resume_cleanup_never_clears_unrelated_tokens(turns, scope):
    from app.ai.workflow.state import build_checkpoint_thread_id

    registry = get_generation_registry()
    id_a, id_b = uuid4(), uuid4()
    owner = uuid4() if scope == "other_owner" else turns.backend.user_id
    entry_a = registry.register(
        id_a, turns.backend.conversation_id, owner, active_agent_id="custom:a", paused=True
    )
    entry_b = registry.register(
        id_b,
        turns.backend.conversation_id,
        turns.backend.user_id,
        active_agent_id="custom:b",
        paused=True,
    )
    thread = (
        str(turns.backend.conversation_id)
        if scope == "unknown_legacy_thread"
        else build_checkpoint_thread_id(str(turns.backend.conversation_id), str(id_a))
    )
    turns.service.generation_control_service = None

    async def finish(**kwargs):
        yield make_event("complete", sequence=1, data={"response": None})

    turns.resume_sources.append(finish)
    assert (await _drain(turns.resume(thread)))[-1].type == "complete"
    assert registry.get(id_b) is entry_b
    assert registry.get(id_a) is (None if scope == "legacy_turn" else entry_a)
