"""How a Stop reaches a worker that is not the one holding the request.

Both halves of this were missing until 2026-09-08, and the gap was invisible
because the same-worker path always worked:

* **Nothing subscribed to the bus.** ``publish_stop`` broadcast into a void, so
  a Stop landing on a worker other than the streaming one transitioned the row
  and interrupted nothing.
* **The streaming worker never read the row.** It checked only its in-process
  cancel event — which only a Stop that landed on *this* worker can set.

So the two mechanisms that make Stop distributed are the subscriber (fast, best
effort) and the worker's own status read (slower, authoritative). The row is the
authority: it cannot un-stop, so the read only has to happen eventually.

The bus is broadcast rather than addressed. Every worker receives every signal
and asks its own registry whether the id is one of its own; addressing would
need a worker registry the lifecycle deliberately does not keep.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.models.generation import GenerationStatus
from app.services.generation_control_bus import (
    InMemoryGenerationControlBus,
    StopSignal,
)
from app.services.generation_registry import get_generation_registry
from app.services.generation_stop_subscriber import (
    handle_stop_signal,
    install_generation_stop_subscriber,
)
from app.services.message_service import _DURABLE_STOP_CHECKPOINTS, _DurableStopWatch

pytestmark = pytest.mark.asyncio

CONVERSATION_ID = uuid4()
USER_ID = uuid4()


@pytest.fixture(autouse=True)
def _clean_registry():
    registry = get_generation_registry()
    for key in list(registry._store):  # noqa: SLF001 - test isolation
        registry._store.pop(key, None)  # noqa: SLF001
    yield
    for key in list(registry._store):  # noqa: SLF001
        registry._store.pop(key, None)  # noqa: SLF001


# ----------------------------------------------------------------------
# the subscriber: a signal that reaches a process does something
# ----------------------------------------------------------------------


async def test_a_stop_signal_cancels_a_generation_this_worker_owns():
    generation_id = uuid4()
    entry = get_generation_registry().register(
        generation_id=generation_id, conversation_id=CONVERSATION_ID, user_id=USER_ID
    )

    applied = await handle_stop_signal(StopSignal(generation_id=generation_id, version=2))

    assert applied is True
    assert entry.is_cancelled


async def test_a_signal_for_another_workers_generation_is_simply_ignored():
    """The ordinary case for every worker but one, and not a failure."""
    mine = get_generation_registry().register(
        generation_id=uuid4(), conversation_id=CONVERSATION_ID, user_id=USER_ID
    )

    applied = await handle_stop_signal(StopSignal(generation_id=uuid4(), version=2))

    assert applied is False
    assert not mine.is_cancelled


async def test_the_signal_leaves_the_registry_entry_in_place():
    """A retried Stop must still have something to cancel.

    Removing the entry on the first signal is what left a retry with nothing to
    interrupt while the turn was still running.
    """
    generation_id = uuid4()
    get_generation_registry().register(
        generation_id=generation_id, conversation_id=CONVERSATION_ID, user_id=USER_ID
    )

    await handle_stop_signal(StopSignal(generation_id=generation_id, version=2))

    assert get_generation_registry().get(generation_id) is not None


async def test_a_failing_registry_does_not_kill_the_subscriber_loop(monkeypatch):
    """The loop is how every future Stop arrives.

    One exception escaping it would disable cancellation for the whole process
    until restart.
    """
    from app.services import generation_stop_subscriber as module

    def explode():
        raise RuntimeError("the registry is on fire")

    monkeypatch.setattr(module, "get_generation_registry", explode)

    assert await handle_stop_signal(StopSignal(generation_id=uuid4(), version=1)) is False


async def test_the_subscriber_is_installed_onto_the_bus():
    """The gap this file exists for: publishing with nobody listening."""
    bus = InMemoryGenerationControlBus()
    generation_id = uuid4()
    entry = get_generation_registry().register(
        generation_id=generation_id, conversation_id=CONVERSATION_ID, user_id=USER_ID
    )

    assert await install_generation_stop_subscriber(bus) is True
    await bus.publish_stop(generation_id, 3)

    assert entry.is_cancelled


async def test_an_unavailable_bus_degrades_instead_of_failing_startup():
    """Stop still works through the row; it is only slower."""

    class _Broken:
        async def subscribe(self, handler):
            raise RuntimeError("redis is down")

    assert await install_generation_stop_subscriber(_Broken()) is False
    assert await install_generation_stop_subscriber(None) is False


async def test_the_startup_hook_subscribes_this_worker():
    """Asserted on the lifespan's own source, because it needs a real app to run.

    Without this call the bus has no subscriber in production and the
    distributed half of Stop is inert — which is exactly the state this
    repository was in.

    Scoped to ``lifespan`` deliberately: searching the whole module for the
    function name also matches its ``def`` line, so the assertion passed with
    the call deleted. That is the failure this test is supposed to catch.
    """
    import inspect

    from app import main

    lifespan_source = inspect.getsource(main.lifespan)
    assert "await _subscribe_generation_stop_signals()" in lifespan_source

    assert "install_generation_stop_subscriber" in inspect.getsource(
        main._subscribe_generation_stop_signals
    )


# ----------------------------------------------------------------------
# the durable read: a worker that never receives a signal still stops
# ----------------------------------------------------------------------


def _generation(status: GenerationStatus = GenerationStatus.RUNNING):
    return SimpleNamespace(
        generation_id=uuid4(),
        conversation_id=CONVERSATION_ID,
        version=2,
        status=status,
    )


class _Control:
    """A lifecycle service that reports a status and counts the reads."""

    def __init__(self, status: GenerationStatus) -> None:
        self.status = status
        self.reads = 0

    async def aget_snapshot(self, *, generation_id, user_id, conversation_id):
        self.reads += 1
        return SimpleNamespace(status=self.status)


async def test_a_worker_stops_when_the_row_says_stop_requested():
    """The mechanism that makes a cross-worker Stop work at all."""
    control = _Control(GenerationStatus.STOP_REQUESTED)
    watch = _DurableStopWatch(control=control, generation=_generation(), user_id=USER_ID)

    assert await watch.stop_requested("tool_execution_end") is True


async def test_a_running_turn_is_not_stopped():
    control = _Control(GenerationStatus.RUNNING)
    watch = _DurableStopWatch(control=control, generation=_generation(), user_id=USER_ID)

    assert await watch.stop_requested("tool_execution_end") is False


async def test_a_token_delta_never_triggers_a_database_read():
    """A read per token would put a query on the hot path of every answer."""
    control = _Control(GenerationStatus.STOP_REQUESTED)
    watch = _DurableStopWatch(control=control, generation=_generation(), user_id=USER_ID)

    for _ in range(50):
        assert await watch.stop_requested("message_delta") is False

    assert control.reads == 0


@pytest.mark.parametrize("event_type", sorted(_DURABLE_STOP_CHECKPOINTS))
async def test_every_declared_checkpoint_actually_checks(event_type):
    """A checkpoint in the set that did not read would be a silent hole."""
    control = _Control(GenerationStatus.STOP_REQUESTED)
    watch = _DurableStopWatch(control=control, generation=_generation(), user_id=USER_ID)

    assert await watch.stop_requested(event_type) is True
    assert control.reads == 1


async def test_the_read_is_throttled_between_boundaries(monkeypatch):
    """Correctness needs the check eventually, not immediately.

    The row cannot un-stop, so a minimum interval costs a little latency and
    removes a query from every tool call in a long turn.
    """
    control = _Control(GenerationStatus.RUNNING)
    watch = _DurableStopWatch(control=control, generation=_generation(), user_id=USER_ID)
    monkeypatch.setattr(type(watch), "_interval", property(lambda self: 999.0))

    for _ in range(10):
        await watch.stop_requested("tool_execution_end")

    assert control.reads == 1


async def test_a_zero_interval_checks_at_every_boundary(monkeypatch):
    control = _Control(GenerationStatus.RUNNING)
    watch = _DurableStopWatch(control=control, generation=_generation(), user_id=USER_ID)
    monkeypatch.setattr(type(watch), "_interval", property(lambda self: 0.0))

    for _ in range(5):
        await watch.stop_requested("tool_execution_end")

    assert control.reads == 5


async def test_a_failed_read_never_stops_a_turn():
    """A stop that cannot be confirmed must not end a good answer."""

    class _Failing:
        async def aget_snapshot(self, **_kwargs):
            raise RuntimeError("the database is unreachable")

    watch = _DurableStopWatch(control=_Failing(), generation=_generation(), user_id=USER_ID)

    assert await watch.stop_requested("tool_execution_end") is False


async def test_a_missing_row_never_stops_a_turn():
    class _Absent:
        async def aget_snapshot(self, **_kwargs):
            return None

    watch = _DurableStopWatch(control=_Absent(), generation=_generation(), user_id=USER_ID)

    assert await watch.stop_requested("tool_execution_end") is False


async def test_a_deployment_without_the_lifecycle_never_reads():
    """No lifecycle service means no durable status to consult."""
    watch = _DurableStopWatch(control=None, generation=_generation(), user_id=USER_ID)

    assert await watch.stop_requested("tool_execution_end") is False


async def test_the_watch_settles_and_stops_reading():
    """Once the answer is yes it cannot become no, so asking again is waste."""
    control = _Control(GenerationStatus.STOP_REQUESTED)
    watch = _DurableStopWatch(control=control, generation=_generation(), user_id=USER_ID)

    assert await watch.stop_requested("tool_execution_end") is True
    assert await watch.stop_requested("tool_execution_end") is False
    assert control.reads == 1


# ----------------------------------------------------------------------
# the wiring: the real stream loop consults the watch
# ----------------------------------------------------------------------
#
# The unit tests above all pass with the call removed from the stream loop, so
# on their own they would let exactly the regression this file documents come
# back. These drive the real `create_message_stream`.


def _streaming_service(control, events):
    """A real ``MessageService`` with only persistence and validation stubbed."""
    from app.models.enums import MessageRole
    from app.schemas.message import MessageCreate
    from app.schemas.workflow import WorkflowExecutionRequest, WorkflowPlanningContext
    from app.services.message_service import MessageService

    service = MessageService.__new__(MessageService)
    service.generation_control_service = control
    service._turn_coordinator = None
    service.custom_agent_service = None
    service.chat_image_service = None
    service.web_image_service = None
    service.tool_approval_setting_repository = None
    service.persisted: list[dict] = []

    from datetime import datetime, timezone

    message_id = uuid4()

    def _created(_entity):
        now = datetime.now(timezone.utc)
        return SimpleNamespace(
            id=message_id,
            conversation_id=CONVERSATION_ID,
            sender=MessageRole.user.value,
            content="hello",
            message_metadata={},
            feedback=None,
            created_at=now,
            updated_at=now,
            deleted_at=None,
        )

    async def _acreate(entity):
        return _created(entity)

    service.repository = SimpleNamespace(create=_created, acreate=_acreate)

    async def _avalidate(*_args, **_kwargs):
        return None

    async def _aget_by_id(_cid):
        # A non-default title, so no title task is started and the turn's
        # terminal path is the only thing under test.
        return SimpleNamespace(
            title="Existing chat",
            persona_prompt=None,
            planning_mode_enabled=False,
            owner_id=USER_ID,
        )

    service.conversation_validation_utils = SimpleNamespace(
        validate_conversation_access=lambda *_a: None,
        avalidate_conversation_access=_avalidate,
        conversation_repository=SimpleNamespace(aget_by_id=_aget_by_id),
    )

    workflow_request = WorkflowExecutionRequest(
        message="hello",
        conversation_id=str(CONVERSATION_ID),
        user_id=str(USER_ID),
        planning=WorkflowPlanningContext(),
    )

    async def _build_request(**_kwargs):
        return (USER_ID, None, workflow_request)

    service._build_user_message_workflow_request = _build_request

    async def _stream(_request):
        for event in events:
            yield event

    service.ai_service = SimpleNamespace(
        invalidate_history_cache=lambda *_a: None,
        execute_request_stream=_stream,
    )

    def _persisted_message(content: str):
        """A persisted row with the attributes the terminal event reads.

        ``id`` is load-bearing: the completion event does ``str(bot_message.id)``
        and a double without it raises into the broad handler, which then
        swallows the whole turn into an error message and the test sees no
        `complete` at all.
        """
        row = {"id": str(uuid4()), "content": content}
        service.persisted.append(row)
        return SimpleNamespace(id=row["id"], model_dump=lambda mode="python": row)

    async def _persist(**_kwargs):
        return _persisted_message("done")

    service._persist_completed_workflow_response = _persist

    async def _noop(**_kwargs):
        return None

    service._compact_checkpoint_after_persist = _noop

    async def _acreate_bot(**kwargs):
        return _persisted_message(kwargs.get("content", ""))

    service._acreate_bot_response_message = _acreate_bot

    return service, MessageCreate(
        conversation_id=CONVERSATION_ID, content="hello", role=MessageRole.user
    )


async def test_the_stream_loop_stops_when_another_worker_asked_it_to():
    """The wiring, end to end through the real loop.

    Nothing in this process signals the registry: the only evidence the turn
    should stop is the row, which is exactly what a Stop landing on another
    worker changes.
    """
    from app.schemas.generation import CreateGeneration
    from app.services.event_streaming.events import make_event
    from tests.generation_control_support import build_control_service

    control = build_control_service()
    started = await control.start_generation(
        CreateGeneration(
            conversation_id=CONVERSATION_ID,
            user_id=USER_ID,
            logical_turn_id=f"turn-{uuid4()}",
            checkpoint_thread_id="wf2:conv:turn-1",
            active_agent_id="chat_agent",
        )
    )
    running = await control.mark_running(
        generation_id=started.generation_id,
        user_id=USER_ID,
        conversation_id=CONVERSATION_ID,
        expected_version=started.version,
    )
    # Another worker's Stop: the row moves, this process is told nothing.
    control._test_repository.force(
        running.generation_id, status=GenerationStatus.STOP_REQUESTED
    )

    events = [
        make_event("agent_selected", sequence=1, agent="chat_agent", data={"agent": "chat_agent"}),
        make_event("message_delta", sequence=2, data={"text": "this should not finish"}),
        make_event("tool_execution_end", sequence=3, tool_name="web_search", data={}),
        make_event("complete", sequence=4, data={"response": None}),
    ]
    service, message_create = _streaming_service(control, events)
    # The row was created by this test, so the service must not create another.
    service._astart_generation = lambda **_kwargs: _already(running)

    seen = [
        event.type
        async for event in service._create_message_stream_holding_turn(
            message_create, USER_ID, uuid4()
        )
    ]

    assert "complete" not in seen, "the turn ran to completion despite a durable stop"


async def _already(snapshot):
    return snapshot


async def test_the_stream_loop_finishes_normally_when_nothing_stopped_it():
    """The guard must not end turns nobody asked to stop."""
    from app.schemas.generation import CreateGeneration
    from app.services.event_streaming.events import make_event
    from tests.generation_control_support import build_control_service

    control = build_control_service()
    started = await control.start_generation(
        CreateGeneration(
            conversation_id=CONVERSATION_ID,
            user_id=USER_ID,
            logical_turn_id=f"turn-{uuid4()}",
            checkpoint_thread_id="wf2:conv:turn-2",
            active_agent_id="chat_agent",
        )
    )
    running = await control.mark_running(
        generation_id=started.generation_id,
        user_id=USER_ID,
        conversation_id=CONVERSATION_ID,
        expected_version=started.version,
    )

    events = [
        make_event("agent_selected", sequence=1, agent="chat_agent", data={"agent": "chat_agent"}),
        make_event("tool_execution_end", sequence=2, tool_name="web_search", data={}),
        make_event("complete", sequence=3, data={"response": None}),
    ]
    service, message_create = _streaming_service(control, events)
    service._astart_generation = lambda **_kwargs: _already(running)

    seen = [
        event.type
        async for event in service._create_message_stream_holding_turn(
            message_create, USER_ID, uuid4()
        )
    ]

    assert "complete" in seen
