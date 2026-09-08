"""MessageService owning start, pause, Continue and Stop.

These drive the **real** :class:`GenerationControlService` over the faithful
repository double, so what is asserted is genuine lifecycle legality rather
than a fake agreeing with itself. Two orderings carry most of the value, and
both are the kind that look fine until they are wrong in production:

* **The lifecycle row exists before the first streamed event.** A Stop that
  arrives on the very first token has to find something durable to transition.
  Allocating after the first event leaves a window in which Stop answers "not
  in flight" for a turn that is demonstrably running.
* **The validated partial is persisted before Continue is offered.** The
  continuation id a client redeems must point at an answer that is already
  saved, or a Continue resumes work whose first half was never written down.

``continuable`` is also only legal from ``running``, which is why a turn that
never leaves ``starting`` cannot be offered a Continue at all — asserted here
rather than discovered when a paused turn silently refuses to continue.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest

from app.models.generation import GenerationStatus
from app.schemas.generation import CreateGeneration, MarkContinuable
from app.services.event_streaming.events import make_event
from app.services.generation_registry import get_generation_registry
from app.services.message_service import MessageService
from tests.generation_control_support import (
    CONVERSATION_ID,
    USER_ID,
    build_control_service,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _clean_registry():
    registry = get_generation_registry()
    for key in list(registry._store):  # noqa: SLF001 - test isolation
        registry._store.pop(key, None)  # noqa: SLF001
    yield
    for key in list(registry._store):  # noqa: SLF001
        registry._store.pop(key, None)  # noqa: SLF001


class _Message:
    """The shape ``MessageRead.model_dump`` consumers touch."""

    def __init__(self, message_id: UUID, content: str, metadata: dict[str, Any]) -> None:
        self.id = message_id
        self.content = content
        self.metadata = metadata

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        return {"id": str(self.id), "content": self.content, "metadata": self.metadata}


def _service(control, *, persist_fails: bool = False) -> MessageService:
    """A service with only the collaborators these paths touch.

    ``__new__`` rather than the constructor: the real one wants a dozen
    repositories and validators that none of this exercises.
    """
    service = MessageService.__new__(MessageService)
    service.generation_control_service = control
    service._turn_coordinator = None
    service.persisted: list[_Message] = []

    async def avalidate(user_id, conversation_id):
        return None

    service.conversation_validation_utils = SimpleNamespace(
        avalidate_conversation_access=avalidate,
        validate_conversation_access=lambda user_id, conversation_id: None,
    )

    async def acreate_bot_response_message(*, conversation_id, content, metadata, message_id=None):
        if persist_fails:
            raise RuntimeError("disk on fire")
        message = _Message(message_id or uuid4(), content, dict(metadata))
        service.persisted.append(message)
        return message

    service._acreate_bot_response_message = acreate_bot_response_message
    return service


async def _started(control):
    return await control.start_generation(
        CreateGeneration(
            conversation_id=CONVERSATION_ID,
            user_id=USER_ID,
            logical_turn_id="turn-1",
            checkpoint_thread_id="wf2:conv:turn-1",
            active_agent_id="chat_agent",
        )
    )


def _pause_event(*, epoch: int = 0, content: str = "Partial findings so far."):
    return make_event(
        "continuation_available",
        sequence=1,
        data={
            "type": "execution_budget_exhausted",
            "generation_id": "unused-by-the-service",
            "logical_turn_id": "turn-1",
            "execution_epoch": epoch,
            "active_agent_id": "search_agent",
            "validated_content": content,
            "budget": {"model_calls": 7, "tool_calls": 12, "exhausted_by": "tool_calls"},
        },
    )


async def _publish_pause(service, control, generation, *, event=None, bot_message_id=None):
    inflight = SimpleNamespace(active_agent_id="search_agent", partial_text="", touch=lambda: None)
    sequence = iter(range(1, 100))
    return [
        item
        async for item in service._apublish_continuation_pause(
            event or _pause_event(),
            generation=generation,
            conversation_id=CONVERSATION_ID,
            user_id=USER_ID,
            bot_message_id=bot_message_id or uuid4(),
            sanitized_persona=None,
            workflow_request=None,
            inflight=inflight,
            tool_artifacts=None,
            next_sequence=lambda: next(sequence),
        )
    ]


# ----------------------------------------------------------------------
# identity allocation
# ----------------------------------------------------------------------


async def test_the_lifecycle_row_is_allocated_before_the_turn_streams():
    control = build_control_service()
    service = _service(control)

    snapshot = await service._astart_generation(
        conversation_id=CONVERSATION_ID, user_id=USER_ID, logical_turn_id=uuid4()
    )

    assert snapshot is not None
    assert snapshot.status is GenerationStatus.STARTING
    assert snapshot.execution_epoch == 0
    assert snapshot.version == 1


async def test_the_logical_turn_is_the_user_message_id():
    """One identity for the turn, so a Continue can find its exact thread.

    The checkpoint thread's turn segment is this same id, which is what lets
    the resume context be read from the lifecycle row alone.
    """
    control = build_control_service()
    service = _service(control)
    user_message_id = uuid4()

    snapshot = await service._astart_generation(
        conversation_id=CONVERSATION_ID, user_id=USER_ID, logical_turn_id=user_message_id
    )

    assert snapshot.logical_turn_id == str(user_message_id)
    row = control._test_repository.rows[snapshot.generation_id]
    assert row["checkpoint_thread_id"].endswith(f":{user_message_id}")


async def test_a_conflicting_active_turn_is_a_retriable_typed_error():
    """The partial unique index doing its job, reported as such."""
    from sqlalchemy.exc import IntegrityError

    from app.ai.workflow.contracts import WorkflowRoutingException

    class _Conflicting:
        async def start_generation(self, command):
            raise IntegrityError("insert", {}, Exception("duplicate key"))

    service = _service(_Conflicting())

    with pytest.raises(WorkflowRoutingException) as caught:
        await service._astart_generation(
            conversation_id=CONVERSATION_ID, user_id=USER_ID, logical_turn_id=uuid4()
        )

    assert caught.value.error.code == "conversation_turn_conflict"
    assert caught.value.error.retriable is True


async def test_an_unwired_control_service_allocates_nothing_and_does_not_raise():
    """A deployment without the lifecycle keeps working, without pretending."""
    service = _service(None)

    assert (
        await service._astart_generation(
            conversation_id=CONVERSATION_ID, user_id=USER_ID, logical_turn_id=uuid4()
        )
        is None
    )


# ----------------------------------------------------------------------
# the status projection every transport publishes
# ----------------------------------------------------------------------


async def test_the_status_projection_carries_the_fence_a_command_needs():
    """R5. Without the version, a client cannot issue a fenced Stop at all."""
    control = build_control_service()
    service = _service(control)
    snapshot = await _started(control)

    data = service._generation_status_data(snapshot)

    assert data["generation_id"] == str(snapshot.generation_id)
    assert data["version"] == snapshot.version
    assert data["execution_epoch"] == 0
    assert data["status"] == "starting"


async def test_the_status_projection_publishes_no_checkpoint_internals():
    """A checkpoint thread is a resume handle, not a field a client may read."""
    control = build_control_service()
    service = _service(control)
    snapshot = await _started(control)

    data = service._generation_status_data(snapshot)

    assert "checkpoint_thread_id" not in data
    assert "research_accounting" not in data
    assert "execution_budget" not in data
    assert "user_id" not in data


# ----------------------------------------------------------------------
# running, and why it is not cosmetic
# ----------------------------------------------------------------------


async def test_a_turn_enters_running_before_the_graph_is_entered():
    control = build_control_service()
    service = _service(control)
    started = await _started(control)

    running = await service._amark_generation_running(started, user_id=USER_ID)

    assert running.status is GenerationStatus.RUNNING
    assert running.version == started.version + 1


async def test_a_turn_still_in_starting_cannot_be_offered_a_continue():
    """Why the running transition exists at all.

    ``continuable`` is legal only from an active status that is not
    ``starting``. A turn that paused at its budget without that transition
    could not be offered a Continue, and the failure would look like the pause
    itself being broken.
    """
    from app.services.generation_control_service import IllegalTransition

    control = build_control_service()
    service = _service(control)
    started = await _started(control)

    with pytest.raises(IllegalTransition):
        await control.mark_continuable(
            MarkContinuable(
                generation_id=started.generation_id,
                conversation_id=CONVERSATION_ID,
                user_id=USER_ID,
                expected_version=started.version,
                assistant_message_id=uuid4(),
            )
        )

    running = await service._amark_generation_running(started, user_id=USER_ID)
    offered = await service._amark_generation_continuable(
        running, user_id=USER_ID, assistant_message_id=uuid4()
    )
    assert offered.status is GenerationStatus.CONTINUABLE


async def test_a_failed_running_transition_returns_a_usable_snapshot():
    """Every later transition fences on what this returns.

    Handing back ``None`` (or a stale snapshot from before a Stop won the race)
    would make the completion transition fail too, and the row would be left
    active with nothing running.
    """
    from app.schemas.generation import StopGenerationCommand

    control = build_control_service()
    service = _service(control)
    started = await _started(control)
    # A Stop got there first, so `starting -> running` is no longer legal.
    await control.request_stop(
        StopGenerationCommand(
            generation_id=started.generation_id,
            conversation_id=CONVERSATION_ID,
            user_id=USER_ID,
            idempotency_key="stop-key-0001",
            expected_version=started.version,
        )
    )

    result = await service._amark_generation_running(started, user_id=USER_ID)

    assert result is not None


# ----------------------------------------------------------------------
# the pause: persist first, offer second
# ----------------------------------------------------------------------


async def test_the_partial_is_persisted_before_the_continuation_is_offered():
    control = build_control_service()
    service = _service(control)
    running = await service._amark_generation_running(await _started(control), user_id=USER_ID)

    events = await _publish_pause(service, control, running)

    assert service.persisted, "the validated partial was never written"
    types = [event.type for event in events]
    assert types == ["message_end", "continuation_available"]


async def test_the_persisted_partial_is_marked_partial_and_continuable():
    control = build_control_service()
    service = _service(control)
    running = await service._amark_generation_running(await _started(control), user_id=USER_ID)

    await _publish_pause(service, control, running)

    metadata = service.persisted[0].metadata
    assert metadata["partial"] is True
    assert metadata["continuable"] is True
    assert metadata["stop_reason"] == "execution_budget_exhausted"
    assert metadata["execution_epoch"] == 0
    assert metadata["execution_budget"]["exhausted_by"] == "tool_calls"


async def test_the_offered_continuation_carries_a_redeemable_id():
    control = build_control_service()
    service = _service(control)
    running = await service._amark_generation_running(await _started(control), user_id=USER_ID)

    events = await _publish_pause(service, control, running)

    offer = events[-1].data
    assert offer["status"] == "continuable"
    assert offer["continuation_available"] is True
    assert offer["continuation_id"], "a continuation was offered with no id to redeem"
    assert offer["assistant_message_id"] == str(service.persisted[0].id)


async def test_a_partial_that_cannot_be_saved_offers_nothing_and_fails_the_turn():
    """A partial nobody can read is not an answer.

    Offering to continue it would be offering to continue nothing, and emitting
    the text without persisting it would show a user an answer that vanishes on
    reload.
    """
    control = build_control_service()
    service = _service(control, persist_fails=True)
    running = await service._amark_generation_running(await _started(control), user_id=USER_ID)

    events = await _publish_pause(service, control, running)

    assert [event.type for event in events] == ["error"]
    assert events[0].data["code"] == "response_persistence_failed"
    assert control._test_repository.status_of(running.generation_id) is GenerationStatus.FAILED


async def test_a_pause_that_cannot_be_recorded_still_publishes_the_answer():
    """The answer is saved; it simply cannot be continued.

    Saying nothing about it would hide a persisted message from the client,
    while advertising the offer would hand out a continuation id that no
    Continue could ever redeem.
    """
    control = build_control_service()
    service = _service(control)
    running = await service._amark_generation_running(await _started(control), user_id=USER_ID)

    async def refuse(command):
        raise RuntimeError("the lifecycle write failed")

    control.mark_continuable = refuse

    events = await _publish_pause(service, control, running)

    assert [event.type for event in events] == ["message_end"]
    assert service.persisted, "the answer was discarded along with the offer"


async def test_an_undecidable_mutation_blocks_the_continuation():
    """Resuming could perform the side effect a second time.

    The partial answer is still persisted and published — the user should see
    what happened — but no continuation id is minted, so there is nothing for a
    Continue to redeem.
    """
    control = build_control_service()
    service = _service(control)
    running = await service._amark_generation_running(await _started(control), user_id=USER_ID)
    event = _pause_event()
    event.data["mutation_outcome_unknown"] = True

    events = await _publish_pause(service, control, running, event=event)

    assert service.persisted, "the answer was withheld along with the offer"
    offer = events[-1].data
    assert offer["continuation_available"] is False
    assert offer["continuation_id"] is None
    assert offer["continuation_block_reason"] == "mutation_outcome_unknown"


async def test_a_decidable_turn_is_not_blocked():
    """The guard must not fire on every pause."""
    control = build_control_service()
    service = _service(control)
    running = await service._amark_generation_running(await _started(control), user_id=USER_ID)

    events = await _publish_pause(service, control, running)

    assert events[-1].data["continuation_available"] is True
    assert events[-1].data["continuation_block_reason"] is None


async def test_a_blocked_continuation_cannot_be_redeemed():
    """The block is enforced, not merely reported."""
    from app.schemas.generation import ContinueGenerationCommand
    from app.services.generation_control_service import ContinuationUnavailable

    control = build_control_service()
    service = _service(control)
    running = await service._amark_generation_running(await _started(control), user_id=USER_ID)
    event = _pause_event()
    event.data["mutation_outcome_unknown"] = True
    await _publish_pause(service, control, running)  # a normal pause first
    blocked = await control.find_by_logical_turn(
        logical_turn_id="turn-1", user_id=USER_ID, conversation_id=CONVERSATION_ID
    )

    with pytest.raises(ContinuationUnavailable):
        await control.prepare_continue(
            ContinueGenerationCommand(
                generation_id=blocked.generation_id,
                continuation_id=uuid4(),
                conversation_id=CONVERSATION_ID,
                user_id=USER_ID,
                idempotency_key="continue-key-0001",
                expected_version=blocked.version,
            )
        )


# ----------------------------------------------------------------------
# terminal transitions
# ----------------------------------------------------------------------


async def test_a_turn_that_finishes_on_its_own_is_completed_not_partial():
    control = build_control_service()
    service = _service(control)
    running = await service._amark_generation_running(await _started(control), user_id=USER_ID)
    assistant_message_id = uuid4()

    await service._amark_generation_completed(
        running,
        user_id=USER_ID,
        assistant_message_id=assistant_message_id,
        terminal_reason="completed",
    )

    row = control._test_repository.rows[running.generation_id]
    assert row["status"] is GenerationStatus.COMPLETED
    assert row["assistant_message_id"] == assistant_message_id
    assert row["continuation_available"] is False


async def test_a_bookkeeping_failure_never_discards_a_good_answer():
    """The answer is already persisted and streamed by the time this runs."""
    control = build_control_service()
    service = _service(control)
    running = await service._amark_generation_running(await _started(control), user_id=USER_ID)

    async def refuse(command):
        raise RuntimeError("the lifecycle write failed")

    control.mark_completed = refuse

    assert (
        await service._amark_generation_completed(
            running, user_id=USER_ID, assistant_message_id=uuid4()
        )
        is None
    )


async def test_only_the_worker_turns_a_stop_request_into_a_stop():
    """``stop_requested`` is a client's ask; ``stopped`` is the worker's answer."""
    from app.schemas.generation import StopGenerationCommand

    control = build_control_service()
    service = _service(control)
    running = await service._amark_generation_running(await _started(control), user_id=USER_ID)

    requested = await control.request_stop(
        StopGenerationCommand(
            generation_id=running.generation_id,
            conversation_id=CONVERSATION_ID,
            user_id=USER_ID,
            idempotency_key="stop-key-0001",
            expected_version=running.version,
        )
    )
    assert requested.status is GenerationStatus.STOP_REQUESTED

    await service._amark_generation_stopped(
        requested,
        user_id=USER_ID,
        assistant_message_id=uuid4(),
        terminal_reason="user_requested",
    )

    assert control._test_repository.status_of(running.generation_id) is GenerationStatus.STOPPED


# ----------------------------------------------------------------------
# Continue
# ----------------------------------------------------------------------


async def test_continue_resumes_the_paused_epoch_not_the_leased_one():
    """The off-by-one that would make every Continue look expired.

    ``prepare_continue`` advances the *row* to the next epoch; the checkpoint
    is still in the previous one and the pause node fences against its own
    state. Sending the leased epoch is refused as stale, and that refusal is
    indistinguishable from a genuinely expired continuation.
    """
    control = build_control_service()
    service = _service(control)
    running = await service._amark_generation_running(await _started(control), user_id=USER_ID)
    offered = await service._amark_generation_continuable(
        running, user_id=USER_ID, assistant_message_id=uuid4()
    )

    resumed: dict[str, Any] = {}

    async def resume_generation_control_stream(**kwargs):
        resumed.update(kwargs)
        yield make_event("complete", sequence=1, data={"response": None})

    service.ai_service = SimpleNamespace(
        resume_generation_control_stream=resume_generation_control_stream
    )

    events = [
        event
        async for event in service.continue_message_generation_stream(
            generation_id=offered.generation_id,
            continuation_id=offered.continuation_id,
            conversation_id=CONVERSATION_ID,
            user_id=USER_ID,
            idempotency_key="continue-key-0001",
            expected_version=offered.version,
        )
    ]

    assert resumed["expected_epoch"] == 0, "the graph was fenced against the leased epoch"
    assert resumed["action"] == "continue"
    assert resumed["thread_id"] == "wf2:conv:turn-1"
    assert "run_start" in [event.type for event in events]


async def test_continue_appends_no_user_message_and_does_not_route():
    """Continue is more of the same answer to the same question.

    Anything that adds a turn would re-route it, and a re-route can land the
    continuation on a different specialist than the one holding the evidence.
    """
    control = build_control_service()
    service = _service(control)
    running = await service._amark_generation_running(await _started(control), user_id=USER_ID)
    offered = await service._amark_generation_continuable(
        running, user_id=USER_ID, assistant_message_id=uuid4()
    )

    async def resume_generation_control_stream(**kwargs):
        yield make_event("complete", sequence=1, data={"response": None})

    service.ai_service = SimpleNamespace(
        resume_generation_control_stream=resume_generation_control_stream
    )
    service.repository = SimpleNamespace(
        acreate=lambda *a, **k: pytest.fail("Continue created a message row of its own")
    )

    events = [
        event
        async for event in service.continue_message_generation_stream(
            generation_id=offered.generation_id,
            continuation_id=offered.continuation_id,
            conversation_id=CONVERSATION_ID,
            user_id=USER_ID,
            idempotency_key="continue-key-0001",
            expected_version=offered.version,
        )
    ]

    assert all(event.type != "user_message_created" for event in events)
    assert all(event.type != "agent_selected" for event in events)


async def test_a_continue_for_a_used_continuation_is_a_typed_refusal():
    """One continuation, one use. The second attempt must not open an epoch."""
    control = build_control_service()
    service = _service(control)
    running = await service._amark_generation_running(await _started(control), user_id=USER_ID)
    offered = await service._amark_generation_continuable(
        running, user_id=USER_ID, assistant_message_id=uuid4()
    )

    calls: list[dict] = []

    async def resume_generation_control_stream(**kwargs):
        calls.append(kwargs)
        yield make_event("complete", sequence=1, data={"response": None})

    service.ai_service = SimpleNamespace(
        resume_generation_control_stream=resume_generation_control_stream
    )

    async def _run(key: str, version: int):
        return [
            event
            async for event in service.continue_message_generation_stream(
                generation_id=offered.generation_id,
                continuation_id=offered.continuation_id,
                conversation_id=CONVERSATION_ID,
                user_id=USER_ID,
                idempotency_key=key,
                expected_version=version,
            )
        ]

    await _run("continue-key-0001", offered.version)
    leased = await control.find_by_logical_turn(
        logical_turn_id="turn-1", user_id=USER_ID, conversation_id=CONVERSATION_ID
    )
    second = await _run("continue-key-0002", leased.version)

    assert len(calls) == 1, "a spent continuation opened a second epoch"
    assert [event.type for event in second] == ["error"]
    assert second[0].data["error_code"] == "continuation_unavailable"


async def test_continue_on_a_deployment_without_controls_says_so():
    service = _service(None)

    events = [
        event
        async for event in service.continue_message_generation_stream(
            generation_id=uuid4(),
            continuation_id=uuid4(),
            conversation_id=CONVERSATION_ID,
            user_id=USER_ID,
            idempotency_key="continue-key-0001",
            expected_version=1,
        )
    ]

    assert [event.type for event in events] == ["error"]


async def test_continue_registers_the_owner_task_so_a_stop_can_reach_it():
    """A resumed epoch is as stoppable as an original one.

    The cooperative event cannot reach a producer blocked inside a provider
    call; the task can, and it does not exist until the coroutine runs.
    """
    control = build_control_service()
    service = _service(control)
    running = await service._amark_generation_running(await _started(control), user_id=USER_ID)
    offered = await service._amark_generation_continuable(
        running, user_id=USER_ID, assistant_message_id=uuid4()
    )

    seen: dict[str, Any] = {}

    async def resume_generation_control_stream(**kwargs):
        entry = get_generation_registry().get(offered.generation_id)
        seen["entry"] = entry
        seen["task"] = getattr(entry, "task", None)
        yield make_event("complete", sequence=1, data={"response": None})

    service.ai_service = SimpleNamespace(
        resume_generation_control_stream=resume_generation_control_stream
    )

    async for _event in service.continue_message_generation_stream(
        generation_id=offered.generation_id,
        continuation_id=offered.continuation_id,
        conversation_id=CONVERSATION_ID,
        user_id=USER_ID,
        idempotency_key="continue-key-0001",
        expected_version=offered.version,
    ):
        pass

    assert seen["entry"] is not None
    assert seen["task"] is asyncio.current_task()
    # Only the worker removes its own entry, and it does so when it is done.
    assert get_generation_registry().get(offered.generation_id) is None
