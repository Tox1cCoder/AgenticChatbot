"""Stop, across workers, and what a disconnect is allowed to conclude.

The defect the durable lifecycle exists to fix: a Stop that lands on a worker
other than the one streaming found nothing in that process's registry and
answered "not in flight" while the turn carried on. The registry is now a
*shortcut* — it makes a same-process Stop immediate — and the row is the
authority, so the ordering is: transition durably first, signal locally second.
A worker that misses the signal still finds ``stop_requested`` on its next
check; a signal sent without the transition reaches only this process.

Two things Stop must never do, both asserted here:

* report ``stopped`` for a stop nobody confirmed. The worker may be
  mid-provider-call in another process. ``stop_requested`` is a successful
  pending state, not a timeout error.
* let a socket close decide a lifecycle status. A disconnect is transport
  recovery; only the worker's own transition makes a turn ``stopped``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.models.generation import GenerationStatus
from app.schemas.generation import CreateGeneration
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


def _service(control) -> MessageService:
    service = MessageService.__new__(MessageService)
    service.generation_control_service = control
    service._turn_coordinator = None
    service.conversation_validation_utils = SimpleNamespace(
        validate_conversation_access=lambda user_id, conversation_id: None,
    )
    service.message_validation_utils = SimpleNamespace(
        validate_message_access=lambda user_id, message_id: None
    )
    return service


async def _running(control, *, logical_turn_id: str = "turn-1"):
    started = await control.start_generation(
        CreateGeneration(
            conversation_id=CONVERSATION_ID,
            user_id=USER_ID,
            logical_turn_id=logical_turn_id,
            checkpoint_thread_id=f"wf2:conv:{logical_turn_id}",
            active_agent_id="chat_agent",
        )
    )
    return await control.mark_running(
        generation_id=started.generation_id,
        user_id=USER_ID,
        conversation_id=CONVERSATION_ID,
        expected_version=started.version,
    )


# ----------------------------------------------------------------------
# the durable Stop
# ----------------------------------------------------------------------


async def test_a_stop_transitions_the_row_before_signalling_this_process():
    """Order matters: the row is what a worker elsewhere can see.

    Signalling first and transitioning second would leave a window in which
    the only record of the Stop is in one process's memory.
    """
    control = build_control_service()
    service = _service(control)
    running = await _running(control)

    published: list = []
    await control._test_bus.subscribe(lambda signal: published.append(signal))

    snapshot = await service.stop_generation(
        generation_id=running.generation_id,
        conversation_id=CONVERSATION_ID,
        user_id=USER_ID,
        idempotency_key="stop-key-0001",
        expected_version=running.version,
    )

    assert snapshot.status is GenerationStatus.STOP_REQUESTED
    assert [item.generation_id for item in published] == [running.generation_id]


async def test_a_stop_for_a_turn_owned_by_another_worker_still_transitions():
    """The whole point. No local registry entry, and the Stop still lands."""
    control = build_control_service()
    service = _service(control)
    running = await _running(control)

    assert get_generation_registry().get(running.generation_id) is None

    snapshot = await service.stop_generation(
        generation_id=running.generation_id,
        conversation_id=CONVERSATION_ID,
        user_id=USER_ID,
        idempotency_key="stop-key-0001",
        expected_version=running.version,
    )

    assert snapshot.status is GenerationStatus.STOP_REQUESTED


async def test_a_stop_never_reports_stopped_before_a_worker_confirms():
    """``stop_requested`` is honest; ``stopped`` would be a claim about work."""
    control = build_control_service(stop_wait_seconds=0.02)
    service = _service(control)
    running = await _running(control)

    snapshot = await service.stop_generation(
        generation_id=running.generation_id,
        conversation_id=CONVERSATION_ID,
        user_id=USER_ID,
        idempotency_key="stop-key-0001",
        expected_version=running.version,
    )

    assert snapshot.status is not GenerationStatus.STOPPED
    assert snapshot.status is GenerationStatus.STOP_REQUESTED


async def test_a_stop_reports_stopped_once_the_worker_has_confirmed():
    control = build_control_service(stop_wait_seconds=1.0)
    service = _service(control)
    running = await _running(control)

    async def confirm_shortly():
        await asyncio.sleep(0.03)
        requested = await control.find_by_logical_turn(
            logical_turn_id="turn-1", user_id=USER_ID, conversation_id=CONVERSATION_ID
        )
        await service._amark_generation_stopped(
            requested,
            user_id=USER_ID,
            assistant_message_id=None,
            terminal_reason="user_requested",
        )

    worker = asyncio.create_task(confirm_shortly())
    snapshot = await service.stop_generation(
        generation_id=running.generation_id,
        conversation_id=CONVERSATION_ID,
        user_id=USER_ID,
        idempotency_key="stop-key-0001",
        expected_version=running.version,
    )
    await worker

    assert snapshot.status is GenerationStatus.STOPPED


async def test_a_stop_signals_the_local_producer_without_dropping_its_entry():
    """The entry survives so a retried Stop still has something to cancel.

    Removing it on the first request is what left a retry with nothing to
    interrupt while the turn was still running.
    """
    control = build_control_service(stop_wait_seconds=0.02)
    service = _service(control)
    running = await _running(control)
    entry = get_generation_registry().register(
        generation_id=running.generation_id,
        conversation_id=CONVERSATION_ID,
        user_id=USER_ID,
    )

    await service.stop_generation(
        generation_id=running.generation_id,
        conversation_id=CONVERSATION_ID,
        user_id=USER_ID,
        idempotency_key="stop-key-0001",
        expected_version=running.version,
    )

    assert entry.is_cancelled
    assert get_generation_registry().get(running.generation_id) is entry


async def test_a_replayed_stop_returns_its_own_recorded_answer():
    control = build_control_service(stop_wait_seconds=0.02)
    service = _service(control)
    running = await _running(control)

    async def _stop():
        return await service.stop_generation(
            generation_id=running.generation_id,
            conversation_id=CONVERSATION_ID,
            user_id=USER_ID,
            idempotency_key="stop-key-0001",
            expected_version=running.version,
        )

    first = await _stop()
    second = await _stop()

    assert second.status is first.status
    assert second.version == first.version


async def test_a_stop_that_lost_the_race_to_the_turn_ending_is_not_an_error():
    """The user asked for the turn to be over, and it is.

    A Stop click that arrives just after the answer landed must report where
    the turn ended rather than raising: nothing is left to stop, which is the
    outcome that was requested.
    """
    control = build_control_service(stop_wait_seconds=0.02)
    service = _service(control)
    running = await _running(control)
    await service._amark_generation_completed(
        running, user_id=USER_ID, assistant_message_id=uuid4(), terminal_reason="completed"
    )
    settled = await control.find_by_logical_turn(
        logical_turn_id="turn-1", user_id=USER_ID, conversation_id=CONVERSATION_ID
    )

    snapshot = await service.stop_generation(
        generation_id=running.generation_id,
        conversation_id=CONVERSATION_ID,
        user_id=USER_ID,
        idempotency_key="stop-key-late-0001",
        expected_version=settled.version,
    )

    assert snapshot.status is GenerationStatus.COMPLETED


async def test_an_illegal_transition_on_an_active_turn_is_still_an_error():
    """The forgiveness above is narrow, and must stay narrow.

    Swallowing every ``IllegalTransition`` would turn a genuine defect — a row
    still active but refusing a legal Stop — into a silent success, and Stop
    would appear to work while doing nothing.
    """
    from app.services.generation_control_service import IllegalTransition

    control = build_control_service(stop_wait_seconds=0.02)
    service = _service(control)
    running = await _running(control)

    async def refuse(command):
        raise IllegalTransition("refused for a reason that is not terminality")

    control.request_stop = refuse

    with pytest.raises(IllegalTransition):
        await service.stop_generation(
            generation_id=running.generation_id,
            conversation_id=CONVERSATION_ID,
            user_id=USER_ID,
            idempotency_key="stop-key-0001",
            expected_version=running.version,
        )


async def test_a_delayed_stop_replay_after_a_continue_is_refused_as_stale():
    """R5, at the service boundary.

    Sequence: a Stop settles against epoch 0; a Continue is accepted and moves
    the row on; a delayed retry of the *original* Stop arrives carrying the old
    fence. Executing it would cancel epoch 1, which nobody asked to stop.
    """
    from app.schemas.generation import ContinueGenerationCommand
    from app.services.generation_control_service import StaleCommand

    control = build_control_service(stop_wait_seconds=0.02)
    service = _service(control)
    running = await _running(control)

    paused = await service._amark_generation_continuable(
        running, user_id=USER_ID, assistant_message_id=uuid4()
    )
    stale_fence = paused.version

    # The Stop that settles the paused turn, then a Continue on top of it.
    await service.stop_generation(
        generation_id=paused.generation_id,
        conversation_id=CONVERSATION_ID,
        user_id=USER_ID,
        idempotency_key="stop-key-0001",
        expected_version=stale_fence,
    )
    reopened = control._test_repository.force(
        paused.generation_id,
        status=GenerationStatus.CONTINUABLE,
        continuation_available=True,
        continuation_id=uuid4(),
    )
    lease = await control.prepare_continue(
        ContinueGenerationCommand(
            generation_id=reopened.generation_id,
            continuation_id=reopened.continuation_id,
            conversation_id=CONVERSATION_ID,
            user_id=USER_ID,
            idempotency_key="continue-key-0001",
            expected_version=reopened.version,
        )
    )
    assert lease.execution_epoch == 1

    # The delayed replay, under a key the ledger has never seen.
    with pytest.raises(StaleCommand):
        await service.stop_generation(
            generation_id=paused.generation_id,
            conversation_id=CONVERSATION_ID,
            user_id=USER_ID,
            idempotency_key="stop-key-delayed-retry",
            expected_version=stale_fence,
        )

    assert control._test_repository.status_of(paused.generation_id) is GenerationStatus.CONTINUING


# ----------------------------------------------------------------------
# the turn-scoped entry point
# ----------------------------------------------------------------------


async def test_a_turn_scoped_stop_resolves_the_generation_from_its_logical_turn():
    control = build_control_service(stop_wait_seconds=0.02)
    service = _service(control)
    user_message_id = uuid4()
    running = await _running(control, logical_turn_id=str(user_message_id))

    result = await service.stop_message_generation(
        conversation_id=CONVERSATION_ID,
        user_id=USER_ID,
        user_message_id=user_message_id,
    )

    assert result["status"] == "stop_requested"
    assert result["generation"]["generation_id"] == str(running.generation_id)
    assert result["generation"]["status"] == "stop_requested"


async def test_a_stale_turn_id_stops_nothing_rather_than_the_turn_running_now():
    """Resolving to "whatever is active" would be the R5 defect one level up."""
    control = build_control_service(stop_wait_seconds=0.02)
    service = _service(control)
    running = await _running(control, logical_turn_id=str(uuid4()))

    result = await service.stop_message_generation(
        conversation_id=CONVERSATION_ID,
        user_id=USER_ID,
        user_message_id=uuid4(),
    )

    assert result["status"] == "not_inflight"
    assert control._test_repository.status_of(running.generation_id) is GenerationStatus.RUNNING


async def test_a_turn_scoped_stop_from_another_user_is_not_found():
    """Existence is itself information about another user's conversation."""
    control = build_control_service(stop_wait_seconds=0.02)
    service = _service(control)
    user_message_id = uuid4()
    await _running(control, logical_turn_id=str(user_message_id))

    result = await service.stop_message_generation(
        conversation_id=CONVERSATION_ID,
        user_id=uuid4(),
        user_message_id=user_message_id,
    )

    assert result["status"] == "not_inflight"


async def test_a_confirmed_stop_is_still_reported_as_cancelled_for_old_clients():
    """Existing clients switch on ``cancelled``; the durable status travels too."""
    control = build_control_service(stop_wait_seconds=0.02)
    service = _service(control)
    user_message_id = uuid4()
    running = await _running(control, logical_turn_id=str(user_message_id))
    await service._amark_generation_stopped(
        running, user_id=USER_ID, assistant_message_id=None, terminal_reason="user_requested"
    )

    result = await service.stop_message_generation(
        conversation_id=CONVERSATION_ID,
        user_id=USER_ID,
        user_message_id=user_message_id,
    )

    assert result["status"] == "cancelled"
    assert result["generation"]["status"] == "stopped"


# ----------------------------------------------------------------------
# a deployment with no lifecycle service
# ----------------------------------------------------------------------


async def test_without_a_lifecycle_service_stop_falls_back_to_this_process():
    control = None
    service = _service(control)
    user_message_id = uuid4()
    get_generation_registry().register(
        generation_id=user_message_id,
        conversation_id=CONVERSATION_ID,
        user_id=USER_ID,
    )

    result = await service.stop_message_generation(
        conversation_id=CONVERSATION_ID,
        user_id=USER_ID,
        user_message_id=user_message_id,
        wait_seconds=0.01,
    )

    assert result["status"] == "stop_requested"
    # The fallback cannot report a durable status, and does not invent one.
    assert "generation" not in result


async def test_the_local_fallback_still_refuses_another_users_entry():
    service = _service(None)
    user_message_id = uuid4()
    get_generation_registry().register(
        generation_id=user_message_id,
        conversation_id=CONVERSATION_ID,
        user_id=uuid4(),
    )

    result = await service.stop_message_generation(
        conversation_id=CONVERSATION_ID,
        user_id=USER_ID,
        user_message_id=user_message_id,
        wait_seconds=0.01,
    )

    assert result["status"] == "not_inflight"


# ----------------------------------------------------------------------
# a disconnect decides nothing
# ----------------------------------------------------------------------


async def test_a_socket_close_alone_changes_no_lifecycle_status():
    """HTTP reconnect is transport recovery, not semantic Stop.

    Nothing here calls a transition, so the row must be exactly where the
    worker left it — a closed socket is not evidence about the turn.
    """
    control = build_control_service()
    running = await _running(control)
    entry = get_generation_registry().register(
        generation_id=running.generation_id,
        conversation_id=CONVERSATION_ID,
        user_id=USER_ID,
    )

    # What a disconnect actually does: the consumer stops reading. Nothing here
    # touches the service, which is the point — no transition is involved.
    entry.request_cancel()

    assert control._test_repository.status_of(running.generation_id) is GenerationStatus.RUNNING
    settled = await control.await_stop_settled(
        generation_id=running.generation_id,
        user_id=USER_ID,
        conversation_id=CONVERSATION_ID,
    )
    assert settled.status is GenerationStatus.RUNNING


async def test_no_registry_only_cancelled_result_remains_on_the_durable_path():
    """``cancelled`` must be derived from the row, never from a resolved future.

    A future resolved by a disconnecting consumer is not the worker agreeing
    that it stopped.
    """
    control = build_control_service(stop_wait_seconds=0.02)
    service = _service(control)
    user_message_id = uuid4()
    running = await _running(control, logical_turn_id=str(user_message_id))
    entry = get_generation_registry().register(
        generation_id=running.generation_id,
        conversation_id=CONVERSATION_ID,
        user_id=USER_ID,
    )
    entry.resolve({"id": "a-message-the-worker-never-committed"})

    result = await service.stop_message_generation(
        conversation_id=CONVERSATION_ID,
        user_id=USER_ID,
        user_message_id=user_message_id,
    )

    assert result["status"] == "stop_requested"
    assert result["message"] is None
