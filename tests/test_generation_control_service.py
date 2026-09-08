"""Lifecycle invariants and command idempotency for generation control.

The repository enforces "one winner" with a version fence; this service decides
*which* transitions are legal and makes each command answer the same way however
many times it arrives. The two questions it exists to keep separate:

* Stop during active work is a *request*. The worker may be mid-provider-call
  in another process, so the honest answer is ``stop_requested`` — never a
  ``stopped`` the service cannot verify.
* Stop on a paused, already-validated answer is a *decision*. Nothing is
  running, so it resolves immediately to ``completed_partial``.

The repository double lives in ``tests/generation_control_support.py`` and is
deliberately faithful about the fence and about returning ``None`` for a
refused transition, because that is the contract this service is written
against. What it cannot model — real concurrency — is covered in
``tests/integration/test_generation_repository_postgres.py``.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from app.models.generation import GenerationCommandAction, GenerationStatus
from app.schemas.generation import (
    ContinueGenerationCommand,
    CreateGeneration,
    GenerationSnapshot,
    MarkCompleted,
    MarkContinuable,
    MarkStopped,
    StopGenerationCommand,
)
from app.services.generation_control_service import (
    ContinuationUnavailable,
    GenerationNotFound,
    IllegalTransition,
    StaleCommand,
)
from tests.generation_control_support import (
    CONVERSATION_ID,
    USER_ID,
    build_control_service,
)

# The repository double and these ids live in ``generation_control_support`` so
# the message-service lifecycle tests drive the *real* control service over the
# same faithful fence rather than a second fake of it.


@pytest.fixture()
def service():
    return build_control_service()


async def _started(service) -> GenerationSnapshot:
    return await service.start_generation(
        CreateGeneration(
            conversation_id=CONVERSATION_ID,
            user_id=USER_ID,
            logical_turn_id="turn-1",
            checkpoint_thread_id="routing-v2:turn-1",
            active_agent_id="chat_agent",
        )
    )


async def _running(service) -> GenerationSnapshot:
    started = await _started(service)
    return await service.mark_running(
        generation_id=started.generation_id,
        user_id=USER_ID,
        conversation_id=CONVERSATION_ID,
        expected_version=started.version,
    )


async def _continuable(service) -> GenerationSnapshot:
    running = await _running(service)
    return await service.mark_continuable(
        MarkContinuable(
            generation_id=running.generation_id,
            user_id=USER_ID,
            conversation_id=CONVERSATION_ID,
            expected_version=running.version,
            assistant_message_id=uuid4(),
            research_accounting={"searches": 3},
        )
    )


def _stop(snapshot: GenerationSnapshot, *, key: str = "stop-key-0001", version: int | None = None):
    return StopGenerationCommand(
        generation_id=snapshot.generation_id,
        conversation_id=CONVERSATION_ID,
        user_id=USER_ID,
        idempotency_key=key,
        expected_version=snapshot.version if version is None else version,
    )


def _continue(
    snapshot: GenerationSnapshot,
    *,
    key: str = "continue-key-0001",
    continuation_id: UUID | None = None,
    version: int | None = None,
):
    return ContinueGenerationCommand(
        generation_id=snapshot.generation_id,
        continuation_id=continuation_id or snapshot.continuation_id or uuid4(),
        conversation_id=CONVERSATION_ID,
        user_id=USER_ID,
        idempotency_key=key,
        expected_version=snapshot.version if version is None else version,
    )


# ----------------------------------------------------------------------
# start and progress
# ----------------------------------------------------------------------


async def test_a_new_generation_starts_before_any_model_call(service):
    snapshot = await _started(service)

    assert snapshot.status is GenerationStatus.STARTING
    assert snapshot.execution_epoch == 0


async def test_marking_running_advances_the_version(service):
    started = await _started(service)

    running = await service.mark_running(
        generation_id=started.generation_id,
        user_id=USER_ID,
        conversation_id=CONVERSATION_ID,
        expected_version=started.version,
    )

    assert running.status is GenerationStatus.RUNNING
    assert running.version == started.version + 1


async def test_a_command_for_an_unknown_generation_is_not_found(service):
    unknown = GenerationSnapshot(
        generation_id=uuid4(),
        logical_turn_id="turn-x",
        conversation_id=CONVERSATION_ID,
        status=GenerationStatus.RUNNING,
        version=1,
        execution_epoch=0,
    )

    with pytest.raises(GenerationNotFound):
        await service.request_stop(_stop(unknown))


async def test_a_command_from_another_user_is_not_found_not_forbidden(service):
    """Existence is itself information; a wrong owner learns nothing."""
    running = await _running(service)
    command = StopGenerationCommand(
        generation_id=running.generation_id,
        conversation_id=CONVERSATION_ID,
        user_id=uuid4(),
        idempotency_key="stop-key-0001",
        expected_version=running.version,
    )

    with pytest.raises(GenerationNotFound):
        await service.request_stop(command)


# ----------------------------------------------------------------------
# stop
# ----------------------------------------------------------------------


async def test_stop_on_running_is_a_request_and_publishes_the_signal(service):
    received = []
    await service._test_bus.subscribe(lambda signal: _collect(received, signal))
    running = await _running(service)

    snapshot = await service.request_stop(_stop(running))

    assert snapshot.status is GenerationStatus.STOP_REQUESTED
    assert [(item.generation_id, item.version) for item in received] == [
        (running.generation_id, snapshot.version)
    ]


async def test_stop_on_continuable_resolves_immediately_without_cancelling(service):
    """Nothing is running: the answer is already validated and persisted."""
    received = []
    await service._test_bus.subscribe(lambda signal: _collect(received, signal))
    paused = await _continuable(service)

    snapshot = await service.request_stop(_stop(paused))

    assert snapshot.status is GenerationStatus.COMPLETED_PARTIAL
    assert received == []


async def test_stop_repeated_with_the_same_key_returns_the_first_answer(service):
    running = await _running(service)
    first = await service.request_stop(_stop(running))

    second = await service.request_stop(_stop(running))

    assert second == first


async def test_stop_repeated_with_a_new_key_is_still_idempotent(service):
    """A client that lost its key must not be able to double-stop."""
    running = await _running(service)
    first = await service.request_stop(_stop(running))

    second = await service.request_stop(
        _stop(running, key="stop-key-0002", version=first.version)
    )

    assert second.status is GenerationStatus.STOP_REQUESTED
    assert second.version == first.version


async def test_a_stop_issued_against_an_older_version_is_refused_as_stale(service):
    """R5. The sequence this exists for.

    Stop is issued against epoch 0, a Continue lands, and the Stop's retry
    arrives late. Applying it would cancel work the user asked for after it.
    """
    running = await _running(service)
    stale = _stop(running, key="stop-key-0003")
    await service.mark_continuable(
        MarkContinuable(
            generation_id=running.generation_id,
            user_id=USER_ID,
            conversation_id=CONVERSATION_ID,
            expected_version=running.version,
            assistant_message_id=uuid4(),
        )
    )

    with pytest.raises(StaleCommand):
        await service.request_stop(stale)


async def test_a_stop_wait_timeout_leaves_the_state_at_stop_requested(service):
    """The worker did not answer in time. That is pending, not stopped."""
    running = await _running(service)

    snapshot = await service.request_stop(_stop(running))
    awaited = await service.await_stop_settled(
        generation_id=running.generation_id,
        user_id=USER_ID,
        conversation_id=CONVERSATION_ID,
    )

    assert snapshot.status is GenerationStatus.STOP_REQUESTED
    assert awaited.status is GenerationStatus.STOP_REQUESTED


async def test_the_worker_records_the_stop_it_actually_completed(service):
    running = await _running(service)
    requested = await service.request_stop(_stop(running))

    stopped = await service.mark_stopped(
        MarkStopped(
            generation_id=running.generation_id,
            user_id=USER_ID,
            conversation_id=CONVERSATION_ID,
            expected_version=requested.version,
            assistant_message_id=uuid4(),
            continuation_available=True,
        )
    )

    assert stopped.status is GenerationStatus.STOPPED
    assert stopped.continuation_available is True


async def test_stopping_a_completed_generation_is_refused(service):
    running = await _running(service)
    completed = await service.mark_completed(
        MarkCompleted(
            generation_id=running.generation_id,
            user_id=USER_ID,
            conversation_id=CONVERSATION_ID,
            expected_version=running.version,
            assistant_message_id=uuid4(),
        )
    )

    with pytest.raises(IllegalTransition):
        await service.request_stop(_stop(completed, key="stop-key-0009"))


# ----------------------------------------------------------------------
# continue
# ----------------------------------------------------------------------


async def test_continue_on_continuable_leases_the_next_epoch(service):
    paused = await _continuable(service)

    lease = await service.prepare_continue(_continue(paused))

    assert lease.execution_epoch == paused.execution_epoch + 1
    assert lease.snapshot.status is GenerationStatus.CONTINUING
    assert lease.checkpoint_thread_id == "routing-v2:turn-1"
    assert lease.active_agent_id == "chat_agent"


async def test_the_lease_names_both_the_leased_and_the_paused_epoch(service):
    """The row moves; the checkpoint does not.

    The pause node fences a resume against the epoch in *graph* state and
    advances it itself, so the value sent back into the graph is the paused
    epoch. Sending the leased one is refused as stale, and that refusal is
    indistinguishable from a legitimately expired continuation — which is
    exactly why these are two named fields rather than one and a subtraction.
    """
    paused = await _continuable(service)

    lease = await service.prepare_continue(_continue(paused))

    assert lease.paused_epoch == paused.execution_epoch
    assert lease.execution_epoch == lease.paused_epoch + 1


async def test_the_lease_carries_the_accounting_the_next_epoch_must_respect(service):
    """R4. An empty budget is indistinguishable from a fresh turn's quota."""
    paused = await _continuable(service)

    lease = await service.prepare_continue(_continue(paused))

    assert lease.research_accounting == {"searches": 3}


async def test_continue_consumes_the_continuation_id(service):
    """One continuation, one use. A replayed id must not open a second epoch."""
    paused = await _continuable(service)
    first = await service.prepare_continue(_continue(paused))

    # Presented with the *current* version, so this is refused for having no
    # continuation left rather than for naming an old state.
    with pytest.raises(ContinuationUnavailable):
        await service.prepare_continue(
            _continue(
                first.snapshot,
                key="continue-key-0002",
                continuation_id=paused.continuation_id,
            )
        )


async def test_continue_with_the_wrong_continuation_id_is_refused(service):
    paused = await _continuable(service)

    with pytest.raises(ContinuationUnavailable):
        await service.prepare_continue(_continue(paused, continuation_id=uuid4()))


async def test_continue_on_stopped_works_only_when_it_was_marked_available(service):
    running = await _running(service)
    requested = await service.request_stop(_stop(running))
    stopped = await service.mark_stopped(
        MarkStopped(
            generation_id=running.generation_id,
            user_id=USER_ID,
            conversation_id=CONVERSATION_ID,
            expected_version=requested.version,
            assistant_message_id=uuid4(),
            continuation_available=True,
            continuation_id=uuid4(),
        )
    )

    lease = await service.prepare_continue(_continue(stopped))

    assert lease.snapshot.status is GenerationStatus.CONTINUING


async def test_continue_is_refused_when_a_mutation_outcome_is_unknown(service):
    """Nobody can say whether the effect happened; replaying may duplicate it."""
    paused = await _continuable(service)
    blocked = service._test_repository.force(
        paused.generation_id,
        continuation_available=False,
        continuation_block_reason="mutation_outcome_unknown",
    )

    with pytest.raises(ContinuationUnavailable, match="mutation_outcome_unknown"):
        await service.prepare_continue(_continue(blocked))


async def test_continue_is_refused_for_a_completed_generation(service):
    running = await _running(service)
    completed = await service.mark_completed(
        MarkCompleted(
            generation_id=running.generation_id,
            user_id=USER_ID,
            conversation_id=CONVERSATION_ID,
            expected_version=running.version,
            assistant_message_id=uuid4(),
        )
    )

    with pytest.raises(ContinuationUnavailable):
        await service.prepare_continue(_continue(completed))


async def test_a_replayed_continue_does_not_open_a_second_epoch(service):
    paused = await _continuable(service)
    first = await service.prepare_continue(_continue(paused))

    replay = await service.prepare_continue(_continue(paused))

    assert replay.execution_epoch == first.execution_epoch
    assert replay.snapshot.execution_epoch == first.snapshot.execution_epoch


async def test_a_continue_issued_against_an_older_version_is_refused_as_stale(service):
    paused = await _continuable(service)
    stale = _continue(paused, key="continue-key-0003")
    service._test_repository.force(paused.generation_id, version=paused.version + 5)

    with pytest.raises(StaleCommand):
        await service.prepare_continue(stale)


# ----------------------------------------------------------------------
# command action safety
# ----------------------------------------------------------------------


async def test_one_key_cannot_be_reused_for_the_other_action(service):
    """A Stop key replayed as a Continue is a client bug, not a Continue."""
    paused = await _continuable(service)
    await service.request_stop(_stop(paused, key="shared-key-0001"))

    with pytest.raises(IllegalTransition):
        await service.prepare_continue(_continue(paused, key="shared-key-0001"))


async def test_the_recorded_command_action_matches_what_was_issued(service):
    running = await _running(service)
    await service.request_stop(_stop(running, key="stop-key-0004"))

    entry = service._test_repository.commands[(running.generation_id, "stop-key-0004")]

    assert entry["action"] is GenerationCommandAction.STOP
    assert entry["result"] is not None


def _collect(sink: list, signal):
    async def _run() -> None:
        sink.append(signal)

    return _run()
