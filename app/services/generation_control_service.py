"""Lifecycle invariants and idempotent commands for one generation.

The repository decides *who wins* a race; this service decides *what is legal*
and makes each command answer the same way however many times it arrives.

Two distinctions carry most of the design:

* **Stop during active work is a request, not an outcome.** The worker may be
  mid-provider-call in another process, so the honest answer is
  ``stop_requested``. Reporting ``stopped`` before the worker confirms it would
  tell a user their tool call was abandoned when it may still be running.
* **Stop on a paused turn is a decision.** The answer there is already
  validated and persisted and nothing is executing, so it resolves immediately
  to ``completed_partial`` and publishes no cancellation.

Ordering inside a command is deliberate: claim the idempotency key *first*, then
check the fence. The reverse looks tidier and breaks replay — a command's own
transition advances the version, so a legitimate retry would fail the fence
check and be reported as stale instead of returning the result it already
produced. Refusals are recorded in the ledger too, so a replayed stale command
is refused the same way rather than being reconsidered against a later epoch.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from app.models.generation import (
    TERMINAL_STATUSES,
    GenerationCommandAction,
    GenerationStatus,
)
from app.schemas.generation import (
    CommandClaim,
    ContinuationLease,
    ContinueGenerationCommand,
    CreateGeneration,
    GenerationSnapshot,
    MarkCompleted,
    MarkContinuable,
    MarkStopped,
    StopGenerationCommand,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ContinuationUnavailable",
    "GenerationControlError",
    "GenerationControlService",
    "GenerationNotFound",
    "IllegalTransition",
    "StaleCommand",
]

#: Statuses a worker may report a stop from. Broad on purpose: a stop can land
#: while the turn is still starting, and refusing it there would leave the row
#: active with nothing running.
_STOPPABLE = (
    GenerationStatus.STARTING,
    GenerationStatus.RUNNING,
    GenerationStatus.FINALIZING_AFTER_LIMIT,
    GenerationStatus.CONTINUING,
    GenerationStatus.STOP_REQUESTED,
)
_COMPLETABLE = (
    GenerationStatus.STARTING,
    GenerationStatus.RUNNING,
    GenerationStatus.FINALIZING_AFTER_LIMIT,
    GenerationStatus.CONTINUING,
)
_CONTINUABLE_FROM = (
    GenerationStatus.RUNNING,
    GenerationStatus.FINALIZING_AFTER_LIMIT,
    GenerationStatus.CONTINUING,
)

_ERROR_KEY = "__control_error__"


class GenerationControlError(Exception):
    """Base for refusals a transport maps to a status code."""

    code = "generation_control_error"

    def __init__(self, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.detail = dict(detail or {})


class GenerationNotFound(GenerationControlError):
    """No such generation *for this owner*. Deliberately indistinguishable.

    A wrong owner and a wrong id return the same thing, because "it exists but
    is not yours" is itself information about another user's conversation.
    """

    code = "generation_not_found"


class IllegalTransition(GenerationControlError):
    """The command is well-formed but not legal from the current status."""

    code = "illegal_transition"


class StaleCommand(GenerationControlError):
    """The command was issued against a version that has since moved on."""

    code = "stale_command"


class ContinuationUnavailable(GenerationControlError):
    """This generation cannot be continued, and the reason is safe to say."""

    code = "continuation_unavailable"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class GenerationControlService:
    """The only place lifecycle legality is decided."""

    def __init__(
        self,
        *,
        repository: Any,
        bus: Any,
        stop_wait_seconds: float = 5.0,
    ) -> None:
        self._repository = repository
        self._bus = bus
        self._stop_wait_seconds = max(0.0, float(stop_wait_seconds))

    # ------------------------------------------------------------------
    # worker-side transitions
    # ------------------------------------------------------------------

    async def start_generation(self, command: CreateGeneration) -> GenerationSnapshot:
        return await self._repository.acreate(command)

    async def mark_running(
        self,
        *,
        generation_id: uuid.UUID,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        expected_version: int,
    ) -> GenerationSnapshot:
        return await self._must_transition(
            generation_id=generation_id,
            user_id=user_id,
            conversation_id=conversation_id,
            expected_statuses=(GenerationStatus.STARTING,),
            expected_version=expected_version,
            values={"status": GenerationStatus.RUNNING},
        )

    async def mark_continuable(self, command: MarkContinuable) -> GenerationSnapshot:
        """Publish a validated partial and offer Continue.

        The continuation id is minted here and is single-use: consuming it is
        what stops a replayed Continue from opening a second epoch.
        """
        blocked = bool(command.continuation_block_reason)
        values: dict[str, Any] = {
            "status": GenerationStatus.CONTINUABLE,
            "assistant_message_id": command.assistant_message_id,
            "continuation_available": not blocked,
            "continuation_id": None if blocked else uuid.uuid4(),
            "continuation_block_reason": command.continuation_block_reason,
        }
        if command.research_accounting is not None:
            values["research_accounting"] = command.research_accounting
        if command.execution_budget is not None:
            values["execution_budget"] = command.execution_budget
        return await self._must_transition(
            generation_id=command.generation_id,
            user_id=command.user_id,
            conversation_id=command.conversation_id,
            expected_statuses=_CONTINUABLE_FROM,
            expected_version=command.expected_version,
            values=values,
        )

    async def mark_stopped(self, command: MarkStopped) -> GenerationSnapshot:
        blocked = bool(command.continuation_block_reason)
        available = bool(command.continuation_available) and not blocked
        return await self._must_transition(
            generation_id=command.generation_id,
            user_id=command.user_id,
            conversation_id=command.conversation_id,
            expected_statuses=_STOPPABLE,
            expected_version=command.expected_version,
            values={
                "status": GenerationStatus.STOPPED,
                "assistant_message_id": command.assistant_message_id,
                "continuation_available": available,
                "continuation_id": command.continuation_id if available else None,
                "continuation_block_reason": command.continuation_block_reason,
                "terminal_reason": command.terminal_reason or "stopped",
                "terminal_at": _now(),
            },
        )

    async def mark_failed(
        self,
        *,
        generation_id: uuid.UUID,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        expected_version: int,
        terminal_reason: str,
    ) -> GenerationSnapshot:
        """The turn produced nothing a client can read.

        Distinct from ``mark_completed(partial=True)``: that one has an answer
        and simply stopped early, while this one has none. Nothing here is
        continuable, because there is no first half to continue.
        """
        return await self._must_transition(
            generation_id=generation_id,
            user_id=user_id,
            conversation_id=conversation_id,
            expected_statuses=_STOPPABLE,
            expected_version=expected_version,
            values={
                "status": GenerationStatus.FAILED,
                "continuation_available": False,
                "continuation_id": None,
                "terminal_reason": terminal_reason,
                "terminal_at": _now(),
            },
        )

    async def mark_completed(self, command: MarkCompleted) -> GenerationSnapshot:
        status = (
            GenerationStatus.COMPLETED_PARTIAL if command.partial else GenerationStatus.COMPLETED
        )
        return await self._must_transition(
            generation_id=command.generation_id,
            user_id=command.user_id,
            conversation_id=command.conversation_id,
            expected_statuses=_COMPLETABLE,
            expected_version=command.expected_version,
            values={
                "status": status,
                "assistant_message_id": command.assistant_message_id,
                "continuation_available": False,
                "continuation_id": None,
                "terminal_reason": command.terminal_reason,
                "terminal_at": _now(),
            },
        )

    # ------------------------------------------------------------------
    # client commands
    # ------------------------------------------------------------------

    async def request_stop(self, command: StopGenerationCommand) -> GenerationSnapshot:
        """Ask for a stop. Idempotent, fenced, and honest about pending work."""
        snapshot = await self._require_owned(
            command.generation_id, command.user_id, command.conversation_id
        )
        claim = await self._repository.aclaim_command(
            generation_id=command.generation_id,
            idempotency_key=command.idempotency_key,
            action=GenerationCommandAction.STOP,
            fence=command.expected_version,
        )
        if not claim.claimed:
            return self._replay(claim, GenerationCommandAction.STOP, GenerationSnapshot)

        try:
            self._require_fresh(command.expected_version, snapshot)
            if snapshot.status in TERMINAL_STATUSES:
                raise IllegalTransition(
                    f"generation is already {snapshot.status.value}",
                    detail={"status": snapshot.status.value},
                )
            if snapshot.status is GenerationStatus.STOP_REQUESTED:
                # Already asked. Re-transitioning would move the fence and make
                # every client's cached version stale for no reason.
                result = snapshot
            elif snapshot.status is GenerationStatus.CONTINUABLE:
                result = await self._accept_partial(command, snapshot)
            else:
                result = await self._request_worker_stop(command, snapshot)
        except GenerationControlError as error:
            await self._record(command, {_ERROR_KEY: _serialize_error(error)})
            raise

        await self._record(command, result.model_dump(mode="json"))
        return result

    async def prepare_continue(self, command: ContinueGenerationCommand) -> ContinuationLease:
        """Lease the next epoch, or refuse with a reason the client can act on."""
        snapshot = await self._require_owned(
            command.generation_id, command.user_id, command.conversation_id
        )
        claim = await self._repository.aclaim_command(
            generation_id=command.generation_id,
            idempotency_key=command.idempotency_key,
            action=GenerationCommandAction.CONTINUE,
            fence=command.expected_version,
        )
        if not claim.claimed:
            return self._replay(claim, GenerationCommandAction.CONTINUE, ContinuationLease)

        try:
            self._require_fresh(command.expected_version, snapshot)
            self._require_continuable(command, snapshot)
            context = await self._repository.aget_resume_context(
                command.generation_id, command.user_id, command.conversation_id
            )
            if context is None:
                raise GenerationNotFound("generation is not readable by this owner")
            advanced = await self._repository.atransition(
                generation_id=command.generation_id,
                user_id=command.user_id,
                conversation_id=command.conversation_id,
                expected_statuses=(GenerationStatus.CONTINUABLE, GenerationStatus.STOPPED),
                expected_version=snapshot.version,
                values={
                    "status": GenerationStatus.CONTINUING,
                    "execution_epoch": snapshot.execution_epoch + 1,
                    "continuation_available": False,
                    "continuation_id": None,
                    "execution_budget": None,
                    "terminal_at": None,
                    "terminal_reason": None,
                },
            )
            if advanced is None:
                raise StaleCommand("another command changed this generation first")
            lease = ContinuationLease(
                snapshot=advanced,
                execution_epoch=advanced.execution_epoch,
                # The epoch the graph is still in. The row moved; the checkpoint
                # did not, and the pause node advances it itself on resume.
                paused_epoch=snapshot.execution_epoch,
                checkpoint_thread_id=context.checkpoint_thread_id,
                active_agent_id=context.active_agent_id,
                research_accounting=context.research_accounting,
            )
        except GenerationControlError as error:
            await self._record(command, {_ERROR_KEY: _serialize_error(error)})
            raise

        await self._record(command, lease.model_dump(mode="json"))
        return lease

    async def aget_snapshot(
        self,
        *,
        generation_id: uuid.UUID,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
    ) -> GenerationSnapshot | None:
        """Read one generation's current state, or ``None`` for a non-owner.

        The read every status endpoint and every "where did this land?" recovery
        path uses. Returns ``None`` rather than raising, because "not yours" and
        "not there" must be indistinguishable.
        """
        return await self._repository.aget_owned(generation_id, user_id, conversation_id)

    async def find_by_logical_turn(
        self,
        *,
        logical_turn_id: str,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
    ) -> GenerationSnapshot | None:
        """The generation for one logical turn, or ``None`` for a non-owner.

        The route a caller holding only a turn-scoped id takes — the user
        message id an older Stop endpoint carries. Exposed here rather than
        letting callers reach into the repository, so owner scoping stays this
        service's concern.
        """
        return await self._repository.aget_by_logical_turn(
            str(logical_turn_id), user_id, conversation_id
        )

    async def await_stop_settled(
        self,
        *,
        generation_id: uuid.UUID,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
    ) -> GenerationSnapshot:
        """Wait briefly for the worker to confirm, then report what is true.

        A timeout is not an error and is not converted into ``stopped``. The
        caller returns ``stop_requested``, which is exactly what happened.
        """
        deadline = time.monotonic() + self._stop_wait_seconds
        snapshot = await self._require_owned(generation_id, user_id, conversation_id)
        while snapshot.status is GenerationStatus.STOP_REQUESTED:
            if time.monotonic() >= deadline:
                return snapshot
            await asyncio.sleep(min(0.05, max(0.005, self._stop_wait_seconds / 20)))
            snapshot = await self._require_owned(generation_id, user_id, conversation_id)
        return snapshot

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    async def _accept_partial(
        self, command: StopGenerationCommand, snapshot: GenerationSnapshot
    ) -> GenerationSnapshot:
        """Nothing is running: take the validated partial and finish the turn."""
        result = await self._repository.atransition(
            generation_id=command.generation_id,
            user_id=command.user_id,
            conversation_id=command.conversation_id,
            expected_statuses=(GenerationStatus.CONTINUABLE,),
            expected_version=snapshot.version,
            values={
                "status": GenerationStatus.COMPLETED_PARTIAL,
                "continuation_available": False,
                "continuation_id": None,
                "terminal_reason": "stopped_partial",
                "terminal_at": _now(),
            },
        )
        if result is None:
            raise StaleCommand("another command changed this generation first")
        return result

    async def _request_worker_stop(
        self, command: StopGenerationCommand, snapshot: GenerationSnapshot
    ) -> GenerationSnapshot:
        result = await self._repository.atransition(
            generation_id=command.generation_id,
            user_id=command.user_id,
            conversation_id=command.conversation_id,
            expected_statuses=tuple(_STOPPABLE),
            expected_version=snapshot.version,
            values={"status": GenerationStatus.STOP_REQUESTED},
        )
        if result is None:
            raise StaleCommand("another command changed this generation first")
        # Published after the durable transition, never instead of it: a worker
        # that misses the signal still finds stop_requested on its next check.
        await self._bus.publish_stop(result.generation_id, result.version)
        return result

    async def _require_owned(
        self, generation_id: uuid.UUID, user_id: uuid.UUID, conversation_id: uuid.UUID
    ) -> GenerationSnapshot:
        snapshot = await self._repository.aget_owned(generation_id, user_id, conversation_id)
        if snapshot is None:
            raise GenerationNotFound("generation is not readable by this owner")
        return snapshot

    async def _must_transition(self, **kwargs: Any) -> GenerationSnapshot:
        result = await self._repository.atransition(**kwargs)
        if result is None:
            raise IllegalTransition(
                "the generation was not in the expected state",
                detail={"expected_version": kwargs.get("expected_version")},
            )
        return result

    @staticmethod
    def _require_fresh(expected_version: int, snapshot: GenerationSnapshot) -> None:
        if int(expected_version) < int(snapshot.version):
            raise StaleCommand(
                "this command was issued against an earlier state",
                detail={"issued_at_version": int(expected_version), "version": snapshot.version},
            )
        if int(expected_version) > int(snapshot.version):
            # A client claiming a version the server never reached is either
            # confused or forged; either way it is not a fence we can honor.
            raise StaleCommand(
                "this command names a state this generation has not reached",
                detail={"issued_at_version": int(expected_version), "version": snapshot.version},
            )

    @staticmethod
    def _require_continuable(
        command: ContinueGenerationCommand, snapshot: GenerationSnapshot
    ) -> None:
        if not snapshot.continuation_available or snapshot.continuation_id is None:
            raise ContinuationUnavailable(
                snapshot.continuation_block_reason or "this generation cannot be continued",
                detail={
                    "status": snapshot.status.value,
                    "reason": snapshot.continuation_block_reason,
                },
            )
        if snapshot.continuation_id != command.continuation_id:
            raise ContinuationUnavailable(
                "the continuation has already been used or was never issued",
                detail={"status": snapshot.status.value},
            )

    async def _record(self, command: Any, result: dict[str, Any]) -> None:
        await self._repository.arecord_command_result(
            generation_id=command.generation_id,
            idempotency_key=command.idempotency_key,
            result=result,
        )

    @staticmethod
    def _replay(claim: CommandClaim, action: GenerationCommandAction, model: type) -> Any:
        """Answer a replay exactly as the first attempt answered."""
        if claim.action is not action:
            raise IllegalTransition(
                "this idempotency key was already used for a different command",
                detail={"recorded_action": claim.action.value},
            )
        if not claim.result:
            # Claimed but not finished: the first attempt is still running, or
            # died before recording. Refusing is safer than acting twice.
            raise IllegalTransition(
                "this command is already in progress",
                detail={"recorded_action": claim.action.value},
            )
        recorded = claim.result.get(_ERROR_KEY)
        if recorded:
            raise _deserialize_error(recorded)
        return model.model_validate(claim.result)


def _serialize_error(error: GenerationControlError) -> dict[str, Any]:
    return {"code": error.code, "message": str(error), "detail": error.detail}


_ERROR_TYPES: dict[str, type[GenerationControlError]] = {
    GenerationNotFound.code: GenerationNotFound,
    IllegalTransition.code: IllegalTransition,
    StaleCommand.code: StaleCommand,
    ContinuationUnavailable.code: ContinuationUnavailable,
}


def _deserialize_error(recorded: dict[str, Any]) -> GenerationControlError:
    error_type = _ERROR_TYPES.get(str(recorded.get("code")), GenerationControlError)
    return error_type(
        str(recorded.get("message") or "command refused"),
        detail=recorded.get("detail") or {},
    )
