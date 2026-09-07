"""Owner-scoped compare-and-set access to the generation lifecycle.

Every mutation here is one statement. "Read the status, decide, then write it"
is the bug this table exists to prevent: two Continues on the same paused turn
would both observe ``continuable``, both pass the check, and both increment the
epoch — producing two answers to one question. The ``WHERE version =
:expected`` clause is what makes the decision, and ``RETURNING`` is what tells
the winner what it won.

Reads are filtered by owner *and* conversation, never by id alone. A generation
id is returned to clients, so an id-only lookup would let one user ask about
another user's turn and learn from the answer whether it exists.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import insert, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.generation import (
    Generation,
    GenerationCommand,
    GenerationCommandAction,
    GenerationStatus,
)
from app.repositories.session_transport import RepositorySessionMixin
from app.schemas.generation import CommandClaim, CreateGeneration, GenerationSnapshot

logger = logging.getLogger(__name__)

__all__ = ["GenerationRepository", "ResumeContext"]


@dataclass(frozen=True)
class ResumeContext:
    """Server-side fields a Continue needs. Never returned to a client."""

    checkpoint_thread_id: str
    active_agent_id: str | None
    research_accounting: dict[str, Any] | None
    execution_budget: dict[str, Any] | None

_RETURNED = tuple(Generation.__table__.c)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class GenerationRepository(RepositorySessionMixin):
    """The only writer of ``generations`` and ``generation_commands``."""

    # ------------------------------------------------------------------
    # creation and reads
    # ------------------------------------------------------------------

    async def acreate(self, command: CreateGeneration) -> GenerationSnapshot:
        """Allocate the lifecycle row for a new turn.

        Raises ``IntegrityError`` when the conversation already has an active
        generation or the logical turn was claimed. Both are the partial unique
        index doing its job, and the caller decides whether that is a conflict
        to report or a race it lost.
        """

        def work(session: Session) -> GenerationSnapshot:
            row = session.execute(
                insert(Generation)
                .values(
                    id=uuid.uuid4(),
                    logical_turn_id=command.logical_turn_id,
                    checkpoint_thread_id=command.checkpoint_thread_id,
                    user_id=command.user_id,
                    conversation_id=command.conversation_id,
                    status=GenerationStatus.STARTING,
                    version=1,
                    execution_epoch=0,
                    active_agent_id=command.active_agent_id,
                    continuation_available=False,
                )
                .returning(*_RETURNED)
            ).one()
            session.commit()
            return GenerationSnapshot.from_row(row)

        return await self._arun(work)

    async def aget_owned(
        self,
        generation_id: uuid.UUID,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
    ) -> GenerationSnapshot | None:
        """Read one generation, or ``None`` for anyone who does not own it."""

        def work(session: Session) -> GenerationSnapshot | None:
            row = session.execute(
                select(*_RETURNED).where(
                    Generation.id == generation_id,
                    Generation.user_id == user_id,
                    Generation.conversation_id == conversation_id,
                )
            ).first()
            return None if row is None else GenerationSnapshot.from_row(row)

        return await self._arun(work)

    async def aget_resume_context(
        self,
        generation_id: uuid.UUID,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
    ) -> ResumeContext | None:
        """The parts of the row a resume needs and a client must not see.

        Separate from ``aget_owned`` on purpose: the checkpoint thread is a
        resume handle and the accounting is server bookkeeping, so neither
        belongs on the snapshot every transport returns.
        """

        def work(session: Session) -> ResumeContext | None:
            row = session.execute(
                select(
                    Generation.checkpoint_thread_id,
                    Generation.active_agent_id,
                    Generation.research_accounting,
                    Generation.execution_budget,
                ).where(
                    Generation.id == generation_id,
                    Generation.user_id == user_id,
                    Generation.conversation_id == conversation_id,
                )
            ).first()
            if row is None:
                return None
            return ResumeContext(
                checkpoint_thread_id=row.checkpoint_thread_id,
                active_agent_id=row.active_agent_id,
                research_accounting=row.research_accounting,
                execution_budget=row.execution_budget,
            )

        return await self._arun(work)

    async def aget_active_for_conversation(
        self, conversation_id: uuid.UUID, user_id: uuid.UUID
    ) -> GenerationSnapshot | None:
        """The turn currently in flight for this conversation, if any."""

        def work(session: Session) -> GenerationSnapshot | None:
            row = session.execute(
                select(*_RETURNED)
                .where(
                    Generation.conversation_id == conversation_id,
                    Generation.user_id == user_id,
                    Generation.status.in_(tuple(_active_statuses())),
                )
                .order_by(Generation.created_at.desc())
                .limit(1)
            ).first()
            return None if row is None else GenerationSnapshot.from_row(row)

        return await self._arun(work)

    # ------------------------------------------------------------------
    # transitions
    # ------------------------------------------------------------------

    async def atransition(
        self,
        *,
        generation_id: uuid.UUID,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        expected_statuses: tuple[GenerationStatus, ...],
        expected_version: int,
        values: dict[str, Any],
    ) -> GenerationSnapshot | None:
        """Apply one transition, or return ``None`` if the fence has moved.

        ``None`` is not an error. It means another worker got there first, and
        the caller's next move is to re-read the row rather than to retry
        blindly — the winning transition may already be the one it wanted.
        """
        if "version" in values:
            raise ValueError("version is owned by the transition, not the caller")

        def work(session: Session) -> GenerationSnapshot | None:
            row = session.execute(
                update(Generation)
                .where(
                    Generation.id == generation_id,
                    Generation.user_id == user_id,
                    Generation.conversation_id == conversation_id,
                    Generation.version == expected_version,
                    Generation.status.in_(expected_statuses),
                )
                .values(**values, version=Generation.version + 1, updated_at=_now())
                .returning(*_RETURNED)
            ).first()
            session.commit()
            if row is None:
                logger.info(
                    "Generation %s not transitioned: version %s or status no longer matched",
                    generation_id,
                    expected_version,
                )
                return None
            return GenerationSnapshot.from_row(row)

        return await self._arun(work)

    # ------------------------------------------------------------------
    # command ledger
    # ------------------------------------------------------------------

    async def aclaim_command(
        self,
        *,
        generation_id: uuid.UUID,
        idempotency_key: str,
        action: GenerationCommandAction,
        fence: int,
    ) -> CommandClaim:
        """Claim a command, or report that it has already been issued.

        The insert is the claim. A second caller with the same key loses on the
        unique index and reads the winner's row, so a retried Stop returns the
        result the first Stop recorded instead of stopping something else.

        The recorded ``fence`` is the caller's, not the row's current version.
        A replay therefore carries the epoch it was *issued* against, which is
        what lets a delayed command be recognised as stale rather than applied.
        """

        def work(session: Session) -> CommandClaim:
            try:
                with session.begin_nested():
                    session.execute(
                        insert(GenerationCommand).values(
                            id=uuid.uuid4(),
                            generation_id=generation_id,
                            idempotency_key=idempotency_key,
                            action=action,
                            fence=fence,
                        )
                    )
            except IntegrityError:
                existing = session.execute(
                    select(
                        GenerationCommand.action,
                        GenerationCommand.fence,
                        GenerationCommand.result,
                    ).where(
                        GenerationCommand.generation_id == generation_id,
                        GenerationCommand.idempotency_key == idempotency_key,
                    )
                ).one()
                session.commit()
                return CommandClaim(
                    claimed=False,
                    action=existing.action,
                    fence=existing.fence,
                    result=existing.result,
                )
            session.commit()
            return CommandClaim(claimed=True, action=action, fence=fence, result=None)

        return await self._arun(work)

    async def arecord_command_result(
        self,
        *,
        generation_id: uuid.UUID,
        idempotency_key: str,
        result: dict[str, Any],
    ) -> None:
        """Record what a claimed command did, so its replay can answer."""

        def work(session: Session) -> None:
            outcome = session.execute(
                update(GenerationCommand)
                .where(
                    GenerationCommand.generation_id == generation_id,
                    GenerationCommand.idempotency_key == idempotency_key,
                    GenerationCommand.result.is_(None),
                )
                .values(result=result, completed_at=_now())
            )
            session.commit()
            if outcome.rowcount == 0:
                # Already recorded. Overwriting is exactly what must not
                # happen: the first result is the one every replay answered.
                logger.info(
                    "Command %s on generation %s already had a recorded result",
                    idempotency_key,
                    generation_id,
                )

        await self._arun(work)


def _active_statuses() -> tuple[GenerationStatus, ...]:
    from app.models.generation import ACTIVE_STATUSES

    return tuple(sorted(ACTIVE_STATUSES, key=lambda status: status.value))
