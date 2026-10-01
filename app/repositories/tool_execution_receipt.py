"""Repository for durable mutation execution receipts.

Every method here is a compare-and-set, never a read-then-write. Two replays
of the same turn can run concurrently, so "check the status, then update it"
would let both observe ``reserved`` and both call the provider. The row's
unique key and the ``WHERE status = ...`` clauses are what make the decision
atomic.

Reads are always filtered by owner. A receipt is not a global cache: one
user's completed effect answering another user's call would be a cross-user
data leak dressed up as a retry.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.tool_execution_receipt import ReceiptStatus, ToolExecutionReceipt
from app.repositories.session_transport import RepositorySessionMixin

logger = logging.getLogger(__name__)

__all__ = ["ToolExecutionReceiptRepository"]


class ToolExecutionReceiptRepository(RepositorySessionMixin):
    """Atomic lifecycle transitions for :class:`ToolExecutionReceipt`."""

    def __init__(
        self,
        session_factory: Callable[[], Any],
        async_session_factory: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__(
            session_factory=session_factory,
            async_session_factory=async_session_factory,
        )

    # ------------------------------------------------------------------
    # reservation
    # ------------------------------------------------------------------

    async def areserve(self, *, scope: Any, key: str) -> Any:
        """Claim the key, or report the row that already holds it.

        The insert is attempted first and the unique-violation is the branch
        that reads the existing row. Reading first would leave a window in
        which two callers both see nothing and both insert.
        """
        from app.services.tool_execution_receipt_service import ReceiptRecord

        def work(session: Session) -> Any:
            record = ToolExecutionReceipt(
                execution_key=key,
                status=ReceiptStatus.RESERVED,
                user_id=scope.user_id,
                conversation_id=scope.conversation_id,
                turn_id=scope.turn_id,
                thread_id=scope.thread_id,
                dispatch_id=scope.dispatch_id,
                task_id=scope.task_id,
                tool_call_id=scope.tool_call_id,
                qualified_tool_id=scope.tool_id,
                provider_idempotency=bool(scope.provider_idempotency),
            )
            session.add(record)
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                return self._existing_record(session, key=key, scope=scope)
            return ReceiptRecord(execution_key=key, status=ReceiptStatus.RESERVED, fresh=True)

        return await self._arun(work)

    @staticmethod
    def _existing_record(session: Session, *, key: str, scope: Any) -> Any:
        """The row this owner is allowed to observe for ``key``.

        A row owned by someone else is reported as if it were freshly reserved:
        this caller must run its own mutation rather than adopt an effect it
        did not cause.
        """
        from app.services.tool_execution_receipt_service import ReceiptRecord

        row = (
            session.execute(
                select(ToolExecutionReceipt).where(
                    ToolExecutionReceipt.execution_key == key,
                    ToolExecutionReceipt.user_id == scope.user_id,
                    ToolExecutionReceipt.conversation_id == scope.conversation_id,
                )
            )
            .scalars()
            .first()
        )
        if row is None:
            logger.warning("Execution key %s... is held by another owner", key[:12])
            return ReceiptRecord(execution_key=key, status=ReceiptStatus.RESERVED, fresh=True)

        result = dict(row.result_json) if isinstance(row.result_json, dict) else None
        if result is not None and row.artifact_ref:
            result.setdefault("artifact_ref", row.artifact_ref)
        return ReceiptRecord(
            execution_key=key,
            status=ReceiptStatus(row.status),
            fresh=False,
            result=result,
            provider_receipt_id=row.provider_receipt_id,
        )

    # ------------------------------------------------------------------
    # terminal transitions
    # ------------------------------------------------------------------

    async def acomplete(
        self,
        *,
        key: str,
        scope: Any,
        result: dict[str, Any] | None,
        provider_receipt_id: str | None,
    ) -> None:
        """Record the effect. Only the owner's reserved row may complete."""
        await self._atransition(
            key=key,
            scope=scope,
            expected=(ReceiptStatus.RESERVED,),
            values={
                "status": ReceiptStatus.COMPLETED,
                "result_json": result,
                "artifact_ref": (result or {}).get("artifact_ref"),
                "provider_receipt_id": provider_receipt_id,
                "completed_at": datetime.now(UTC),
            },
        )

    async def afail(self, *, key: str, scope: Any, error_code: str) -> None:
        """Record that the provider never accepted the call."""
        await self._atransition(
            key=key,
            scope=scope,
            expected=(ReceiptStatus.RESERVED,),
            values={"status": ReceiptStatus.FAILED, "error_code": str(error_code)[:128]},
        )

    async def amark_outcome_unknown(self, *, key: str, scope: Any) -> None:
        """Record that nobody can say whether the effect happened."""
        await self._atransition(
            key=key,
            scope=scope,
            expected=(ReceiptStatus.RESERVED,),
            values={"status": ReceiptStatus.OUTCOME_UNKNOWN},
        )

    async def _atransition(
        self,
        *,
        key: str,
        scope: Any,
        expected: tuple[ReceiptStatus, ...],
        values: dict[str, Any],
    ) -> None:
        """Compare-and-set on the caller's own row.

        Owner-scoped like every read here. A caller whose key is held by
        another owner is told its reservation is fresh (see
        :meth:`_existing_record`), runs its own call, and then closes "its"
        receipt; matching on the key alone would write that outcome onto the
        other owner's row.
        """

        def work(session: Session) -> None:
            outcome = session.execute(
                update(ToolExecutionReceipt)
                .where(
                    ToolExecutionReceipt.execution_key == key,
                    ToolExecutionReceipt.user_id == scope.user_id,
                    ToolExecutionReceipt.conversation_id == scope.conversation_id,
                    ToolExecutionReceipt.status.in_(expected),
                )
                .values(**values)
            )
            session.commit()
            if outcome.rowcount == 0:
                # Already closed, or not this owner's row. Not an error either
                # way: overwriting a recorded outcome, or someone else's, is
                # exactly what must not happen.
                logger.info(
                    "Receipt %s... was already terminal or not owned; %s not applied",
                    key[:12],
                    values.get("status"),
                )

        await self._arun(work)
