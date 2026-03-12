"""Repository for durable HITL interrupt lifecycle records."""

import logging
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select, update

from app.models.hitl_interrupt import HITLInterrupt, HITLInterruptStatus

logger = logging.getLogger(__name__)


class HITLInterruptRepository:
    """CRUD and lifecycle management for HITL interrupt records."""

    def __init__(self, session_factory: callable):
        self.session_factory = session_factory

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def create(
        self,
        interrupt_id: str,
        conversation_id: UUID,
        user_id: UUID,
        thread_id: str,
        expires_at: datetime,
        action_requests_json: list,
        assistant_message_id: UUID | None = None,
    ) -> HITLInterrupt:
        """Persist a new interrupt session in PENDING state."""
        record = HITLInterrupt(
            id=interrupt_id,
            conversation_id=conversation_id,
            user_id=user_id,
            thread_id=thread_id,
            assistant_message_id=assistant_message_id,
            status=HITLInterruptStatus.PENDING,
            expires_at=expires_at,
            action_requests_json=action_requests_json,
        )
        with self.session_factory() as db:
            db.add(record)
            db.commit()
            db.refresh(record)
            return record

    def try_transition_to_resolving(
        self,
        interrupt_id: str,
        conversation_id: UUID,
        resolved_by_user_id: UUID,
    ) -> bool:
        """
        Atomically move the interrupt from PENDING to RESOLVING.

        Returns True if this caller won the race (first-write-wins).
        Returns False if the interrupt was already resolved, resolving,
        expired, or not found.
        """
        now = datetime.now(timezone.utc)
        with self.session_factory() as db:
            result = db.execute(
                update(HITLInterrupt)
                .where(
                    HITLInterrupt.id == interrupt_id,
                    HITLInterrupt.conversation_id == conversation_id,
                    HITLInterrupt.status == HITLInterruptStatus.PENDING,
                    HITLInterrupt.expires_at > now,
                )
                .values(
                    status=HITLInterruptStatus.RESOLVING,
                    resolved_by_user_id=resolved_by_user_id,
                    updated_at=now,
                )
            )
            db.commit()
            return result.rowcount == 1

    def mark_resolved(
        self,
        interrupt_id: str,
        resolution_source: str = "user",
    ) -> None:
        """Mark an interrupt as fully resolved after graph resumption."""
        now = datetime.now(timezone.utc)
        with self.session_factory() as db:
            db.execute(
                update(HITLInterrupt)
                .where(
                    HITLInterrupt.id == interrupt_id,
                    HITLInterrupt.status.in_(
                        [
                            HITLInterruptStatus.PENDING,
                            HITLInterruptStatus.RESOLVING,
                        ]
                    ),
                )
                .values(
                    status=HITLInterruptStatus.RESOLVED,
                    resolved_at=now,
                    resolution_source=resolution_source,
                    updated_at=now,
                )
            )
            db.commit()

    def mark_expired(self, interrupt_id: str) -> None:
        """Mark an interrupt as expired."""
        now = datetime.now(timezone.utc)
        with self.session_factory() as db:
            db.execute(
                update(HITLInterrupt)
                .where(
                    HITLInterrupt.id == interrupt_id,
                    HITLInterrupt.status == HITLInterruptStatus.PENDING,
                )
                .values(
                    status=HITLInterruptStatus.EXPIRED,
                    resolution_source="timeout",
                    updated_at=now,
                )
            )
            db.commit()

    def update_assistant_message_id(self, interrupt_id: str, assistant_message_id: UUID) -> None:
        """Attach the persisted assistant message ID to the interrupt record."""
        with self.session_factory() as db:
            db.execute(
                update(HITLInterrupt)
                .where(HITLInterrupt.id == interrupt_id)
                .values(
                    assistant_message_id=assistant_message_id,
                    updated_at=datetime.now(timezone.utc),
                )
            )
            db.commit()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_by_id(self, interrupt_id: str) -> HITLInterrupt | None:
        """Look up an interrupt record by its ID."""
        with self.session_factory() as db:
            return db.get(HITLInterrupt, interrupt_id)

    def get_pending_by_conversation(self, conversation_id: UUID) -> list[HITLInterrupt]:
        """Return all PENDING interrupt records for a conversation."""
        with self.session_factory() as db:
            stmt = select(HITLInterrupt).where(
                HITLInterrupt.conversation_id == conversation_id,
                HITLInterrupt.status == HITLInterruptStatus.PENDING,
            )
            return list(db.execute(stmt).scalars().all())

    def get_expired_pending(self, now: datetime) -> list[HITLInterrupt]:
        """Return PENDING records whose expiry time has passed."""
        with self.session_factory() as db:
            stmt = select(HITLInterrupt).where(
                HITLInterrupt.status == HITLInterruptStatus.PENDING,
                HITLInterrupt.expires_at <= now,
            )
            return list(db.execute(stmt).scalars().all())
