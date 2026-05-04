"""Repository for durable per-conversation memory summaries."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager, suppress
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.conversation_memory_summary import ConversationMemorySummary
from app.models.message import Message

logger = logging.getLogger(__name__)


class ConversationMemorySummaryRepository:
    """Single-row-per-conversation upsert and lookup."""

    def __init__(self, session_factory: Callable[[], AbstractContextManager[Session]]):
        self.session_factory = session_factory

    def get_by_conversation_id(self, conversation_id: UUID) -> ConversationMemorySummary | None:
        with self.session_factory() as session:
            stmt = select(ConversationMemorySummary).where(
                ConversationMemorySummary.conversation_id == conversation_id
            )
            row = session.execute(stmt).scalar_one_or_none()
            if row is not None:
                session.expunge(row)
            return row

    def upsert(
        self,
        *,
        conversation_id: UUID,
        user_id: UUID,
        summary_text: str,
        last_summarized_message_id: UUID | None,
        source_message_count: int,
        estimated_tokens: int,
    ) -> ConversationMemorySummary:
        """Insert or update the per-conversation summary row.

        Race-safe by:
        * locking the existing row with ``SELECT ... FOR UPDATE`` so concurrent
          updates serialize on Postgres;
        * catching ``IntegrityError`` on first-insert races (two workers reach
          INSERT before either commits) and retrying the read+update path so
          the second writer updates instead of duplicating;
        * refusing to regress the DB-message cursor by comparing the existing
          and incoming ``last_summarized_message_id`` rows by ``(created_at, id)``.
        """
        for attempt in range(2):
            try:
                return self._upsert_once(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    summary_text=summary_text,
                    last_summarized_message_id=last_summarized_message_id,
                    source_message_count=source_message_count,
                    estimated_tokens=estimated_tokens,
                )
            except IntegrityError as exc:
                # Concurrent INSERT lost the unique-constraint race. Retry once;
                # the SELECT will now find the winning row and update it.
                if attempt == 0:
                    logger.info(
                        "Conversation summary upsert retry after IntegrityError "
                        "for conversation=%s",
                        conversation_id,
                    )
                    continue
                raise exc
        # Unreachable: the loop either returns or re-raises.
        raise RuntimeError("upsert retry exhausted")

    def _upsert_once(
        self,
        *,
        conversation_id: UUID,
        user_id: UUID,
        summary_text: str,
        last_summarized_message_id: UUID | None,
        source_message_count: int,
        estimated_tokens: int,
    ) -> ConversationMemorySummary:
        with self.session_factory() as session:
            stmt = select(ConversationMemorySummary).where(
                ConversationMemorySummary.conversation_id == conversation_id
            )
            with suppress(Exception):
                # Postgres row lock; harmless on engines that ignore the hint.
                stmt = stmt.with_for_update()
            row = session.execute(stmt).scalar_one_or_none()
            if row is None:
                row = ConversationMemorySummary(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    summary_version=0,
                )
                session.add(row)
            else:
                # Out-of-order completion: a later task already wrote a more
                # complete cursor. Leave it alone instead of regressing. The
                # source count is telemetry only; ordering is determined by the
                # message cursor itself because token-triggered refreshes can
                # summarize fewer rows while still advancing farther.
                if self._incoming_cursor_regresses(
                    session,
                    existing_message_id=row.last_summarized_message_id,
                    incoming_message_id=last_summarized_message_id,
                ):
                    logger.info(
                        "Skipping summary upsert that would regress cursor for conversation=%s "
                        "(existing=%s, incoming=%s)",
                        conversation_id,
                        row.last_summarized_message_id,
                        last_summarized_message_id,
                    )
                    session.expunge(row)
                    return row

            row.summary_text = summary_text
            row.last_summarized_message_id = last_summarized_message_id
            row.source_message_count = source_message_count
            row.estimated_tokens = estimated_tokens
            row.summary_version = (row.summary_version or 0) + 1

            session.commit()
            session.refresh(row)
            session.expunge(row)
            return row

    def _incoming_cursor_regresses(
        self,
        session: Session,
        *,
        existing_message_id: UUID | None,
        incoming_message_id: UUID | None,
    ) -> bool:
        """Return True when ``incoming_message_id`` is older than the stored cursor."""
        if existing_message_id is None:
            return False
        if incoming_message_id is None:
            return True
        if incoming_message_id == existing_message_id:
            return False

        positions = self._load_cursor_positions(
            session,
            [existing_message_id, incoming_message_id],
        )
        existing_position = positions.get(existing_message_id)
        incoming_position = positions.get(incoming_message_id)

        # The FK should make missing rows impossible. If the incoming cursor
        # cannot be verified, prefer preserving the existing durable summary.
        if incoming_position is None:
            logger.warning(
                "Incoming summary cursor message %s could not be loaded; preserving existing cursor %s",
                incoming_message_id,
                existing_message_id,
            )
            return True
        if existing_position is None:
            logger.warning(
                "Existing summary cursor message %s could not be loaded; accepting incoming cursor %s",
                existing_message_id,
                incoming_message_id,
            )
            return False

        return incoming_position < existing_position

    @staticmethod
    def _load_cursor_positions(
        session: Session,
        message_ids: list[UUID],
    ) -> dict[UUID, tuple]:
        unique_ids = list(dict.fromkeys(message_ids))
        stmt = select(Message).where(Message.id.in_(unique_ids))
        rows = session.execute(stmt).scalars().all()
        return {row.id: (row.created_at, row.id) for row in rows}
