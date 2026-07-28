"""Transactional persistence and durable state for conversation compaction."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, NamedTuple
from uuid import UUID, uuid4

from sqlalchemy import and_, case, exists, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, lazyload

from app.models.conversation import Conversation
from app.models.conversation_memory_summary import ConversationMemorySummary
from app.models.conversation_summary_job import ConversationSummaryJob, SummaryJobStatus
from app.models.enums import MessageRole
from app.models.message import Message
from app.repositories.session_transport import RepositorySessionMixin


class SummaryJobClaim(NamedTuple):
    """A committed worker lease and its immutable captured target."""

    conversation_id: UUID
    requested_through_sequence: int
    lease_token: UUID
    attempt_count: int = 0


@dataclass(frozen=True)
class CompactionInput:
    """Detached authoritative input loaded after a worker commits its lease."""

    claim: SummaryJobClaim
    owner_id: UUID
    summary_payload: dict[str, Any] | None
    summary_version: int
    last_summarized_sequence: int | None
    messages: tuple[Message, ...]


class ConversationCompactionRepository(RepositorySessionMixin):
    """Own every short transaction in the compaction durability protocol."""

    def __init__(
        self,
        session_factory: Callable[[], Session],
        async_session_factory: Callable[[], Any] | None = None,
    ):
        super().__init__(
            session_factory=session_factory,
            async_session_factory=async_session_factory,
        )

    @staticmethod
    def _utcnow() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _job_upsert_statement(
        *,
        conversation_id: UUID,
        requested_through_sequence: int,
        now: datetime,
    ):
        table = ConversationSummaryJob.__table__
        incoming = insert(table).values(
            conversation_id=conversation_id,
            requested_through_sequence=requested_through_sequence,
            status=SummaryJobStatus.PENDING.value,
            attempt_count=0,
            available_at=now,
            lease_token=None,
            lease_expires_at=None,
            last_error_code=None,
            updated_at=now,
        )
        live_lease = and_(
            table.c.status == SummaryJobStatus.PROCESSING.value,
            table.c.lease_token.is_not(None),
            table.c.lease_expires_at > now,
        )
        return incoming.on_conflict_do_update(
            index_elements=[table.c.conversation_id],
            set_={
                "requested_through_sequence": func.greatest(
                    table.c.requested_through_sequence,
                    incoming.excluded.requested_through_sequence,
                ),
                "status": case(
                    (live_lease, table.c.status),
                    else_=SummaryJobStatus.PENDING.value,
                ),
                "attempt_count": case((live_lease, table.c.attempt_count), else_=0),
                "available_at": case((live_lease, table.c.available_at), else_=now),
                "lease_token": case((live_lease, table.c.lease_token), else_=None),
                "lease_expires_at": case((live_lease, table.c.lease_expires_at), else_=None),
                "last_error_code": case((live_lease, table.c.last_error_code), else_=None),
                "updated_at": now,
            },
        )

    @staticmethod
    def _claim_select_statement(
        *,
        conversation_id: UUID | None,
        now: datetime,
    ):
        statement = select(ConversationSummaryJob).where(
            ConversationSummaryJob.status.in_(
                [SummaryJobStatus.PENDING.value, SummaryJobStatus.RETRY.value]
            )
        )
        if conversation_id is None:
            statement = statement.where(ConversationSummaryJob.available_at <= now)
        else:
            statement = statement.where(ConversationSummaryJob.conversation_id == conversation_id)
        return (
            statement.order_by(
                ConversationSummaryJob.available_at,
                ConversationSummaryJob.created_at,
            )
            .limit(1)
            .with_for_update(skip_locked=True)
        )

    @staticmethod
    def _memory_cas_update_statement(
        *,
        claim: SummaryJobClaim,
        base_summary_version: int,
        base_cursor: int | None,
        summary_payload: dict[str, Any],
        summary_schema_version: int,
        source_message_count: int,
        source_token_count: int,
        summary_token_count: int,
        provider: str,
        model: str,
        tokenizer: str,
        prompt_version: str,
        last_summarized_sequence: int | None = None,
        now: datetime | None = None,
    ):
        written_at = now or ConversationCompactionRepository._utcnow()
        lease_is_owned = exists(
            select(ConversationSummaryJob.conversation_id).where(
                ConversationSummaryJob.conversation_id == claim.conversation_id,
                ConversationSummaryJob.status == SummaryJobStatus.PROCESSING.value,
                ConversationSummaryJob.lease_token == claim.lease_token,
                ConversationSummaryJob.lease_expires_at > written_at,
            )
        )
        return (
            update(ConversationMemorySummary)
            .where(
                ConversationMemorySummary.conversation_id == claim.conversation_id,
                ConversationMemorySummary.summary_version == base_summary_version,
                ConversationMemorySummary.last_summarized_sequence.is_not_distinct_from(
                    base_cursor
                ),
                lease_is_owned,
            )
            .values(
                summary_payload=summary_payload,
                summary_schema_version=summary_schema_version,
                last_summarized_sequence=(
                    claim.requested_through_sequence
                    if last_summarized_sequence is None
                    else last_summarized_sequence
                ),
                summary_version=ConversationMemorySummary.summary_version + 1,
                source_message_count=source_message_count,
                source_token_count=source_token_count,
                summary_token_count=summary_token_count,
                provider=provider,
                model=model,
                tokenizer=tokenizer,
                prompt_version=prompt_version,
                is_valid=True,
                updated_at=written_at,
            )
        )

    def _persist_message_in_session(
        self, session: Session, message_data: Mapping[str, Any]
    ) -> Message:
        """Allocate a sequence, insert the message, and request assistant work once.

        Session-taking body shared by :meth:`persist_message` and
        :meth:`apersist_message`. Runs unchanged under ``AsyncSession.run_sync``.
        """
        data = dict(message_data)
        conversation_id = data["conversation_id"]
        data.pop("sequence", None)
        now = self._utcnow()

        allocated = session.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(next_message_sequence=Conversation.next_message_sequence + 1)
            .returning(Conversation.next_message_sequence - 1)
        ).scalar_one_or_none()
        if allocated is None:
            session.rollback()
            raise ValueError("conversation_not_found")

        message = Message(**data, sequence=int(allocated), feedback=None)
        session.add(message)
        session.flush()
        if message.sender == MessageRole.assistant.value:
            session.execute(
                self._job_upsert_statement(
                    conversation_id=conversation_id,
                    requested_through_sequence=message.sequence,
                    now=now,
                )
            )
        session.commit()
        session.expunge(message)
        return message

    def persist_message(self, message_data: Mapping[str, Any]) -> Message:
        """Allocate a sequence, insert a message, and request assistant work once."""
        return self._run(lambda session: self._persist_message_in_session(session, message_data))

    async def apersist_message(self, message_data: Mapping[str, Any]) -> Message:
        """Async twin of :meth:`persist_message`."""
        return await self._arun(
            lambda session: self._persist_message_in_session(session, message_data)
        )

    def request_backfill(
        self,
        conversation_id: UUID,
        requested_through_sequence: int | None = None,
    ) -> bool:
        """Coalesce a rebuild request at an assistant boundary."""
        now = self._utcnow()
        with self.session_factory() as session:
            target = requested_through_sequence
            if target is None:
                target = session.execute(
                    select(func.max(Message.sequence)).where(
                        Message.conversation_id == conversation_id,
                        Message.sender == MessageRole.assistant.value,
                        Message.deleted_at.is_(None),
                    )
                ).scalar_one()
            if target is None:
                return False
            existing = session.get(ConversationSummaryJob, conversation_id)
            if (
                existing is not None
                and existing.requested_through_sequence >= target
                and existing.status
                in {
                    SummaryJobStatus.PENDING.value,
                    SummaryJobStatus.PROCESSING.value,
                    SummaryJobStatus.RETRY.value,
                }
            ):
                return False
            session.execute(
                self._job_upsert_statement(
                    conversation_id=conversation_id,
                    requested_through_sequence=int(target),
                    now=now,
                )
            )
            session.commit()
            return True

    def list_backfill_candidates(self, *, limit: int = 100) -> list[tuple[UUID, int]]:
        """Return historical conversations whose durable target is absent or stale."""
        assistant_targets = (
            select(
                Message.conversation_id.label("conversation_id"),
                func.max(Message.sequence).label("target"),
            )
            .where(
                Message.sender == MessageRole.assistant.value,
                Message.deleted_at.is_(None),
            )
            .group_by(Message.conversation_id)
            .subquery()
        )
        with self.session_factory() as session:
            rows = session.execute(
                select(assistant_targets.c.conversation_id, assistant_targets.c.target)
                .join(
                    Conversation,
                    Conversation.id == assistant_targets.c.conversation_id,
                )
                .outerjoin(
                    ConversationSummaryJob,
                    ConversationSummaryJob.conversation_id == assistant_targets.c.conversation_id,
                )
                .outerjoin(
                    ConversationMemorySummary,
                    ConversationMemorySummary.conversation_id
                    == assistant_targets.c.conversation_id,
                )
                .where(
                    Conversation.deleted_at.is_(None),
                    or_(
                        ConversationSummaryJob.conversation_id.is_(None),
                        ConversationSummaryJob.requested_through_sequence
                        < assistant_targets.c.target,
                        and_(
                            ConversationMemorySummary.conversation_id.is_not(None),
                            ConversationMemorySummary.is_valid.is_(False),
                            ConversationSummaryJob.status.in_(
                                [
                                    SummaryJobStatus.IDLE.value,
                                    SummaryJobStatus.DEAD.value,
                                ]
                            ),
                        ),
                    ),
                )
                .order_by(Conversation.created_at, Conversation.id)
                .limit(limit)
            ).all()
            return [(row.conversation_id, int(row.target)) for row in rows]

    def claim_job(
        self,
        conversation_id: UUID | None = None,
        *,
        lease_seconds: int,
        now: datetime | None = None,
    ) -> SummaryJobClaim | None:
        """Claim one job and commit its lease before returning."""
        claimed_at = now or self._utcnow()
        with self.session_factory() as session:
            job = session.execute(
                self._claim_select_statement(
                    conversation_id=conversation_id,
                    now=claimed_at,
                )
            ).scalar_one_or_none()
            if job is None:
                return None
            token = uuid4()
            job.status = SummaryJobStatus.PROCESSING.value
            job.lease_token = token
            job.lease_expires_at = claimed_at + timedelta(seconds=lease_seconds)
            job.updated_at = claimed_at
            target = int(job.requested_through_sequence)
            session.commit()
            return SummaryJobClaim(
                job.conversation_id,
                target,
                token,
                int(job.attempt_count),
            )

    def load_compaction_input(
        self,
        claim: SummaryJobClaim,
        *,
        owner_id: UUID | None = None,
    ) -> CompactionInput | None:
        """Load detached transcript state only for the active lease and owner."""
        now = self._utcnow()
        with self.session_factory() as session:
            conversation_statement = select(Conversation).where(
                Conversation.id == claim.conversation_id
            )
            if owner_id is not None:
                conversation_statement = conversation_statement.where(
                    Conversation.owner_id == owner_id
                )
            conversation = session.execute(conversation_statement).scalar_one_or_none()
            if conversation is None:
                return None
            lease_exists = session.execute(
                select(ConversationSummaryJob.conversation_id).where(
                    ConversationSummaryJob.conversation_id == claim.conversation_id,
                    ConversationSummaryJob.status == SummaryJobStatus.PROCESSING.value,
                    ConversationSummaryJob.lease_token == claim.lease_token,
                    ConversationSummaryJob.lease_expires_at > now,
                )
            ).scalar_one_or_none()
            if lease_exists is None:
                return None

            memory = session.get(ConversationMemorySummary, claim.conversation_id)
            cursor = (
                int(memory.last_summarized_sequence)
                if memory is not None
                and memory.is_valid
                and memory.last_summarized_sequence is not None
                else None
            )
            clauses = [
                Message.conversation_id == claim.conversation_id,
                Message.sequence <= claim.requested_through_sequence,
                Message.deleted_at.is_(None),
            ]
            if cursor is not None:
                clauses.append(Message.sequence > cursor)
            rows = list(
                session.execute(select(Message).where(*clauses).order_by(Message.sequence))
                .scalars()
                .all()
            )
            rows = [row for row in rows if not self._is_hidden_artifact(row)]
            for row in rows:
                session.expunge(row)
            return CompactionInput(
                claim=claim,
                owner_id=conversation.owner_id,
                summary_payload=(
                    dict(memory.summary_payload) if memory is not None and memory.is_valid else None
                ),
                summary_version=int(memory.summary_version) if memory is not None else 0,
                last_summarized_sequence=cursor,
                messages=tuple(rows),
            )

    def persist_memory_cas(
        self,
        claim: SummaryJobClaim,
        *,
        base_summary_version: int,
        base_cursor: int | None,
        summary_payload: dict[str, Any],
        summary_schema_version: int,
        last_summarized_sequence: int,
        source_message_count: int,
        source_token_count: int,
        summary_token_count: int,
        provider: str,
        model: str,
        tokenizer: str,
        prompt_version: str,
    ) -> bool:
        """Write validated memory only if cursor, version, and lease still match."""
        if last_summarized_sequence > claim.requested_through_sequence:
            return False
        now = self._utcnow()
        with self.session_factory() as session:
            current = session.get(ConversationMemorySummary, claim.conversation_id)
            if current is None:
                lease = session.execute(
                    select(ConversationSummaryJob.conversation_id).where(
                        ConversationSummaryJob.conversation_id == claim.conversation_id,
                        ConversationSummaryJob.status == SummaryJobStatus.PROCESSING.value,
                        ConversationSummaryJob.lease_token == claim.lease_token,
                        ConversationSummaryJob.lease_expires_at > now,
                    )
                ).scalar_one_or_none()
                if lease is None or base_summary_version != 0 or base_cursor is not None:
                    return False
                session.add(
                    ConversationMemorySummary(
                        conversation_id=claim.conversation_id,
                        summary_payload=summary_payload,
                        summary_schema_version=summary_schema_version,
                        last_summarized_sequence=last_summarized_sequence,
                        summary_version=1,
                        source_message_count=source_message_count,
                        source_token_count=source_token_count,
                        summary_token_count=summary_token_count,
                        provider=provider,
                        model=model,
                        tokenizer=tokenizer,
                        prompt_version=prompt_version,
                        is_valid=True,
                        updated_at=now,
                    )
                )
                session.commit()
                return True

            result = session.execute(
                self._memory_cas_update_statement(
                    claim=claim,
                    base_summary_version=base_summary_version,
                    base_cursor=base_cursor,
                    summary_payload=summary_payload,
                    summary_schema_version=summary_schema_version,
                    last_summarized_sequence=last_summarized_sequence,
                    source_message_count=source_message_count,
                    source_token_count=source_token_count,
                    summary_token_count=summary_token_count,
                    provider=provider,
                    model=model,
                    tokenizer=tokenizer,
                    prompt_version=prompt_version,
                    now=now,
                )
            )
            if result.rowcount != 1:
                session.rollback()
                return False
            session.commit()
            return True

    def complete_claim(self, claim: SummaryJobClaim) -> str | None:
        """Release an owned lease to idle or pending without losing a newer target."""
        now = self._utcnow()
        with self.session_factory() as session:
            job = session.execute(
                select(ConversationSummaryJob)
                .where(
                    ConversationSummaryJob.conversation_id == claim.conversation_id,
                    ConversationSummaryJob.status == SummaryJobStatus.PROCESSING.value,
                    ConversationSummaryJob.lease_token == claim.lease_token,
                    ConversationSummaryJob.lease_expires_at > now,
                )
                .with_for_update()
            ).scalar_one_or_none()
            if job is None:
                return None
            caught_up = job.requested_through_sequence <= claim.requested_through_sequence
            job.status = (
                SummaryJobStatus.IDLE.value if caught_up else SummaryJobStatus.PENDING.value
            )
            job.available_at = now
            job.attempt_count = 0
            job.lease_token = None
            job.lease_expires_at = None
            job.last_error_code = None
            job.updated_at = now
            status = job.status
            session.commit()
            return status

    def fail_claim(
        self,
        claim: SummaryJobClaim,
        *,
        error_code: str,
        permanent: bool,
        retry_at: datetime | None = None,
        max_attempts: int = 5,
    ) -> bool:
        """Transition an owned lease to bounded retry or dead state."""
        now = self._utcnow()
        safe_code = self._sanitize_error_code(error_code)
        with self.session_factory() as session:
            job = session.execute(
                select(ConversationSummaryJob)
                .where(
                    ConversationSummaryJob.conversation_id == claim.conversation_id,
                    ConversationSummaryJob.status == SummaryJobStatus.PROCESSING.value,
                    ConversationSummaryJob.lease_token == claim.lease_token,
                    ConversationSummaryJob.lease_expires_at > now,
                )
                .with_for_update()
            ).scalar_one_or_none()
            if job is None:
                return False
            job.attempt_count += 1
            exhausted = job.attempt_count >= max_attempts
            job.status = (
                SummaryJobStatus.DEAD.value
                if permanent or exhausted
                else SummaryJobStatus.RETRY.value
            )
            job.available_at = retry_at or now
            job.lease_token = None
            job.lease_expires_at = None
            job.last_error_code = safe_code
            job.updated_at = now
            session.commit()
            return True

    def reconcile_due_jobs(
        self,
        *,
        now: datetime | None = None,
        limit: int = 100,
        dispatch_debounce_seconds: int = 60,
        max_attempts: int = 5,
    ) -> list[UUID]:
        """Recover expired leases and reserve due rows for best-effort dispatch."""
        reconciled_at = now or self._utcnow()
        with self.session_factory() as session:
            expired = list(
                session.execute(
                    select(ConversationSummaryJob)
                    .where(
                        ConversationSummaryJob.status == SummaryJobStatus.PROCESSING.value,
                        ConversationSummaryJob.lease_expires_at <= reconciled_at,
                    )
                    .with_for_update(skip_locked=True)
                )
                .scalars()
                .all()
            )
            for job in expired:
                job.attempt_count += 1
                job.status = (
                    SummaryJobStatus.DEAD.value
                    if job.attempt_count >= max_attempts
                    else SummaryJobStatus.RETRY.value
                )
                job.available_at = reconciled_at
                job.lease_token = None
                job.lease_expires_at = None
                job.last_error_code = "lease_expired"
                job.updated_at = reconciled_at

            due = list(
                session.execute(
                    select(ConversationSummaryJob)
                    .where(
                        ConversationSummaryJob.status.in_(
                            [
                                SummaryJobStatus.PENDING.value,
                                SummaryJobStatus.RETRY.value,
                            ]
                        ),
                        ConversationSummaryJob.available_at <= reconciled_at,
                    )
                    .order_by(ConversationSummaryJob.available_at)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
                .scalars()
                .all()
            )
            debounce_until = reconciled_at + timedelta(seconds=dispatch_debounce_seconds)
            conversation_ids = [job.conversation_id for job in due]
            for job in due:
                job.available_at = debounce_until
                job.updated_at = reconciled_at
            session.commit()
            return conversation_ids

    def get_owned_valid_memory(
        self,
        conversation_id: UUID,
        owner_id: UUID,
    ) -> ConversationMemorySummary | None:
        """Return valid memory only through the authoritative ownership join."""
        with self.session_factory() as session:
            memory = session.execute(
                select(ConversationMemorySummary)
                .join(
                    Conversation,
                    Conversation.id == ConversationMemorySummary.conversation_id,
                )
                .where(
                    ConversationMemorySummary.conversation_id == conversation_id,
                    ConversationMemorySummary.is_valid.is_(True),
                    Conversation.owner_id == owner_id,
                    Conversation.deleted_at.is_(None),
                )
            ).scalar_one_or_none()
            if memory is not None:
                session.expunge(memory)
            return memory

    def get_compaction_health_snapshot(self) -> dict[str, Any]:
        """Return aggregate queue, lease, and sequence-lag health without IDs."""
        now = self._utcnow()
        with self.session_factory() as session:
            count_rows = session.execute(
                select(ConversationSummaryJob.status, func.count()).group_by(
                    ConversationSummaryJob.status
                )
            ).all()
            job_counts = {str(status): int(count) for status, count in count_rows}
            oldest_updated = session.execute(
                select(func.min(ConversationSummaryJob.updated_at)).where(
                    ConversationSummaryJob.status.in_(
                        [
                            SummaryJobStatus.PENDING.value,
                            SummaryJobStatus.RETRY.value,
                        ]
                    )
                )
            ).scalar_one()
            expired_lease_count = int(
                session.execute(
                    select(func.count())
                    .select_from(ConversationSummaryJob)
                    .where(
                        ConversationSummaryJob.status == SummaryJobStatus.PROCESSING.value,
                        ConversationSummaryJob.lease_expires_at <= now,
                    )
                ).scalar_one()
                or 0
            )
            lag = func.greatest(
                ConversationSummaryJob.requested_through_sequence
                - func.coalesce(ConversationMemorySummary.last_summarized_sequence, 0),
                0,
            )
            max_lag, total_lag = session.execute(
                select(func.coalesce(func.max(lag), 0), func.coalesce(func.sum(lag), 0))
                .select_from(ConversationSummaryJob)
                .outerjoin(
                    ConversationMemorySummary,
                    ConversationMemorySummary.conversation_id
                    == ConversationSummaryJob.conversation_id,
                )
            ).one()
        oldest_age = 0.0
        if oldest_updated is not None:
            if oldest_updated.tzinfo is None:
                oldest_updated = oldest_updated.replace(tzinfo=timezone.utc)
            oldest_age = max(0.0, (now - oldest_updated).total_seconds())
        return {
            "job_counts": job_counts,
            "oldest_actionable_age_seconds": oldest_age,
            "expired_lease_count": expired_lease_count,
            "max_sequence_lag": int(max_lag or 0),
            "total_sequence_lag": int(total_lag or 0),
        }

    def invalidate_for_mutation(
        self,
        message_id: UUID,
        *,
        new_content: str | None = None,
        delete: bool = False,
    ) -> bool:
        """Mutate a message and atomically invalidate covered derived memory."""
        if delete == (new_content is not None):
            raise ValueError("exactly_one_mutation_required")
        now = self._utcnow()
        with self.session_factory() as session:
            message = session.execute(
                select(Message)
                .options(lazyload(Message.feedback))
                .where(Message.id == message_id)
                .with_for_update()
            ).scalar_one_or_none()
            if message is None or message.deleted_at is not None:
                return False
            if delete:
                message.deleted_at = now
            else:
                message.content = new_content
            message.updated_at = now

            memory = session.execute(
                select(ConversationMemorySummary)
                .where(ConversationMemorySummary.conversation_id == message.conversation_id)
                .with_for_update()
            ).scalar_one_or_none()
            covered = (
                memory is not None
                and memory.is_valid
                and memory.last_summarized_sequence is not None
                and message.sequence <= memory.last_summarized_sequence
            )
            if covered:
                memory.summary_payload = {}
                memory.last_summarized_sequence = None
                memory.is_valid = False
                memory.summary_version += 1
                memory.updated_at = now
                target = session.execute(
                    select(func.max(Message.sequence)).where(
                        Message.conversation_id == message.conversation_id,
                        Message.sender == MessageRole.assistant.value,
                        Message.deleted_at.is_(None),
                    )
                ).scalar_one()
                if target is not None:
                    session.execute(
                        self._job_upsert_statement(
                            conversation_id=message.conversation_id,
                            requested_through_sequence=int(target),
                            now=now,
                        )
                    )
                    session.execute(
                        update(ConversationSummaryJob)
                        .where(ConversationSummaryJob.conversation_id == message.conversation_id)
                        .values(
                            status=SummaryJobStatus.PENDING.value,
                            attempt_count=0,
                            available_at=now,
                            lease_token=None,
                            lease_expires_at=None,
                            last_error_code=None,
                            updated_at=now,
                        )
                    )
            session.commit()
            return True

    @staticmethod
    def _is_hidden_artifact(message: Message) -> bool:
        if message.sender != MessageRole.assistant.value:
            return False
        if (message.content or "").strip():
            return False
        metadata = message.message_metadata or {}
        return metadata.get("paused") is True or bool(metadata.get("interrupt"))

    @staticmethod
    def _sanitize_error_code(error_code: str) -> str:
        sanitized = re.sub(r"[^a-z0-9_.-]+", "_", error_code.lower()).strip("_")
        return (sanitized or "unknown_error")[:64]
