"""Atomic persistence for forking a conversation through one completed reply.

A fork copies the source transcript prefix into new rows owned by a new child
conversation. Every read and write happens in ONE fresh PostgreSQL transaction
opened after the caller holds the source's turn lock:

* ``REPEATABLE READ`` is set before the first query, so the source settings,
  the prefix, its lifecycle rows, agent links and media all come from one
  snapshot. Message edits and deletes do not take the turn lock, so without
  one snapshot the size check, the eligibility check and the copy could each
  see a different transcript.
* the source conversation row is locked ``FOR UPDATE`` under a transaction-
  local ``lock_timeout``, which coordinates with its concurrent update/delete
  and turns a stuck holder into a bounded, retriable conflict.
* a serialization failure (``40001``) restarts the whole transaction on a new
  session, because a repeatable-read snapshot cannot be refreshed in place.

The pre-lock replay lookup (:meth:`ConversationForkRepository.apreflight`)
uses its own short session that is closed before the lock is requested, so no
pre-lock snapshot can leak into the fork.
"""

from __future__ import annotations

import enum
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Text, cast, func, insert, select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session, lazyload

from app.models.chat_image import ChatImage
from app.models.conversation import Conversation
from app.models.custom_agent import ConversationCustomAgent, CustomAgent
from app.models.document import Document
from app.models.enums import MessageRole
from app.models.generation import ACTIVE_STATUSES, Generation, GenerationStatus
from app.models.message import Message
from app.models.project import Project
from app.models.web_image_reference import WebImageReference
from app.repositories.conversation_compaction import (
    ConversationCompactionRepository,
    is_hidden_transcript_artifact,
)
from app.repositories.session_transport import RepositorySessionMixin
from app.services.conversation_fork_metadata import (
    CHAT_IMAGE,
    FORK_CUTOFF_ELIGIBLE_KEY,
    FORK_HISTORY_KEY,
    WEB_IMAGE,
    MediaReference,
    drop_unavailable_media,
    media_references,
    snapshot_fork_message,
)

_FORK_SUFFIX = " (fork)"
_DEFAULT_TITLE_SOURCE_CHARS = 248

_SERIALIZATION_FAILURE = "40001"
_LOCK_NOT_AVAILABLE = "55P03"
_UNIQUE_VIOLATION = "23505"
_FORK_REQUEST_CONSTRAINT = "uq_conversations_owner_fork_request"

#: Generation states that finished the turn. ``continuation_available`` and
#: the terminal assistant message decide whether *this* row is its answer.
_FINISHED_STATUSES = frozenset({GenerationStatus.COMPLETED, GenerationStatus.COMPLETED_PARTIAL})

#: Metadata that only an execution-owned reply carries. A legacy reply holding
#: any of these without a matching generation row cannot be verified as final.
_LIVE_HANDLE_KEYS = ("generation_id", "continuation_id", "logical_turn_id", "execution_epoch")
_UNFINISHED_FLAGS = ("paused", "stopped", "continuable")


class ForkErrorCode(str, enum.Enum):
    """Every refusal the fork can produce; the service maps each to HTTP."""

    SOURCE_NOT_FOUND = "CONVERSATION_NOT_FOUND"
    SOURCE_ACCESS_DENIED = "CONVERSATION_ACCESS_DENIED"
    MESSAGE_NOT_FOUND = "FORK_MESSAGE_NOT_FOUND"
    POINT_NOT_COMPLETED = "FORK_POINT_NOT_COMPLETED"
    REQUEST_CONFLICT = "FORK_REQUEST_CONFLICT"
    RESULT_DELETED = "FORK_RESULT_DELETED"
    DOCUMENT_CONTEXT_UNSUPPORTED = "FORK_DOCUMENT_CONTEXT_UNSUPPORTED"
    HISTORY_TOO_LARGE = "FORK_HISTORY_TOO_LARGE"
    # Lowercase on purpose: the existing workflow code clients already handle.
    TURN_CONFLICT = "conversation_turn_conflict"


class ForkError(Exception):
    """A typed refusal. Never raised for an unrelated database failure."""

    def __init__(self, code: ForkErrorCode) -> None:
        super().__init__(code.value)
        self.code = code


@dataclass(frozen=True)
class ForkPlan:
    """Everything one fork attempt needs, resolved before the transaction."""

    source_conversation_id: UUID
    owner_id: UUID
    message_id: UUID
    request_id: UUID
    fingerprint: str
    title: str | None
    max_messages: int
    max_bytes: int
    lock_timeout_seconds: float


def default_fork_title(source_title: str) -> str:
    """The source title trimmed so the suffixed title fits ``String(255)``."""
    return f"{source_title[:_DEFAULT_TITLE_SOURCE_CHARS].rstrip()}{_FORK_SUFFIX}".strip()


def compute_fork_eligibility(rows: Sequence[Any], generations: Iterable[Any]) -> dict[UUID, bool]:
    """Whether each assistant row is a completed fork point.

    A durable generation is the authority: it must have finished
    (``completed``/``completed_partial``), offer no continuation, and name
    this row as its terminal assistant message -- otherwise an early partial
    epoch would pass because its turn eventually finished. A row copied by an
    earlier fork keeps the eligibility recorded when it was copied, because
    the ids that linked it to its generation were removed. Only a legacy reply
    with no generation is judged from its pause/stop metadata.
    """
    generations = list(generations)
    by_message = {g.assistant_message_id: g for g in generations if g.assistant_message_id}
    by_id = {g.id: g for g in generations}
    by_turn = {str(g.logical_turn_id): g for g in generations if g.logical_turn_id}
    return {
        row.id: _row_is_fork_point(row, by_message, by_id, by_turn)
        for row in rows
        if row.sender == MessageRole.assistant.value
    }


def _row_is_fork_point(
    row: Any,
    by_message: dict[Any, Any],
    by_id: dict[Any, Any],
    by_turn: dict[str, Any],
) -> bool:
    if is_hidden_transcript_artifact(row) or not (row.content or "").strip():
        return False
    metadata = row.message_metadata if isinstance(row.message_metadata, dict) else {}
    if metadata.get(FORK_HISTORY_KEY) is True:
        return metadata.get(FORK_CUTOFF_ELIGIBLE_KEY) is True
    generation = _matching_generation(row.id, metadata, by_message, by_id, by_turn)
    if generation is not None:
        return (
            generation.status in _FINISHED_STATUSES
            and not generation.continuation_available
            and generation.assistant_message_id == row.id
        )
    if bool(metadata.get("interrupt")) or any(metadata.get(k) for k in _UNFINISHED_FLAGS):
        return False
    return not any(metadata.get(key) is not None for key in _LIVE_HANDLE_KEYS)


def _matching_generation(
    message_id: UUID,
    metadata: dict[str, Any],
    by_message: dict[Any, Any],
    by_id: dict[Any, Any],
    by_turn: dict[str, Any],
) -> Any:
    """Assistant message id first, then generation id, then the turn identity."""
    if message_id in by_message:
        return by_message[message_id]
    generation_id = _as_uuid(metadata.get("generation_id"))
    if generation_id in by_id:
        return by_id[generation_id]
    for key in ("logical_turn_id", "reply_to_user_message_id"):
        value = metadata.get(key)
        if isinstance(value, str) and value in by_turn:
            return by_turn[value]
    return None


def _as_uuid(value: Any) -> UUID | None:
    if not isinstance(value, str):
        return None
    try:
        return UUID(value)
    except ValueError:
        return None


def _sqlstate(exc: DBAPIError) -> str | None:
    original = getattr(exc, "orig", None)
    return getattr(original, "sqlstate", None) or getattr(original, "pgcode", None)


def _constraint_name(exc: DBAPIError) -> str | None:
    diag = getattr(getattr(exc, "orig", None), "diag", None)
    return getattr(diag, "constraint_name", None)


class ConversationForkRepository(RepositorySessionMixin):
    """Owns the fork's preflight lookup and its single snapshot transaction."""

    def __init__(
        self,
        session_factory: Callable[[], Any],
        async_session_factory: Callable[[], Any] | None = None,
        *,
        max_serialization_attempts: int = 3,
    ) -> None:
        super().__init__(
            session_factory=session_factory,
            async_session_factory=async_session_factory,
        )
        self.max_serialization_attempts = max(1, int(max_serialization_attempts))

    # -- public API ---------------------------------------------------------

    async def apreflight(self, *, plan: ForkPlan) -> Conversation | None:
        """Return a replayed child, or check the caller may lock the source.

        Runs on its own short session, closed before the caller requests the
        turn lock. A replay is answered even if the source has since been
        deleted: the child is an independent conversation.
        """
        return await self._arun_refusable(lambda session: self._preflight(session, plan))

    async def afork_snapshot(self, *, plan: ForkPlan) -> tuple[Conversation, bool]:
        """Create the child in one transaction; ``(child, created)``.

        ``created`` is False when the request ID already produced a child
        (a replay found inside the transaction or after losing an insert race).
        """
        for attempt in range(1, self.max_serialization_attempts + 1):
            try:
                return await self._arun_refusable(
                    lambda session: self._fork_in_transaction(session, plan)
                )
            except DBAPIError as exc:
                state = _sqlstate(exc)
                if state == _SERIALIZATION_FAILURE and attempt < self.max_serialization_attempts:
                    continue
                if state in (_SERIALIZATION_FAILURE, _LOCK_NOT_AVAILABLE):
                    raise ForkError(ForkErrorCode.TURN_CONFLICT) from exc
                if state == _UNIQUE_VIOLATION and (
                    _constraint_name(exc) == _FORK_REQUEST_CONSTRAINT
                ):
                    return await self._areplay_after_contention(plan, exc)
                raise
        raise AssertionError("unreachable: the loop returns or raises")  # pragma: no cover

    # -- preflight and replay -------------------------------------------------

    def _preflight(self, session: Session, plan: ForkPlan) -> Conversation | None:
        existing = self._find_existing_child(session, plan)
        if existing is not None:
            return self._replayed(session, existing, plan)
        source = session.execute(
            select(Conversation.owner_id, Conversation.deleted_at).where(
                Conversation.id == plan.source_conversation_id
            )
        ).one_or_none()
        self._require_owned_source(source, plan)
        return None

    @staticmethod
    def _find_existing_child(session: Session, plan: ForkPlan) -> Conversation | None:
        return session.execute(
            select(Conversation).where(
                Conversation.owner_id == plan.owner_id,
                Conversation.fork_request_id == plan.request_id,
            )
        ).scalar_one_or_none()

    @staticmethod
    def _replayed(session: Session, child: Conversation, plan: ForkPlan) -> Conversation:
        if child.fork_request_fingerprint != plan.fingerprint:
            raise ForkError(ForkErrorCode.REQUEST_CONFLICT)
        if child.deleted_at is not None:
            raise ForkError(ForkErrorCode.RESULT_DELETED)
        session.expunge(child)
        return child

    async def _areplay_after_contention(
        self, plan: ForkPlan, original: DBAPIError
    ) -> tuple[Conversation, bool]:
        """Another worker committed this request ID first; answer with its child."""

        def read_winner(session: Session) -> Conversation:
            winner = self._find_existing_child(session, plan)
            if winner is None:
                raise original
            return self._replayed(session, winner, plan)

        return await self._arun_refusable(read_winner), False

    async def _arun_refusable(self, work: Callable[[Session], Any]) -> Any:
        """Run ``work`` on a fresh session; a refusal leaves it as a value, not an error.

        The shared async session context logs every exception escaping it as a
        failed rollback, which would turn each expected 4xx into an error
        trace. A refusal is returned out of the session instead and raised
        here; closing the session rolls back whatever was left uncommitted.
        """

        def capture(session: Session) -> tuple[Any, ForkError | None]:
            try:
                return work(session), None
            except ForkError as refusal:
                return None, refusal

        result, refusal = await self._arun(capture)
        if refusal is not None:
            raise refusal
        return result

    @staticmethod
    def _require_owned_source(source: Any, plan: ForkPlan) -> None:
        if source is None or source.deleted_at is not None:
            raise ForkError(ForkErrorCode.SOURCE_NOT_FOUND)
        if source.owner_id != plan.owner_id:
            raise ForkError(ForkErrorCode.SOURCE_ACCESS_DENIED)

    # -- the transaction --------------------------------------------------------

    def _fork_in_transaction(self, session: Session, plan: ForkPlan) -> tuple[Conversation, bool]:
        self._begin_repeatable_read(session, plan)
        existing = self._find_existing_child(session, plan)
        if existing is not None:
            return self._replayed(session, existing, plan), False
        source = self._lock_source(session, plan)
        cutoff = self._load_cutoff(session, plan)
        self._refuse_active_generation(session, plan)
        self._refuse_documents(session, plan)
        self._refuse_oversized_prefix(session, plan, cutoff.sequence)
        rows = self._load_prefix(session, plan, cutoff.sequence)
        eligibility = compute_fork_eligibility(rows, self._load_generations(session, plan))
        if not eligibility.get(cutoff.id):
            raise ForkError(ForkErrorCode.POINT_NOT_COMPLETED)

        child = self._insert_child(session, plan, source, cutoff)
        self._copy_agent_links(session, plan, child.id)
        available = self._available_media(session, plan, rows)
        self._insert_messages(session, child.id, rows, eligibility, available)
        self._enqueue_summary_job(session, child.id, cutoff.sequence)
        session.commit()
        session.expunge(child)
        return child, True

    @staticmethod
    def _begin_repeatable_read(session: Session, plan: ForkPlan) -> None:
        """Isolation must be set before the first query of the transaction."""
        session.connection(execution_options={"isolation_level": "REPEATABLE READ"})
        timeout_ms = max(1, int(plan.lock_timeout_seconds * 1000))
        # SET cannot take bind parameters; ``timeout_ms`` is an int.
        session.execute(text(f"SET LOCAL lock_timeout = {timeout_ms}"))

    def _lock_source(self, session: Session, plan: ForkPlan) -> Conversation:
        # Membership writers and project deletion lock the project before the
        # conversation. A fork must follow that order before its child's FK
        # takes a project lock, otherwise deletion and forking can deadlock.
        candidate = session.execute(
            select(Conversation).where(Conversation.id == plan.source_conversation_id)
        ).scalar_one_or_none()
        self._require_owned_source(candidate, plan)
        if candidate.project_id is not None:
            project = session.execute(
                select(Project.id)
                .where(
                    Project.id == candidate.project_id,
                    Project.owner_id == plan.owner_id,
                    Project.deleted_at.is_(None),
                )
                .with_for_update(read=True)
            ).scalar_one_or_none()
            if project is None:
                raise ForkError(ForkErrorCode.TURN_CONFLICT)
        source = session.execute(
            select(Conversation)
            .where(Conversation.id == plan.source_conversation_id)
            .with_for_update()
        ).scalar_one_or_none()
        self._require_owned_source(source, plan)
        return source

    @staticmethod
    def _load_cutoff(session: Session, plan: ForkPlan) -> Message:
        cutoff = session.execute(
            select(Message)
            .options(lazyload(Message.feedback))
            .where(
                Message.id == plan.message_id,
                Message.conversation_id == plan.source_conversation_id,
                Message.deleted_at.is_(None),
            )
        ).scalar_one_or_none()
        if cutoff is None:
            raise ForkError(ForkErrorCode.MESSAGE_NOT_FOUND)
        if cutoff.sender != MessageRole.assistant.value:
            raise ForkError(ForkErrorCode.POINT_NOT_COMPLETED)
        return cutoff

    @staticmethod
    def _refuse_active_generation(session: Session, plan: ForkPlan) -> None:
        """An active row, even an abandoned one, is a running turn; the reaper owns it."""
        active = session.execute(
            select(Generation.id)
            .where(
                Generation.conversation_id == plan.source_conversation_id,
                Generation.status.in_(list(ACTIVE_STATUSES)),
            )
            .limit(1)
        ).scalar_one_or_none()
        if active is not None:
            raise ForkError(ForkErrorCode.TURN_CONFLICT)

    @staticmethod
    def _refuse_documents(session: Session, plan: ForkPlan) -> None:
        """Document access is conversation-scoped; copied ids would grant nothing."""
        document = session.execute(
            select(Document.id)
            .where(Document.conversation_id == plan.source_conversation_id)
            .limit(1)
        ).scalar_one_or_none()
        if document is not None:
            raise ForkError(ForkErrorCode.DOCUMENT_CONTEXT_UNSUPPORTED)

    @staticmethod
    def _refuse_oversized_prefix(session: Session, plan: ForkPlan, cutoff_sequence: int) -> None:
        """Count and size the prefix in SQL before any metadata is materialized."""
        metadata_bytes = func.coalesce(func.octet_length(cast(Message.message_metadata, Text)), 0)
        count, size = session.execute(
            select(
                func.count(),
                func.coalesce(func.sum(func.octet_length(Message.content) + metadata_bytes), 0),
            ).where(
                Message.conversation_id == plan.source_conversation_id,
                Message.deleted_at.is_(None),
                Message.sequence <= cutoff_sequence,
            )
        ).one()
        if int(count) > plan.max_messages or int(size) > plan.max_bytes:
            raise ForkError(ForkErrorCode.HISTORY_TOO_LARGE)

    @staticmethod
    def _load_prefix(session: Session, plan: ForkPlan, cutoff_sequence: int) -> list[Message]:
        rows = session.execute(
            select(Message)
            .options(lazyload(Message.feedback))
            .where(
                Message.conversation_id == plan.source_conversation_id,
                Message.deleted_at.is_(None),
                Message.sequence <= cutoff_sequence,
            )
            .order_by(Message.sequence)
        ).scalars()
        return [row for row in rows if not is_hidden_transcript_artifact(row)]

    @staticmethod
    def _load_generations(session: Session, plan: ForkPlan) -> list[Any]:
        """Every lifecycle row of the source in one query (no per-message lookups)."""
        return list(
            session.execute(
                select(
                    Generation.id,
                    Generation.status,
                    Generation.assistant_message_id,
                    Generation.continuation_available,
                    Generation.logical_turn_id,
                ).where(
                    Generation.conversation_id == plan.source_conversation_id,
                    Generation.user_id == plan.owner_id,
                )
            ).all()
        )

    @staticmethod
    def _insert_child(
        session: Session, plan: ForkPlan, source: Conversation, cutoff: Message
    ) -> Conversation:
        now = datetime.now(UTC)
        child = Conversation(
            id=uuid.uuid4(),
            created_at=now,
            updated_at=now,
            deleted_at=None,
            owner_id=plan.owner_id,
            title=plan.title if plan.title is not None else default_fork_title(source.title),
            persona_prompt=source.persona_prompt,
            planning_mode_enabled=source.planning_mode_enabled,
            project_id=source.project_id,
            plan_lifecycle=None,
            next_message_sequence=int(cutoff.sequence) + 1,
            forked_from_conversation_id=source.id,
            forked_from_message_id=cutoff.id,
            forked_through_sequence=int(cutoff.sequence),
            fork_request_id=plan.request_id,
            fork_request_fingerprint=plan.fingerprint,
        )
        session.add(child)
        # Flushed first: agent links, messages and the summary job reference it.
        session.flush()
        return child

    @staticmethod
    def _copy_agent_links(session: Session, plan: ForkPlan, child_id: UUID) -> None:
        """Current live attachments of the owner's agents; project defaults are not re-seeded."""
        links = session.execute(
            select(ConversationCustomAgent.custom_agent_id, ConversationCustomAgent.agent_order)
            .join(CustomAgent, CustomAgent.id == ConversationCustomAgent.custom_agent_id)
            .where(
                ConversationCustomAgent.conversation_id == plan.source_conversation_id,
                ConversationCustomAgent.owner_id == plan.owner_id,
                CustomAgent.owner_id == plan.owner_id,
                CustomAgent.deleted_at.is_(None),
            )
            .order_by(ConversationCustomAgent.agent_order, ConversationCustomAgent.created_at)
        ).all()
        if not links:
            return
        session.execute(
            insert(ConversationCustomAgent.__table__),
            [
                {
                    "id": uuid.uuid4(),
                    "created_at": datetime.now(UTC),
                    "owner_id": plan.owner_id,
                    "conversation_id": child_id,
                    "custom_agent_id": link.custom_agent_id,
                    "agent_order": link.agent_order,
                }
                for link in links
            ],
        )

    @staticmethod
    def _available_media(
        session: Session, plan: ForkPlan, rows: Sequence[Message]
    ) -> set[MediaReference]:
        """Which referenced images the owner still owns: live chat images, selected web images."""
        references: set[MediaReference] = set()
        for row in rows:
            references |= media_references(row.message_metadata)
        chat_ids = [asset for kind, asset in references if kind == CHAT_IMAGE]
        web_ids = [asset for kind, asset in references if kind == WEB_IMAGE]
        available: set[MediaReference] = set()
        if chat_ids:
            owned = session.execute(
                select(ChatImage.id).where(
                    ChatImage.id.in_(chat_ids),
                    ChatImage.user_id == plan.owner_id,
                    ChatImage.deleted_at.is_(None),
                )
            ).scalars()
            available |= {(CHAT_IMAGE, asset) for asset in owned}
        if web_ids:
            selected = session.execute(
                select(WebImageReference.id).where(
                    WebImageReference.id.in_(web_ids),
                    WebImageReference.user_id == plan.owner_id,
                    WebImageReference.deleted_at.is_(None),
                    WebImageReference.lifecycle_state == "selected",
                )
            ).scalars()
            available |= {(WEB_IMAGE, asset) for asset in selected}
        return available

    @staticmethod
    def _insert_messages(
        session: Session,
        child_id: UUID,
        rows: Sequence[Message],
        eligibility: dict[UUID, bool],
        available: set[MediaReference],
    ) -> None:
        # The whole map exists before any metadata is transformed, so a reply
        # can point at a user message regardless of iteration order.
        id_map = {str(row.id): uuid.uuid4() for row in rows}
        text_map = {source: str(target) for source, target in id_map.items()}
        values = []
        for row in rows:
            content, metadata = snapshot_fork_message(
                content=row.content, metadata=row.message_metadata, message_id_map=text_map
            )
            metadata = drop_unavailable_media(content, metadata, available=available)
            if row.id in eligibility:
                metadata[FORK_CUTOFF_ELIGIBLE_KEY] = eligibility[row.id]
            values.append(
                {
                    "id": id_map[str(row.id)],
                    "created_at": row.created_at,
                    "updated_at": row.updated_at,
                    "deleted_at": None,
                    "conversation_id": child_id,
                    "sender": row.sender,
                    "content": content,
                    "message_metadata": metadata,
                    "sequence": row.sequence,
                }
            )
        session.execute(insert(Message.__table__), values)

    @staticmethod
    def _enqueue_summary_job(session: Session, child_id: UUID, cutoff_sequence: int) -> None:
        """A fresh job through the cutoff; its cursor FK needs the copied rows first."""
        ConversationCompactionRepository.enqueue_job_in_session(
            session,
            conversation_id=child_id,
            requested_through_sequence=int(cutoff_sequence),
        )


__all__ = [
    "ConversationForkRepository",
    "ForkError",
    "ForkErrorCode",
    "ForkPlan",
    "compute_fork_eligibility",
    "default_fork_title",
]
