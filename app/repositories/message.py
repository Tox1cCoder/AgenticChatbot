import logging
from collections.abc import Callable
from typing import Any
from uuid import UUID

from sqlalchemy import asc, desc, func, select
from sqlalchemy.orm import Session, joinedload

from app.factories.message_factory import MessageFactory
from app.models.enums import MessageRole
from app.models.message import Message
from app.repositories.command_strategy import DefaultCommandStrategy
from app.repositories.conversation_compaction import ConversationCompactionRepository
from app.repositories.query_strategy import DefaultQueryStrategy
from app.repositories.session_transport import RepositorySessionMixin
from app.repositories.utils.pagination import Paginator
from app.schemas.message import MessageCreate, MessageUpdate
from app.utils.validation.pagination_validation import validate_pagination_params

logger = logging.getLogger(__name__)


class MessageCRUDStrategy(
    DefaultCommandStrategy[Message, MessageCreate, MessageUpdate],
    DefaultQueryStrategy[Message],
):
    """Custom CRUD strategy for Message operations"""

    def __init__(self, model: type[Message]):
        DefaultCommandStrategy.__init__(self, model)
        DefaultQueryStrategy.__init__(self, model)

    def get_by_conversation_id(
        self,
        db: Session,
        conversation_id: UUID,
        page: int = 1,
        limit: int = 10,
        order_by: str = "created_at",
        order_direction: str = "asc",
        include_feedback: bool = False,
    ) -> Paginator[Message]:
        """Get messages by conversation ID with page-based pagination and ordering"""

        validate_pagination_params(page, limit)

        # Get total count first
        total = self.count_by_conversation_id(db, conversation_id)

        # Get paginated items
        offset = (page - 1) * limit
        statement = select(Message).where(
            Message.conversation_id == conversation_id,
            Message.deleted_at.is_(None),
        )

        # Apply eager loading if requested
        if include_feedback:
            statement = statement.options(joinedload(Message.feedback))

        # Apply ordering if specified. ``order_by`` is Optional at the
        # repository boundary, so the None check must come first — this mirrors
        # DefaultQueryStrategy, which this class overrides.
        if order_by and hasattr(Message, order_by):
            order_column = getattr(Message, order_by)
            statement = statement.order_by(
                asc(order_column) if order_direction.lower() == "asc" else desc(order_column)
            )
        else:
            # Default ordering
            statement = statement.order_by(Message.created_at.asc())

        statement = statement.offset(offset).limit(limit)

        # Use unique() when eager loading to handle joined loads
        if include_feedback:
            items = list(db.execute(statement).scalars().unique().all())
        else:
            items = list(db.execute(statement).scalars().all())

        return Paginator.create(items, total, page, limit)

    def count_by_conversation_id(self, db: Session, conversation_id: UUID) -> int:
        """Count messages by conversation ID using SQL ``COUNT`` and excluding soft-deleted rows."""
        statement = select(func.count(Message.id)).where(
            Message.conversation_id == conversation_id,
            Message.deleted_at.is_(None),
        )
        return int(db.execute(statement).scalar() or 0)

    def get_by_user_id(
        self,
        db: Session,
        user_id: UUID,
        page: int = 1,
        limit: int = 10,
        order_by: str = "created_at",
        order_direction: str = "desc",
        include_feedback: bool = False,
    ) -> Paginator[Message]:
        """Get messages by conversation owner (user_id) with page-based pagination and ordering"""

        validate_pagination_params(page, limit)

        # Get total count first
        total = self.count_by_user_id(db, user_id)

        # Get paginated items
        offset = (page - 1) * limit
        # Join with conversations to get messages from user's conversations
        statement = (
            select(Message)
            .join(Message.conversation)
            .where(
                Message.conversation.has(owner_id=user_id),
                Message.deleted_at.is_(None),
            )
        )

        # Apply eager loading if requested
        if include_feedback:
            statement = statement.options(joinedload(Message.feedback))

        # Apply ordering if specified. ``order_by`` is Optional at the
        # repository boundary, so the None check must come first — this mirrors
        # DefaultQueryStrategy, which this class overrides.
        if order_by and hasattr(Message, order_by):
            order_column = getattr(Message, order_by)
            statement = statement.order_by(
                asc(order_column) if order_direction.lower() == "asc" else desc(order_column)
            )
        else:
            # Default ordering
            statement = statement.order_by(Message.created_at.asc())

        statement = statement.offset(offset).limit(limit)

        # Use unique() when eager loading to handle joined loads
        if include_feedback:
            items = list(db.execute(statement).scalars().unique().all())
        else:
            items = list(db.execute(statement).scalars().all())

        return Paginator.create(items, total, page, limit)

    def count_by_user_id(self, db: Session, user_id: UUID) -> int:
        """Count an owner's non-deleted messages using SQL ``COUNT``."""
        statement = (
            select(func.count(Message.id))
            .join(Message.conversation)
            .where(
                Message.conversation.has(owner_id=user_id),
                Message.deleted_at.is_(None),
            )
        )
        return int(db.execute(statement).scalar() or 0)

    def get_prompt_history(
        self,
        db: Session,
        conversation_id: UUID,
        *,
        before_message_id: UUID | None = None,
        after_message_id: UUID | None = None,
        after_sequence: int | None = None,
        limit: int | None = 50,
    ) -> list[Message]:
        """Return prompt-eligible messages for a conversation in ascending order.

        Always filters out soft-deleted rows. When a cursor is supplied, the
        immutable sequence of that anchor message is used to bound the result
        strictly before/after the cursor. The newest ``limit`` rows in
        the window are selected (DESC + limit) and then returned ASC so prompt
        history reads naturally from oldest to newest.

        ``limit=None`` (or ``0``) disables the cap. Empty paused or interrupted
        assistant placeholders are filtered in Python — JSONB metadata
        predicates differ across dialects and the candidate set is small.
        """

        before_anchor = self._lookup_sequence(db, before_message_id) if before_message_id else None
        after_anchor = self._lookup_sequence(db, after_message_id) if after_message_id else None

        clauses = [
            Message.conversation_id == conversation_id,
            Message.deleted_at.is_(None),
        ]

        if before_anchor is not None:
            clauses.append(Message.sequence < before_anchor)

        if after_anchor is not None:
            clauses.append(Message.sequence > after_anchor)

        if after_sequence is not None:
            clauses.append(Message.sequence > after_sequence)

        statement = select(Message).where(*clauses).order_by(Message.sequence.desc())
        if limit is not None and limit > 0:
            statement = statement.limit(limit)

        rows = list(db.execute(statement).scalars().all())
        rows.reverse()
        return [row for row in rows if not self._is_hidden_artifact(row)]

    @staticmethod
    def _lookup_sequence(db: Session, message_id: UUID) -> int | None:
        statement = select(Message).where(Message.id == message_id)
        anchor = db.execute(statement).scalar_one_or_none()
        return int(anchor.sequence) if anchor is not None else None

    @staticmethod
    def _is_hidden_artifact(message: Message) -> bool:
        """Empty paused/interrupt assistant placeholders are not real transcript turns."""
        if message.sender != MessageRole.assistant.value:
            return False
        if (message.content or "").strip():
            return False
        metadata = message.message_metadata or {}
        if metadata.get("paused") is True:
            return True
        return bool(metadata.get("interrupt"))

    def search_by_content(
        self,
        db: Session,
        conversation_id: UUID,
        query: str,
        limit: int = 10,
    ) -> list[Message]:
        """
        Search messages by content in a specific conversation.

        Args:
            db: Database session
            conversation_id: ID of the conversation to search in
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of matching messages ordered by relevance (most recent first)
        """
        # Use case-insensitive pattern matching
        search_pattern = f"%{query}%"

        statement = (
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.content.ilike(search_pattern),
            )
            .order_by(Message.created_at.desc())
            .limit(limit)
        )

        return list(db.execute(statement).scalars().all())

    def get_canvas_artifact_candidates(
        self,
        db: Session,
        conversation_id: UUID,
        *,
        limit: int = 20,
    ) -> list[Message]:
        """Return newest assistant rows that advertise canvas metadata."""
        statement = (
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.sender == MessageRole.assistant.value,
                Message.deleted_at.is_(None),
                Message.message_metadata.op("?")("canvas_artifact"),
            )
            .order_by(Message.sequence.desc())
            .limit(max(1, limit))
        )
        return list(db.execute(statement).scalars().all())

    def get_latest_assistant_by_conversation(
        self,
        db: Session,
        conversation_id: UUID,
    ) -> Message | None:
        statement = (
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.sender == MessageRole.assistant.value,
                Message.deleted_at.is_(None),
            )
            .order_by(Message.sequence.desc())
            .limit(1)
        )
        return db.execute(statement).scalars().first()


class MessageRepository(RepositorySessionMixin):
    """Repository for Message model using session factory pattern"""

    def __init__(
        self,
        session_factory: callable,
        compaction_repository: ConversationCompactionRepository | None = None,
        compaction_publisher: Callable[[UUID], Any] | None = None,
        async_session_factory: Callable[[], Any] | None = None,
    ):
        """Initialize repository with session factory for dependency injection."""
        super().__init__(
            session_factory=session_factory,
            async_session_factory=async_session_factory,
        )
        self._crud_strategy = MessageCRUDStrategy(Message)
        # The fallback delegate must be async-capable too, or acreate() would
        # raise for any caller that did not inject a compaction repository.
        self._compaction_repository = compaction_repository or ConversationCompactionRepository(
            session_factory,
            async_session_factory=async_session_factory,
        )
        self._compaction_publisher = compaction_publisher

    def get_by_conversation_id(
        self,
        conversation_id: UUID,
        page: int = 1,
        limit: int = 10,
        order_by: str | None = None,
        order_direction: str = "asc",
        include_feedback: bool = False,
    ) -> Paginator[Message]:
        """Get messages by conversation ID with page-based pagination and ordering"""
        with self.session_factory() as session:
            return self._crud_strategy.get_by_conversation_id(
                session,
                conversation_id,
                page,
                limit,
                order_by,
                order_direction,
                include_feedback,
            )

    async def aget_by_conversation_id(
        self,
        conversation_id: UUID,
        page: int = 1,
        limit: int = 10,
        order_by: str | None = None,
        order_direction: str = "asc",
        include_feedback: bool = False,
    ) -> Paginator[Message]:
        """Async twin of :meth:`get_by_conversation_id`."""
        return await self._arun(
            lambda session: self._crud_strategy.get_by_conversation_id(
                session,
                conversation_id,
                page,
                limit,
                order_by,
                order_direction,
                include_feedback,
            )
        )

    def count_by_conversation_id(self, conversation_id: UUID) -> int:
        """Count messages by conversation ID"""
        with self.session_factory() as session:
            return self._crud_strategy.count_by_conversation_id(session, conversation_id)

    async def acount_by_conversation_id(self, conversation_id: UUID) -> int:
        """Async twin of :meth:`count_by_conversation_id`."""
        return await self._arun(
            lambda session: self._crud_strategy.count_by_conversation_id(session, conversation_id)
        )

    def get_by_user_id(
        self,
        user_id: UUID,
        page: int = 1,
        limit: int = 10,
        order_by: str | None = None,
        order_direction: str = "desc",
        include_feedback: bool = False,
    ) -> Paginator[Message]:
        """Get messages by user ID with page-based pagination and ordering"""
        with self.session_factory() as session:
            return self._crud_strategy.get_by_user_id(
                session,
                user_id,
                page,
                limit,
                order_by,
                order_direction,
                include_feedback,
            )

    def count_by_user_id(self, user_id: UUID) -> int:
        """Count messages by user ID"""
        with self.session_factory() as session:
            return self._crud_strategy.count_by_user_id(session, user_id)

    def create(self, input_schema: MessageCreate) -> Message:
        """Atomically allocate a sequence and persist durable assistant work."""
        if isinstance(input_schema, dict):
            message_data = input_schema
        else:
            message_data = MessageFactory.create_from_schema_with_role(
                input_schema,
                input_schema.role,
            )
        message = self._compaction_repository.persist_message(message_data)
        self._publish_compaction_notification(message)
        return message

    async def acreate(self, input_schema: MessageCreate) -> Message:
        """Async twin of :meth:`create`.

        Building ``message_data`` touches no database, so only the persistence
        step is awaited; the post-commit notification stays outside the
        transaction exactly as in the sync path.
        """
        if isinstance(input_schema, dict):
            message_data = input_schema
        else:
            message_data = MessageFactory.create_from_schema_with_role(
                input_schema,
                input_schema.role,
            )
        message = await self._compaction_repository.apersist_message(message_data)
        self._publish_compaction_notification(message)
        return message

    def _publish_compaction_notification(self, message: Message) -> None:
        """Request compaction for a durable assistant message, best effort."""
        if message.sender != MessageRole.assistant.value or self._compaction_publisher is None:
            return
        try:
            self._compaction_publisher(message.conversation_id)
        except Exception:
            # The message and coalesced database job are already committed.
            # Reconciliation recovers a lost notification.
            logger.warning("Conversation compaction notification failed code=broker_publish_failed")

    def get_by_id(self, id: UUID) -> Message | None:
        """Get message by ID"""
        with self.session_factory() as session:
            return self._crud_strategy.get_by_id(session, id)

    async def aget_by_id(self, id: UUID) -> Message | None:
        """Async twin of :meth:`get_by_id`."""
        return await self._arun(lambda session: self._crud_strategy.get_by_id(session, id))

    def get_all(self, page: int = 1, limit: int = 10) -> list[Message]:
        """Get all messages with page-based pagination"""
        with self.session_factory() as session:
            return self._crud_strategy.get_all(session, page, limit)

    def update(self, id: UUID, input_schema: MessageUpdate) -> Message | None:
        """Update a message and invalidate covered memory in one transaction."""
        content = input_schema.content
        if content is None or not self._compaction_repository.invalidate_for_mutation(
            id,
            new_content=content,
        ):
            return None
        return self.get_by_id(id)

    def delete(self, id: UUID) -> bool:
        """Soft-delete a message and invalidate covered memory atomically."""
        return self._compaction_repository.invalidate_for_mutation(id, delete=True)

    def get_prompt_history(
        self,
        conversation_id: UUID,
        *,
        before_message_id: UUID | None = None,
        after_message_id: UUID | None = None,
        after_sequence: int | None = None,
        limit: int | None = 50,
    ) -> list[Message]:
        """Return prompt-eligible messages for a conversation.

        Excludes soft-deleted rows and empty paused/interrupt assistant
        placeholders. Uses immutable sequence cursor positioning so the anchor
        message itself is never included. The newest ``limit`` rows in
        the window are selected; pass ``None`` or ``0`` to disable the cap.
        """
        with self.session_factory() as session:
            return self._crud_strategy.get_prompt_history(
                session,
                conversation_id,
                before_message_id=before_message_id,
                after_message_id=after_message_id,
                after_sequence=after_sequence,
                limit=limit,
            )

    @staticmethod
    def _latest_by_conversation_statement(conversation_id: UUID):
        return (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(1)
        )

    def get_latest_by_conversation(self, conversation_id: UUID) -> Message | None:
        """Retrieve the most recent message in a conversation."""
        with self.session_factory() as session:
            statement = self._latest_by_conversation_statement(conversation_id)
            return session.execute(statement).scalars().first()

    async def aget_latest_by_conversation(self, conversation_id: UUID) -> Message | None:
        """Async twin of :meth:`get_latest_by_conversation`."""
        return await self._arun(
            lambda session: (
                session.execute(self._latest_by_conversation_statement(conversation_id))
                .scalars()
                .first()
            )
        )

    def get_canvas_artifact_candidates(
        self,
        conversation_id: UUID,
        *,
        limit: int = 20,
    ) -> list[Message]:
        with self.session_factory() as session:
            return self._crud_strategy.get_canvas_artifact_candidates(
                session,
                conversation_id,
                limit=limit,
            )

    def get_latest_assistant_by_conversation(self, conversation_id: UUID) -> Message | None:
        with self.session_factory() as session:
            return self._crud_strategy.get_latest_assistant_by_conversation(
                session,
                conversation_id,
            )

    def exists(self, id: UUID) -> bool:
        """Check if message exists"""
        with self.session_factory() as session:
            return self._crud_strategy.exists(session, id)
