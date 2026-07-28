from collections.abc import Callable
from typing import Any
from uuid import UUID

from sqlalchemy import asc, case, desc, exists, func, or_, select
from sqlalchemy.orm import Session, joinedload

from app.models.conversation import Conversation
from app.models.enums import PlanLifecycle
from app.models.message import Message
from app.repositories.command_strategy import DefaultCommandStrategy
from app.repositories.query_strategy import DefaultQueryStrategy
from app.repositories.session_transport import RepositorySessionMixin
from app.repositories.utils.pagination import Paginator
from app.schemas.conversation import ConversationCreate, ConversationUpdate
from app.utils.validation.pagination_validation import validate_pagination_params


def _build_owned_conversation_queries(
    *,
    owner_id: UUID,
    page: int,
    limit: int,
    order_by: str,
    order_direction: str,
    search: str | None,
) -> tuple[Any, Any]:
    """Build matching count and page queries for one owner's conversations."""
    conditions = [
        Conversation.owner_id == owner_id,
        Conversation.deleted_at.is_(None),
    ]
    normalized_search = search.strip().lower() if isinstance(search, str) else ""

    if normalized_search:
        title = func.lower(Conversation.title)
        title_exact = title == normalized_search
        title_prefix = title.startswith(normalized_search, autoescape=True)
        title_contains = title.contains(normalized_search, autoescape=True)
        message_match = exists(
            select(Message.id).where(
                Message.conversation_id == Conversation.id,
                Message.deleted_at.is_(None),
                func.lower(Message.content).contains(normalized_search, autoescape=True),
            )
        ).correlate(Conversation)
        conditions.append(or_(title_contains, message_match))
        relevance = case(
            (title_exact, 0),
            (title_prefix, 1),
            (title_contains, 2),
            (message_match, 3),
            else_=4,
        )
        ordering = (relevance.asc(), Conversation.updated_at.desc(), Conversation.id.asc())
    else:
        order_column = getattr(Conversation, order_by, Conversation.updated_at)
        ordering = (asc(order_column) if order_direction.lower() == "asc" else desc(order_column),)

    count_statement = select(func.count(Conversation.id)).where(*conditions)
    page_statement = (
        select(Conversation)
        .where(*conditions)
        .order_by(*ordering)
        .offset((page - 1) * limit)
        .limit(limit)
    )
    return count_statement, page_statement


class ConversationCRUDStrategy(
    DefaultCommandStrategy[Conversation, ConversationCreate, ConversationUpdate],
    DefaultQueryStrategy[Conversation],
):
    """Custom CRUD strategy for Conversation operations"""

    def __init__(self, model: type[Conversation]):
        DefaultCommandStrategy.__init__(self, model)
        DefaultQueryStrategy.__init__(self, model)

    def get_by_owner_id(
        self,
        db: Session,
        owner_id: UUID,
        page: int = 1,
        limit: int = 10,
        order_by: str = "updated_at",
        order_direction: str = "desc",
        search: str | None = None,
    ) -> Paginator[Conversation]:
        """Get conversations by owner ID with page-based pagination and ordering"""

        validate_pagination_params(page, limit)
        count_statement, page_statement = _build_owned_conversation_queries(
            owner_id=owner_id,
            page=page,
            limit=limit,
            order_by=order_by,
            order_direction=order_direction,
            search=search,
        )
        total = int(db.execute(count_statement).scalar() or 0)
        items = list(db.execute(page_statement).scalars().all())

        return Paginator.create(items, total, page, limit)

    def count_by_owner_id(
        self,
        db: Session,
        owner_id: UUID,
        search: str | None = None,
    ) -> int:
        """Count conversations by owner ID"""
        count_statement, _ = _build_owned_conversation_queries(
            owner_id=owner_id,
            page=1,
            limit=1,
            order_by="updated_at",
            order_direction="desc",
            search=search,
        )
        return int(db.execute(count_statement).scalar() or 0)

    def get_with_messages(self, db: Session, conversation_id: UUID) -> Conversation | None:
        """Get conversation with its messages"""
        statement = (
            select(Conversation)
            .options(joinedload(Conversation.messages))
            .where(Conversation.id == conversation_id, Conversation.deleted_at.is_(None))
        )
        return db.execute(statement).unique().scalar_one_or_none()

    def get_with_recent_messages(
        self,
        db: Session,
        owner_id: UUID,
        latest_messages: int = 3,
        order_by: str = "updated_at",
        order_direction: str = "desc",
        page: int = 1,
        limit: int = 10,
        search: str | None = None,
    ) -> list[Conversation]:
        """Get conversations with limited recent messages and total message count"""
        _, page_statement = _build_owned_conversation_queries(
            owner_id=owner_id,
            page=page,
            limit=limit,
            order_by=order_by,
            order_direction=order_direction,
            search=search,
        )
        conversations = list(db.execute(page_statement).scalars().all())

        # Load recent messages and count total messages for each conversation
        for conversation in conversations:
            # Get total message count
            count_statement = select(func.count(Message.id)).where(
                Message.conversation_id == conversation.id
            )
            total_message_count = db.execute(count_statement).scalar() or 0

            # Get recent messages
            message_statement = (
                select(Message)
                .where(Message.conversation_id == conversation.id)
                .order_by(desc(Message.created_at))
                .limit(latest_messages)
            )
            recent_messages = list(db.execute(message_statement).scalars().all())
            # Reverse to get oldest first
            recent_messages_reversed = recent_messages[::-1]

            # Create detached copies of messages for the conversation
            messages_for_attribute = []
            for msg in recent_messages_reversed:
                messages_for_attribute.append(msg)

            # Expunge conversation from session first
            db.expunge(conversation)

            # Set messages and message count directly in __dict__
            conversation.__dict__["messages"] = messages_for_attribute
            conversation.__dict__["message_count"] = total_message_count

        return conversations

    def user_owns_conversation(self, db: Session, owner_id: UUID, conversation_id: UUID) -> bool:
        """Check if user owns the conversation"""
        statement = select(Conversation.id).where(
            Conversation.id == conversation_id,
            Conversation.owner_id == owner_id,
            Conversation.deleted_at.is_(None),
        )
        return db.execute(statement).scalar() is not None

    def get_soft_deleted(self, db: Session) -> list[Conversation]:
        """Return conversations that have been soft-deleted (deleted_at is set)."""
        statement = select(Conversation).where(Conversation.deleted_at.is_not(None))
        return list(db.execute(statement).scalars().all())


class ConversationRepository(RepositorySessionMixin):
    """Repository for Conversation model using session factory pattern"""

    def __init__(
        self,
        session_factory: callable,
        async_session_factory: Callable[[], Any] | None = None,
    ):
        """Initialize repository with session factory for dependency injection."""
        super().__init__(
            session_factory=session_factory,
            async_session_factory=async_session_factory,
        )
        self._crud_strategy = ConversationCRUDStrategy(Conversation)

    def get_by_owner_id(
        self,
        owner_id: UUID,
        page: int = 1,
        limit: int = 10,
        order_by: str = "updated_at",
        order_direction: str = "desc",
        include: list[str] = None,
        latest_messages: int = 3,
        search: str | None = None,
    ) -> Paginator[Conversation]:
        """Get conversations by owner ID with optional includes"""
        if include is None:
            include = []

        validate_pagination_params(page, limit)
        with self.session_factory() as session:
            if "messages" in include:
                conversations = self._crud_strategy.get_with_recent_messages(
                    session,
                    owner_id,
                    latest_messages,
                    order_by,
                    order_direction,
                    page,
                    limit,
                    search,
                )
                # Get total count for pagination
                total = self._crud_strategy.count_by_owner_id(session, owner_id, search)
                return Paginator.create(conversations, total, page, limit)
            else:
                return self._crud_strategy.get_by_owner_id(
                    session,
                    owner_id,
                    page,
                    limit,
                    order_by,
                    order_direction,
                    search,
                )

    def count_by_owner_id(self, owner_id: UUID, search: str | None = None) -> int:
        """Count conversations by owner ID"""
        with self.session_factory() as session:
            return self._crud_strategy.count_by_owner_id(session, owner_id, search)

    def get_with_messages(self, conversation_id: UUID) -> Conversation | None:
        """Get conversation with its messages"""
        with self.session_factory() as session:
            return self._crud_strategy.get_with_messages(session, conversation_id)

    def user_owns_conversation(self, owner_id: UUID, conversation_id: UUID) -> bool:
        """Check if user owns the conversation"""
        with self.session_factory() as session:
            return self._crud_strategy.user_owns_conversation(session, owner_id, conversation_id)

    async def auser_owns_conversation(self, owner_id: UUID, conversation_id: UUID) -> bool:
        """Async twin of :meth:`user_owns_conversation`."""
        return await self._arun(
            lambda session: self._crud_strategy.user_owns_conversation(
                session, owner_id, conversation_id
            )
        )

    def create(self, input_schema: ConversationCreate) -> Conversation:
        """Create a new conversation"""
        with self.session_factory() as session:
            return self._crud_strategy.create(session, input_schema)

    def get_by_id(self, id: UUID) -> Conversation | None:
        """Get conversation by ID"""
        with self.session_factory() as session:
            return self._crud_strategy.get_by_id(session, id)

    async def aget_by_id(self, id: UUID) -> Conversation | None:
        """Async twin of :meth:`get_by_id`."""
        return await self._arun(lambda session: self._crud_strategy.get_by_id(session, id))

    def get_all(self, page: int = 1, limit: int = 10) -> list[Conversation]:
        """Get all conversations with page-based pagination"""
        with self.session_factory() as session:
            return self._crud_strategy.get_all(session, page, limit)

    def _update_in_session(
        self, session: Session, id: UUID, input_schema: ConversationUpdate
    ) -> Conversation | None:
        """Load-then-update body shared by both transports.

        Both statements must share one transaction, which is why this is a
        single session-taking function rather than two calls.
        """
        db_obj = self._crud_strategy.get_by_id(session, id)
        if db_obj is None:
            return None
        return self._crud_strategy.update(session, db_obj, input_schema)

    def update(self, id: UUID, input_schema: ConversationUpdate) -> Conversation | None:
        """Update conversation by ID"""
        return self._run(lambda session: self._update_in_session(session, id, input_schema))

    async def aupdate(self, id: UUID, input_schema: ConversationUpdate) -> Conversation | None:
        """Async twin of :meth:`update`."""
        return await self._arun(lambda session: self._update_in_session(session, id, input_schema))

    def set_plan_lifecycle(self, id: UUID, lifecycle: PlanLifecycle | None) -> Conversation | None:
        """Persist the internal plan lifecycle state."""
        with self.session_factory() as session:
            db_obj = self._crud_strategy.get_by_id(session, id)
            if db_obj is None:
                return None

            db_obj.plan_lifecycle = lifecycle
            session.commit()
            session.refresh(db_obj)
            return db_obj

    def delete(self, id: UUID) -> bool:
        """Delete conversation by ID"""
        with self.session_factory() as session:
            return self._crud_strategy.delete(session, id)

    def exists(self, id: UUID) -> bool:
        """Check if conversation exists"""
        with self.session_factory() as session:
            return self._crud_strategy.exists(session, id)

    async def aexists(self, id: UUID) -> bool:
        """Async twin of :meth:`exists`."""
        return await self._arun(lambda session: self._crud_strategy.exists(session, id))

    def get_soft_deleted(self) -> list[Conversation]:
        """Get all soft-deleted conversations (used by checkpoint retention cleanup)."""
        with self.session_factory() as session:
            return self._crud_strategy.get_soft_deleted(session)
