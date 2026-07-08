from uuid import UUID

from sqlalchemy import asc, desc, func, select
from sqlalchemy.orm import Session, joinedload

from app.models.conversation import Conversation
from app.models.enums import PlanLifecycle
from app.models.message import Message
from app.repositories.command_strategy import DefaultCommandStrategy
from app.repositories.query_strategy import DefaultQueryStrategy
from app.repositories.utils.pagination import Paginator
from app.schemas.conversation import ConversationCreate, ConversationUpdate
from app.utils.validation.pagination_validation import validate_pagination_params


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
    ) -> Paginator[Conversation]:
        """Get conversations by owner ID with page-based pagination and ordering"""

        validate_pagination_params(page, limit)
        # Get total count first
        total = self.count_by_owner_id(db, owner_id)

        # Get paginated items
        offset = (page - 1) * limit
        statement = select(Conversation).where(
            Conversation.owner_id == owner_id, Conversation.deleted_at.is_(None)
        )
        # Apply ordering if specified
        if hasattr(Conversation, order_by):
            order_column = getattr(Conversation, order_by)
            statement = statement.order_by(
                asc(order_column) if order_direction.lower() == "asc" else desc(order_column)
            )
        else:
            # Default ordering
            statement = statement.order_by(Conversation.updated_at.desc())

        statement = statement.offset(offset).limit(limit)
        items = list(db.execute(statement).scalars().all())

        return Paginator.create(items, total, page, limit)

    def count_by_owner_id(self, db: Session, owner_id: UUID) -> int:
        """Count conversations by owner ID"""
        statement = select(Conversation).where(
            Conversation.owner_id == owner_id, Conversation.deleted_at.is_(None)
        )
        return len(list(db.execute(statement).scalars().all()))

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
    ) -> list[Conversation]:
        """Get conversations with limited recent messages and total message count"""
        # Get paginated conversations for the user
        statement = select(Conversation).where(
            Conversation.owner_id == owner_id, Conversation.deleted_at.is_(None)
        )

        # Apply ordering if specified
        if hasattr(Conversation, order_by):
            order_column = getattr(Conversation, order_by)
            statement = statement.order_by(
                asc(order_column) if order_direction.lower() == "asc" else desc(order_column)
            )
        else:
            # Default ordering
            statement = statement.order_by(Conversation.updated_at.desc())

        # Apply pagination
        offset = (page - 1) * limit
        statement = statement.offset(offset).limit(limit)

        conversations = list(db.execute(statement).scalars().all())

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


class ConversationRepository:
    """Repository for Conversation model using session factory pattern"""

    def __init__(self, session_factory: callable):
        """Initialize repository with session factory for dependency injection."""
        self.session_factory = session_factory
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
                )
                # Get total count for pagination
                total = self._crud_strategy.count_by_owner_id(session, owner_id)
                return Paginator.create(conversations, total, page, limit)
            else:
                return self._crud_strategy.get_by_owner_id(
                    session, owner_id, page, limit, order_by, order_direction
                )

    def count_by_owner_id(self, owner_id: UUID) -> int:
        """Count conversations by owner ID"""
        with self.session_factory() as session:
            return self._crud_strategy.count_by_owner_id(session, owner_id)

    def get_with_messages(self, conversation_id: UUID) -> Conversation | None:
        """Get conversation with its messages"""
        with self.session_factory() as session:
            return self._crud_strategy.get_with_messages(session, conversation_id)

    def user_owns_conversation(self, owner_id: UUID, conversation_id: UUID) -> bool:
        """Check if user owns the conversation"""
        with self.session_factory() as session:
            return self._crud_strategy.user_owns_conversation(session, owner_id, conversation_id)

    def create(self, input_schema: ConversationCreate) -> Conversation:
        """Create a new conversation"""
        with self.session_factory() as session:
            return self._crud_strategy.create(session, input_schema)

    def get_by_id(self, id: UUID) -> Conversation | None:
        """Get conversation by ID"""
        with self.session_factory() as session:
            return self._crud_strategy.get_by_id(session, id)

    def get_all(self, page: int = 1, limit: int = 10) -> list[Conversation]:
        """Get all conversations with page-based pagination"""
        with self.session_factory() as session:
            return self._crud_strategy.get_all(session, page, limit)

    def update(self, id: UUID, input_schema: ConversationUpdate) -> Conversation | None:
        """Update conversation by ID"""
        with self.session_factory() as session:
            db_obj = self._crud_strategy.get_by_id(session, id)
            if db_obj is None:
                return None
            return self._crud_strategy.update(session, db_obj, input_schema)

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

    def get_soft_deleted(self) -> list[Conversation]:
        """Get all soft-deleted conversations (used by checkpoint retention cleanup)."""
        with self.session_factory() as session:
            return self._crud_strategy.get_soft_deleted(session)
