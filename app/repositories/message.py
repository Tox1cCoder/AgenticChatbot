from typing import List, Optional
from uuid import UUID
from contextlib import AbstractContextManager
from sqlalchemy.orm import Session
from sqlalchemy import select, asc, desc

from app.models.message import Message
from app.models.conversation import Conversation
from app.repositories.command_strategy import DefaultCommandStrategy
from app.repositories.query_strategy import DefaultQueryStrategy
from app.repositories.utils.pagination import Paginator
from app.schemas.message import MessageCreate, MessageUpdate

from app.utils.validation.pagination_validation import validate_pagination_params


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
        order_by: Optional[str] = None,
        order_direction: str = "asc",
    ) -> Paginator[Message]:
        """Get messages by conversation ID with page-based pagination and ordering"""

        validate_pagination_params(page, limit)

        # Get total count first
        total = self.count_by_conversation_id(db, conversation_id)

        # Get paginated items
        offset = (page - 1) * limit
        statement = select(Message).where(Message.conversation_id == conversation_id)

        # Apply ordering if specified
        if order_by:
            order_column = getattr(Message, order_by, None)
            if order_column is not None:
                if order_direction.lower() == "asc":
                    statement = statement.order_by(asc(order_column))
                else:
                    statement = statement.order_by(desc(order_column))
        else:
            # Default ordering - chronological order (oldest first for proper conversation flow)
            statement = statement.order_by(Message.created_at.asc())

        statement = statement.offset(offset).limit(limit)
        items = list(db.execute(statement).scalars().all())

        return Paginator.create(items, total, page, limit)

    def count_by_conversation_id(self, db: Session, conversation_id: UUID) -> int:
        """Count messages by conversation ID"""
        statement = select(Message).where(Message.conversation_id == conversation_id)
        return len(list(db.execute(statement).scalars().all()))

    def get_by_user_id(
        self,
        db: Session,
        user_id: UUID,
        page: int = 1,
        limit: int = 10,
        order_by: Optional[str] = None,
        order_direction: str = "desc",
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
            .where(Message.conversation.has(owner_id=user_id))
        )

        # Apply ordering if specified
        if order_by:
            order_column = getattr(Message, order_by, None)
            if order_column is not None:
                if order_direction.lower() == "asc":
                    statement = statement.order_by(asc(order_column))
                else:
                    statement = statement.order_by(desc(order_column))
        else:
            # Default ordering
            statement = statement.order_by(Message.created_at.asc())

        statement = statement.offset(offset).limit(limit)
        items = list(db.execute(statement).scalars().all())

        return Paginator.create(items, total, page, limit)

    def count_by_user_id(self, db: Session, user_id: UUID) -> int:
        """Count messages by user ID"""
        statement = (
            select(Message)
            .join(Message.conversation)
            .where(Message.conversation.has(owner_id=user_id))
        )
        return len(list(db.execute(statement).scalars().all()))

    def get_conversation_history(
        self, db: Session, conversation_id: UUID, limit: int = 50
    ) -> List[Message]:
        """Get recent conversation history"""
        statement = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        messages = list(db.execute(statement).scalars().all())
        return list(reversed(messages))

    def get_conversation_thread(
        self, db: Session, conversation_id: UUID
    ) -> List[Message]:
        """Get all messages in a conversation thread ordered by creation time"""
        statement = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
        )
        return list(db.execute(statement).scalars().all())


class MessageRepository:
    """Repository for Message model using session factory pattern"""

    def __init__(self, session_factory: callable):
        """Initialize repository with session factory for dependency injection."""
        self.session_factory = session_factory
        self._crud_strategy = MessageCRUDStrategy(Message)

    def get_by_conversation_id(
        self,
        conversation_id: UUID,
        page: int = 1,
        limit: int = 10,
        order_by: Optional[str] = None,
        order_direction: str = "asc",
    ) -> Paginator[Message]:
        """Get messages by conversation ID with page-based pagination and ordering"""
        with self.session_factory() as session:
            return self._crud_strategy.get_by_conversation_id(
                session, conversation_id, page, limit, order_by, order_direction
            )

    def count_by_conversation_id(self, conversation_id: UUID) -> int:
        """Count messages by conversation ID"""
        with self.session_factory() as session:
            return self._crud_strategy.count_by_conversation_id(
                session, conversation_id
            )

    def get_by_user_id(
        self,
        user_id: UUID,
        page: int = 1,
        limit: int = 10,
        order_by: Optional[str] = None,
        order_direction: str = "desc",
    ) -> Paginator[Message]:
        """Get messages by user ID with page-based pagination and ordering"""
        with self.session_factory() as session:
            return self._crud_strategy.get_by_user_id(
                session, user_id, page, limit, order_by, order_direction
            )

    def count_by_user_id(self, user_id: UUID) -> int:
        """Count messages by user ID"""
        with self.session_factory() as session:
            return self._crud_strategy.count_by_user_id(session, user_id)

    def create(self, input_schema: MessageCreate) -> Message:
        """Create a new message"""
        with self.session_factory() as session:
            return self._crud_strategy.create(session, input_schema)

    def get_by_id(self, id: UUID) -> Optional[Message]:
        """Get message by ID"""
        with self.session_factory() as session:
            return self._crud_strategy.get_by_id(session, id)

    def get_all(self, page: int = 1, limit: int = 10) -> list[Message]:
        """Get all messages with page-based pagination"""
        with self.session_factory() as session:
            return self._crud_strategy.get_all(session, page, limit)

    def update(self, id: UUID, input_schema: MessageUpdate) -> Optional[Message]:
        """Update message by ID"""
        with self.session_factory() as session:
            db_obj = self._crud_strategy.get_by_id(session, id)
            if db_obj is None:
                return None
            return self._crud_strategy.update(session, db_obj, input_schema)

    def delete(self, id: UUID) -> bool:
        """Delete message by ID"""
        with self.session_factory() as session:
            return self._crud_strategy.delete(session, id)

    def exists(self, id: UUID) -> bool:
        """Check if message exists"""
        with self.session_factory() as session:
            return self._crud_strategy.exists(session, id)
