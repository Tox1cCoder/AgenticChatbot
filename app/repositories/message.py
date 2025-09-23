from typing import List, Optional
from uuid import UUID
from contextlib import AbstractContextManager
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.models.message import Message
from app.models.conversation import Conversation
from app.repositories.command_strategy import DefaultCommandStrategy
from app.repositories.query_strategy import DefaultQueryStrategy
from app.schemas.message import MessageCreate, MessageUpdate


class MessageCRUDStrategy(
    DefaultCommandStrategy[Message, MessageCreate, MessageUpdate],
    DefaultQueryStrategy[Message],
):
    """Custom CRUD strategy for Message operations"""

    def __init__(self, model: type[Message]):
        DefaultCommandStrategy.__init__(self, model)
        DefaultQueryStrategy.__init__(self, model)

    def get_by_conversation_id(
        self, db: Session, conversation_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[Message]:
        """Get messages by conversation ID ordered by creation time"""
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .offset(skip)
            .limit(limit)
        )
        return list(db.execute(stmt).scalars().all())

    def get_conversation_history(
        self, db: Session, conversation_id: UUID, limit: int = 50
    ) -> List[Message]:
        """Get recent conversation history"""
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        messages = list(db.execute(stmt).scalars().all())
        return list(reversed(messages))  # Return in chronological order

    def get_latest_message(
        self, db: Session, conversation_id: UUID
    ) -> Optional[Message]:
        """Get the latest message in a conversation"""
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(1)
        )
        return db.execute(stmt).scalar_one_or_none()

    def get_conversation_thread(
        self, db: Session, conversation_id: UUID
    ) -> List[Message]:
        """Get all messages in a conversation thread ordered by creation time"""
        stmt = select(Message).where(Message.conversation_id == conversation_id)
        return list(db.execute(stmt).scalars().all())

    def get_by_user_id(
        self, db: Session, user_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[Message]:
        """Get messages by conversation owner (user_id) with pagination"""
        # Join with conversations to get messages from user's conversations
        stmt = (
            select(Message)
            .join(Message.conversation)
            .where(Message.conversation.has(user_id=user_id))
            .order_by(Message.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(db.execute(stmt).scalars().all())


class MessageRepository:
    """Repository for Message model using session factory pattern"""

    def __init__(self, session_factory: callable):
        """Initialize repository with session factory for dependency injection."""
        self.session_factory = session_factory
        self._crud_strategy = MessageCRUDStrategy(Message)

    def get_by_conversation_id(
        self, conversation_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[Message]:
        """Get messages by conversation ID ordered by creation time"""
        with self.session_factory() as session:
            return self._crud_strategy.get_by_conversation_id(
                session, conversation_id, skip, limit
            )

    def get_conversation_history(
        self, conversation_id: UUID, limit: int = 50
    ) -> List[Message]:
        """Get recent conversation history"""
        with self.session_factory() as session:
            return self._crud_strategy.get_conversation_history(
                session, conversation_id, limit
            )

    def get_latest_message(self, conversation_id: UUID) -> Optional[Message]:
        """Get the latest message in a conversation"""
        with self.session_factory() as session:
            return self._crud_strategy.get_latest_message(session, conversation_id)

    def get_conversation_thread(self, conversation_id: UUID) -> List[Message]:
        """Get all messages in a conversation thread ordered by creation time"""
        with self.session_factory() as session:
            return self._crud_strategy.get_conversation_thread(session, conversation_id)

    def get_by_user_id(
        self, user_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[Message]:
        """Get messages by user ID with pagination"""
        with self.session_factory() as session:
            return self._crud_strategy.get_by_user_id(session, user_id, skip, limit)

    def create(self, input_schema: MessageCreate) -> Message:
        """Create a new message"""
        with self.session_factory() as session:
            return self._crud_strategy.create(session, input_schema)

    def get_by_id(self, id: UUID) -> Optional[Message]:
        """Get message by ID"""
        with self.session_factory() as session:
            return self._crud_strategy.get_by_id(session, id)

    def get_all(self, skip: int = 0, limit: int = 100) -> list[Message]:
        """Get all messages with pagination"""
        with self.session_factory() as session:
            return self._crud_strategy.get_all(session, skip, limit)

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
