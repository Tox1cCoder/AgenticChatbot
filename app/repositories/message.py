from typing import List, Optional
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.models.message import Message
from app.repositories.strategy import Repository, DefaultCRUDStrategy
from app.schemas.message import MessageCreate, MessageUpdate


class MessageCRUDStrategy(DefaultCRUDStrategy[Message, MessageCreate, MessageUpdate]):
    """Custom CRUD strategy for Message operations"""

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

    def get_conversation_message_count(self, db: Session, conversation_id: UUID) -> int:
        """Get count of messages in a conversation"""
        stmt = select(Message.id).where(Message.conversation_id == conversation_id)
        return len(list(db.execute(stmt).scalars().all()))

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
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
        )
        return list(db.execute(stmt).scalars().all())


class MessageRepository(Repository[Message, MessageCreate, MessageUpdate]):
    """Repository for Message model using strategy pattern"""

    def __init__(self, db: Session):
        strategy = MessageCRUDStrategy(Message)
        super().__init__(db, strategy)

    def get_by_conversation_id(
        self, conversation_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[Message]:
        """Get messages by conversation ID ordered by creation time"""
        return self._crud_strategy.get_by_conversation_id(
            self.db, conversation_id, skip, limit
        )

    def get_conversation_history(
        self, conversation_id: UUID, limit: int = 50
    ) -> List[Message]:
        """Get recent conversation history"""
        return self._crud_strategy.get_conversation_history(
            self.db, conversation_id, limit
        )

    def get_conversation_message_count(self, conversation_id: UUID) -> int:
        """Get count of messages in a conversation"""
        return self._crud_strategy.get_conversation_message_count(
            self.db, conversation_id
        )

    def get_latest_message(self, conversation_id: UUID) -> Optional[Message]:
        """Get the latest message in a conversation"""
        return self._crud_strategy.get_latest_message(self.db, conversation_id)

    def get_conversation_thread(self, conversation_id: UUID) -> List[Message]:
        """Get all messages in a conversation thread ordered by creation time"""
        return self._crud_strategy.get_conversation_thread(self.db, conversation_id)
