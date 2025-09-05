from typing import List
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.models.message import Message
from app.repositories.base import BaseRepository
from app.schemas.message import MessageCreate, MessageUpdate


class MessageRepository(BaseRepository[Message, MessageCreate, MessageUpdate]):
    """Repository for Message model with custom methods"""
    
    def __init__(self, db: Session):
        super().__init__(Message, db)
    
    def get_by_conversation_id(self, conversation_id: int, skip: int = 0, limit: int = 100) -> List[Message]:
        """Get messages by conversation ID ordered by creation time"""
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .offset(skip)
            .limit(limit)
        )
        return list(self.db.execute(stmt).scalars().all())
    
    def get_conversation_history(self, conversation_id: int, limit: int = 50) -> List[Message]:
        """Get recent conversation history"""
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        messages = list(self.db.execute(stmt).scalars().all())
        return list(reversed(messages))  # Return in chronological order
    
    def get_messages_by_user(self, user_id: int, skip: int = 0, limit: int = 100) -> List[Message]:
        """Get messages by user ID"""
        stmt = (
            select(Message)
            .where(Message.user_id == user_id)
            .order_by(Message.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(self.db.execute(stmt).scalars().all())
    
    def get_conversation_message_count(self, conversation_id: int) -> int:
        """Get count of messages in a conversation"""
        stmt = select(Message.id).where(Message.conversation_id == conversation_id)
        return len(list(self.db.execute(stmt).scalars().all()))
    
    def get_latest_message(self, conversation_id: int) -> Message | None:
        """Get the latest message in a conversation"""
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(1)
        )
        return self.db.execute(stmt).scalar_one_or_none()
