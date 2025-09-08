from typing import List, Optional
from uuid import UUID
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import select

from app.models.conversation import Conversation
from app.repositories.base import BaseRepository
from app.schemas.conversation import ConversationCreate, ConversationUpdate


class ConversationRepository(
    BaseRepository[Conversation, ConversationCreate, ConversationUpdate]
):
    """Repository for Conversation model with custom methods"""

    def __init__(self, db: Session):
        super().__init__(Conversation, db)

    def get_by_user_id(
        self, user_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[Conversation]:
        """Get conversations by user ID"""
        stmt = (
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(self.db.execute(stmt).scalars().all())

    def get_with_messages(self, conversation_id: UUID) -> Optional[Conversation]:
        """Get conversation with its messages"""
        stmt = (
            select(Conversation)
            .options(joinedload(Conversation.messages))
            .where(Conversation.id == conversation_id)
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def get_user_conversation_count(self, user_id: UUID) -> int:
        """Get count of conversations for a user"""
        stmt = select(Conversation.id).where(Conversation.user_id == user_id)
        return len(list(self.db.execute(stmt).scalars().all()))

    def user_owns_conversation(self, user_id: UUID, conversation_id: UUID) -> bool:
        """Check if user owns the conversation"""
        stmt = select(Conversation.id).where(
            Conversation.id == conversation_id, Conversation.user_id == user_id
        )
        return self.db.execute(stmt).scalar() is not None
