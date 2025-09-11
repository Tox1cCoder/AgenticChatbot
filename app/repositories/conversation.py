from typing import List, Optional
from uuid import UUID
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import select

from app.models.conversation import Conversation
from app.repositories.strategy import Repository, DefaultCRUDStrategy
from app.schemas.conversation import ConversationCreate, ConversationUpdate


class ConversationCRUDStrategy(
    DefaultCRUDStrategy[Conversation, ConversationCreate, ConversationUpdate]
):
    """Custom CRUD strategy for Conversation operations"""

    def get_by_user_id(
        self, db: Session, user_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[Conversation]:
        """Get conversations by user ID"""
        stmt = (
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(db.execute(stmt).scalars().all())

    def get_with_messages(
        self, db: Session, conversation_id: UUID
    ) -> Optional[Conversation]:
        """Get conversation with its messages"""
        stmt = (
            select(Conversation)
            .options(joinedload(Conversation.messages))
            .where(Conversation.id == conversation_id)
        )
        return db.execute(stmt).scalar_one_or_none()

    def get_user_conversation_count(self, db: Session, user_id: UUID) -> int:
        """Get count of conversations for a user"""
        stmt = select(Conversation.id).where(Conversation.user_id == user_id)
        return len(list(db.execute(stmt).scalars().all()))

    def user_owns_conversation(
        self, db: Session, user_id: UUID, conversation_id: UUID
    ) -> bool:
        """Check if user owns the conversation"""
        stmt = select(Conversation.id).where(
            Conversation.id == conversation_id, Conversation.user_id == user_id
        )
        return db.execute(stmt).scalar() is not None


class ConversationRepository(
    Repository[Conversation, ConversationCreate, ConversationUpdate]
):
    """Repository for Conversation model"""

    def __init__(self, db: Session):
        strategy = ConversationCRUDStrategy(Conversation)
        super().__init__(db, strategy)

    def get_by_user_id(
        self, user_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[Conversation]:
        """Get conversations by user ID"""
        return self._crud_strategy.get_by_user_id(self.db, user_id, skip, limit)

    def get_with_messages(self, conversation_id: UUID) -> Optional[Conversation]:
        """Get conversation with its messages"""
        return self._crud_strategy.get_with_messages(self.db, conversation_id)

    def get_user_conversation_count(self, user_id: UUID) -> int:
        """Get count of conversations for a user"""
        return self._crud_strategy.get_user_conversation_count(self.db, user_id)

    def user_owns_conversation(self, user_id: UUID, conversation_id: UUID) -> bool:
        """Check if user owns the conversation"""
        return self._crud_strategy.user_owns_conversation(
            self.db, user_id, conversation_id
        )
