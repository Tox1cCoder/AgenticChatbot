from typing import List, Optional
from uuid import UUID
from contextlib import AbstractContextManager
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import select, asc, desc

from app.models.conversation import Conversation
from app.repositories.command_strategy import DefaultCommandStrategy
from app.repositories.query_strategy import DefaultQueryStrategy
from app.schemas.conversation import ConversationCreate, ConversationUpdate


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
        order_by: Optional[str] = None,
        order_direction: str = "desc",
    ) -> List[Conversation]:
        """Get conversations by owner ID with page-based pagination and ordering"""
        offset = (page - 1) * limit
        stmt = select(Conversation).where(
            Conversation.owner_id == owner_id, Conversation.deleted_at.is_(None)
        )

        # Apply ordering if specified
        if order_by:
            order_column = getattr(Conversation, order_by, None)
            if order_column is not None:
                if order_direction.lower() == "asc":
                    stmt = stmt.order_by(asc(order_column))
                else:
                    stmt = stmt.order_by(desc(order_column))
        else:
            # Default ordering
            stmt = stmt.order_by(Conversation.updated_at.asc())

        stmt = stmt.offset(offset).limit(limit)
        return list(db.execute(stmt).scalars().all())

    def count_by_owner_id(self, db: Session, owner_id: UUID) -> int:
        """Count conversations by owner ID"""
        stmt = select(Conversation).where(
            Conversation.owner_id == owner_id, Conversation.deleted_at.is_(None)
        )
        return len(list(db.execute(stmt).scalars().all()))

    def get_with_messages(
        self, db: Session, conversation_id: UUID
    ) -> Optional[Conversation]:
        """Get conversation with its messages"""
        stmt = (
            select(Conversation)
            .options(joinedload(Conversation.messages))
            .where(
                Conversation.id == conversation_id, Conversation.deleted_at.is_(None)
            )
        )
        return db.execute(stmt).scalar_one_or_none()

    def user_owns_conversation(
        self, db: Session, owner_id: UUID, conversation_id: UUID
    ) -> bool:
        """Check if user owns the conversation"""
        stmt = select(Conversation.id).where(
            Conversation.id == conversation_id,
            Conversation.owner_id == owner_id,
            Conversation.deleted_at.is_(None),
        )
        return db.execute(stmt).scalar() is not None


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
        order_by: Optional[str] = None,
        order_direction: str = "desc",
    ) -> List[Conversation]:
        """Get conversations by owner ID with page-based pagination and ordering"""
        with self.session_factory() as session:
            return self._crud_strategy.get_by_owner_id(
                session, owner_id, page, limit, order_by, order_direction
            )

    def count_by_owner_id(self, owner_id: UUID) -> int:
        """Count conversations by owner ID"""
        with self.session_factory() as session:
            return self._crud_strategy.count_by_owner_id(session, owner_id)

    def get_with_messages(self, conversation_id: UUID) -> Optional[Conversation]:
        """Get conversation with its messages"""
        with self.session_factory() as session:
            return self._crud_strategy.get_with_messages(session, conversation_id)

    def user_owns_conversation(self, owner_id: UUID, conversation_id: UUID) -> bool:
        """Check if user owns the conversation"""
        with self.session_factory() as session:
            return self._crud_strategy.user_owns_conversation(
                session, owner_id, conversation_id
            )

    def create(self, input_schema: ConversationCreate) -> Conversation:
        """Create a new conversation"""
        with self.session_factory() as session:
            return self._crud_strategy.create(session, input_schema)

    def get_by_id(self, id: UUID) -> Optional[Conversation]:
        """Get conversation by ID"""
        with self.session_factory() as session:
            return self._crud_strategy.get_by_id(session, id)

    def get_all(self, page: int = 1, limit: int = 10) -> list[Conversation]:
        """Get all conversations with page-based pagination"""
        with self.session_factory() as session:
            return self._crud_strategy.get_all(session, page, limit)

    def update(
        self, id: UUID, input_schema: ConversationUpdate
    ) -> Optional[Conversation]:
        """Update conversation by ID"""
        with self.session_factory() as session:
            db_obj = self._crud_strategy.get_by_id(session, id)
            if db_obj is None:
                return None
            return self._crud_strategy.update(session, db_obj, input_schema)

    def delete(self, id: UUID) -> bool:
        """Delete conversation by ID"""
        with self.session_factory() as session:
            return self._crud_strategy.delete(session, id)

    def exists(self, id: UUID) -> bool:
        """Check if conversation exists"""
        with self.session_factory() as session:
            return self._crud_strategy.exists(session, id)
