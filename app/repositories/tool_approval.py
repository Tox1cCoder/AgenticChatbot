"""Repository for managing tool approval audit trail records."""

from typing import List, Optional
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import select, desc

from app.models.tool_approval import ToolApproval
from app.repositories.command_strategy import DefaultCommandStrategy
from app.repositories.query_strategy import DefaultQueryStrategy


class ToolApprovalCRUDStrategy(
    DefaultCommandStrategy[ToolApproval, dict, dict],
    DefaultQueryStrategy[ToolApproval],
):
    """Custom CRUD strategy for ToolApproval operations."""

    def __init__(self, model: type[ToolApproval]):
        DefaultCommandStrategy.__init__(self, model)
        DefaultQueryStrategy.__init__(self, model)

    def get_by_conversation_id(
        self, db: Session, conversation_id: UUID, limit: int = 100
    ) -> List[ToolApproval]:
        """
        Retrieve all tool approvals for a specific conversation.

        Args:
            db: Database session
            conversation_id: UUID of the conversation
            limit: Maximum number of records to return

        Returns:
            List of ToolApproval records ordered by decision time (newest first)
        """
        statement = (
            select(ToolApproval)
            .where(ToolApproval.conversation_id == conversation_id)
            .order_by(desc(ToolApproval.decided_at))
            .limit(limit)
        )
        return list(db.execute(statement).scalars().all())

    def get_by_interrupt_id(self, db: Session, interrupt_id: str) -> List[ToolApproval]:
        """
        Retrieve all tool approvals for a specific interrupt.

        Args:
            db: Database session
            interrupt_id: Interrupt identifier

        Returns:
            List of ToolApproval records for the interrupt
        """
        statement = (
            select(ToolApproval)
            .where(ToolApproval.interrupt_id == interrupt_id)
            .order_by(ToolApproval.decided_at)
        )
        return list(db.execute(statement).scalars().all())

    def get_by_user_id(
        self, db: Session, user_id: UUID, limit: int = 100
    ) -> List[ToolApproval]:
        """
        Retrieve all tool approvals made by a specific user.

        Args:
            db: Database session
            user_id: UUID of the user
            limit: Maximum number of records to return

        Returns:
            List of ToolApproval records ordered by decision time (newest first)
        """
        statement = (
            select(ToolApproval)
            .where(ToolApproval.user_id == user_id)
            .order_by(desc(ToolApproval.decided_at))
            .limit(limit)
        )
        return list(db.execute(statement).scalars().all())

    def create(self, db: Session, approval_data: dict) -> ToolApproval:
        """
        Create a new tool approval record.

        Args:
            db: Database session
            approval_data: Dictionary containing approval information

        Returns:
            Created ToolApproval instance
        """
        approval = ToolApproval(**approval_data)
        db.add(approval)
        db.commit()
        db.refresh(approval)
        return approval


class ToolApprovalRepository:
    """Repository for tool approval operations."""

    def __init__(self, session_factory: callable):
        """
        Initialize repository with session factory for dependency injection.

        Args:
            session_factory: Callable that returns a context-managed DB session
        """
        self.session_factory = session_factory
        self.strategy = ToolApprovalCRUDStrategy(ToolApproval)

    def create(self, approval_data: dict) -> ToolApproval:
        """Create a new tool approval record."""
        with self.session_factory() as db:
            return self.strategy.create(db, approval_data)

    def get_by_conversation_id(
        self, conversation_id: UUID, limit: int = 100
    ) -> List[ToolApproval]:
        """Get all tool approvals for a conversation."""
        with self.session_factory() as db:
            return self.strategy.get_by_conversation_id(db, conversation_id, limit)

    def get_by_interrupt_id(self, interrupt_id: str) -> List[ToolApproval]:
        """Get all tool approvals for an interrupt."""
        with self.session_factory() as db:
            return self.strategy.get_by_interrupt_id(db, interrupt_id)

    def get_by_user_id(self, user_id: UUID, limit: int = 100) -> List[ToolApproval]:
        """Get all tool approvals made by a user."""
        with self.session_factory() as db:
            return self.strategy.get_by_user_id(db, user_id, limit)
