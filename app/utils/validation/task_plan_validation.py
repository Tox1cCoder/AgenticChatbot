"""
TaskPlan validation utilities.
"""

from typing import Optional
from uuid import UUID

from app.repositories.task_plan import TaskPlanRepository
from app.repositories.conversation import ConversationRepository
from app.core.exceptions import ResourceNotFoundException


class TaskPlanValidationUtils:
    """Utilities for task plan-related validations."""

    def __init__(self, session_factory: callable):
        """Initialize validation utils with session factory for dependency injection."""
        self.session_factory = session_factory
        self.task_plan_repository = TaskPlanRepository(session_factory)
        self.conversation_repository = ConversationRepository(session_factory)

    def validate_task_exists(self, task_id: UUID) -> None:
        """Validate that a task plan exists.

        Args:
            task_id: The task ID to check

        Raises:
            ResourceNotFoundException: If task doesn't exist
        """
        if not self.task_plan_repository.exists(task_id):
            raise ResourceNotFoundException(
                detail="Task plan not found",
                error_code="TASK_PLAN_NOT_FOUND",
            )

    def validate_task_access(self, user_id: UUID, task_id: UUID) -> None:
        """Verify user owns the conversation containing the task.

        Args:
            user_id: The user ID to check ownership
            task_id: The task ID to check access for

        Raises:
            ResourceNotFoundException: If task not found or access denied
        """
        task = self.task_plan_repository.get_by_id(task_id)
        if not task:
            raise ResourceNotFoundException(
                detail="Task plan not found",
                error_code="TASK_PLAN_NOT_FOUND",
            )

        conversation = self.conversation_repository.get_by_id(task.conversation_id)
        if not conversation:
            raise ResourceNotFoundException(
                detail="Conversation not found",
                error_code="CONVERSATION_NOT_FOUND",
            )

        if conversation.owner_id != user_id:
            raise ResourceNotFoundException(
                detail="Task plan not found",
                error_code="TASK_PLAN_NOT_FOUND",
            )

    @staticmethod
    def _coerce_uuid(value: UUID | str | None) -> Optional[UUID]:
        """Convert stored dependency identifiers into UUID objects."""
        if value is None:
            return None
        if isinstance(value, UUID):
            return value
        try:
            return UUID(str(value))
        except (ValueError, TypeError):
            return None
