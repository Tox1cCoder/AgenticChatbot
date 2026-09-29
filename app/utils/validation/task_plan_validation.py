"""
TaskPlan validation utilities.
"""

from uuid import UUID

from app.core.exceptions import ResourceNotFoundException
from app.repositories.conversation import ConversationRepository
from app.repositories.task_plan import TaskPlanRepository


class TaskPlanValidationUtils:
    """Utilities for task plan-related validations."""

    def __init__(self, session_factory: callable):
        """Initialize validation utils with session factory for dependency injection."""
        self.session_factory = session_factory
        self.task_plan_repository = TaskPlanRepository(session_factory)
        self.conversation_repository = ConversationRepository(session_factory)

    def validate_task_access(self, user_id: UUID, task_id: UUID) -> None:
        """Raise not-found unless ``user_id`` owns the task's conversation.

        A foreign task reads as missing, not forbidden, so ids cannot be probed.
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
