"""
TaskPlan service interface definition.
"""

from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID

from app.models.enums import PlanLifecycle
from app.schemas.task_plan import PlanningStatusResponse, TaskPlanRead, TaskPlanUpdate


class ITaskPlanService(ABC):
    """Interface for TaskPlan service operations."""

    @abstractmethod
    async def create_task_plan(
        self,
        conversation_id: UUID,
        user_message: str,
        user_id: UUID,
    ) -> list[TaskPlanRead]:
        """Create task plan from user request using planning agent.

        Args:
            conversation_id: The conversation to create tasks for
            user_message: The user's request to generate a plan for
            user_id: The user making the request

        Returns:
            List of created TaskPlanRead schemas
        """
        pass

    @abstractmethod
    def sync_todos_from_agent(
        self,
        conversation_id: UUID,
        todos: list[dict[str, Any]],
        user_id: UUID,
        preserve_existing_status: bool = False,
        lifecycle: PlanLifecycle | None = None,
    ) -> list[TaskPlanRead]:
        """Persist todo data produced by the planning agent.

        Args:
            conversation_id: The conversation to update
            todos: Planning agent todo payload
            user_id: The user requesting the sync (for ownership validation)
            preserve_existing_status: Whether matched existing task statuses should win

        Returns:
            List of TaskPlanRead schemas for the stored tasks
        """
        pass

    @abstractmethod
    def create_task_plan_from_list(
        self,
        conversation_id: UUID,
        task_descriptions: list[str],
        user_id: UUID,
    ) -> list[TaskPlanRead]:
        """Create or append task plan items from a manual list of descriptions.

        Args:
            conversation_id: The conversation to create tasks for
            task_descriptions: List of task descriptions
            user_id: The user making the request

        Returns:
            List of created TaskPlanRead schemas
        """
        pass

    @abstractmethod
    def get_by_id(self, task_id: UUID, user_id: UUID) -> TaskPlanRead:
        """Get task by ID with access validation.

        Args:
            task_id: The task ID to retrieve
            user_id: The user making the request

        Returns:
            TaskPlanRead schema

        Raises:
            ResourceNotFoundException: If task not found or access denied
        """
        pass

    @abstractmethod
    def get_conversation_tasks(
        self,
        conversation_id: UUID,
        user_id: UUID,
        include_completed: bool = False,
    ) -> list[TaskPlanRead]:
        """Get all tasks for a conversation.

        Args:
            conversation_id: The conversation to get tasks for
            user_id: The user making the request
            include_completed: Whether to include completed tasks

        Returns:
            List of TaskPlanRead schemas
        """
        pass

    @abstractmethod
    def get_active_or_next_task(self, conversation_id: UUID, user_id: UUID) -> TaskPlanRead | None:
        """Get the active task, otherwise the next pending task.

        Args:
            conversation_id: The conversation to get the active/next task for
            user_id: The user making the request

        Returns:
            TaskPlanRead schema or None if no active or pending tasks
        """
        pass

    @abstractmethod
    def set_plan_lifecycle(
        self,
        conversation_id: UUID,
        user_id: UUID,
        lifecycle: PlanLifecycle | None,
    ) -> None:
        """Persist an internal lifecycle transition for a conversation plan."""
        pass

    @abstractmethod
    def update_task(
        self,
        task_id: UUID,
        user_id: UUID,
        task_update_data: TaskPlanUpdate,
    ) -> TaskPlanRead:
        """Update task with ownership validation.

        Args:
            task_id: The task ID to update
            user_id: The user making the request
            task_update_data: Update data

        Returns:
            Updated TaskPlanRead schema
        """
        pass

    @abstractmethod
    def mark_task_completed(self, task_id: UUID, user_id: UUID) -> TaskPlanRead:
        """Mark task as completed.

        Args:
            task_id: The task ID to mark as completed
            user_id: The user making the request

        Returns:
            Updated TaskPlanRead schema
        """
        pass

    @abstractmethod
    def mark_task_in_progress(self, task_id: UUID, user_id: UUID) -> TaskPlanRead:
        """Mark task as in progress.

        Args:
            task_id: The task ID to mark as in progress
            user_id: The user making the request

        Returns:
            Updated TaskPlanRead schema
        """
        pass

    @abstractmethod
    def delete_task(self, task_id: UUID, user_id: UUID) -> bool:
        """Delete task with ownership validation.

        Args:
            task_id: The task ID to delete
            user_id: The user making the request

        Returns:
            True if deleted successfully
        """
        pass

    @abstractmethod
    def get_planning_status(self, conversation_id: UUID, user_id: UUID) -> PlanningStatusResponse:
        """Get planning mode status and progress.

        Args:
            conversation_id: The conversation to get status for
            user_id: The user making the request

        Returns:
            PlanningStatusResponse with planning status and progress
        """
        pass
