"""
TaskPlan factory for creating TaskPlan entities.
"""

from __future__ import annotations

from typing import Dict, Any, List
from uuid import uuid4, UUID

from app.schemas.task_plan import TaskPlanCreate
from app.models.enums import TaskStatus
from app.utils.timestamp_utils import TimestampUtils
from app.ai.schemas import Task, Plan


class TaskPlanFactory:
    """Factory for creating TaskPlan entities."""

    @staticmethod
    def create_from_schema(task_data: TaskPlanCreate) -> Dict[str, Any]:
        """Create TaskPlan data dictionary from TaskPlanCreate schema.

        Args:
            task_data: TaskPlanCreate schema with task data

        Returns:
            Dictionary ready for database insertion
        """
        return {
            "id": uuid4(),
            "conversation_id": task_data.conversation_id,
            "task_order": task_data.task_order,
            "description": task_data.description,
            "status": TaskStatus.pending,
            "dependencies": TaskPlanFactory._serialize_dependencies(
                task_data.dependencies or []
            ),
            "task_metadata": task_data.task_metadata or {},
            "created_at": TimestampUtils.now(),
            "updated_at": TimestampUtils.now(),
            "completed_at": None,
        }

    @staticmethod
    def create_from_plan_task(
        conversation_id: UUID,
        task: Task,
        task_order: int,
        task_id: UUID = None,
    ) -> Dict[str, Any]:
        """Create TaskPlan data from planning agent's Task schema.

        Note: Dependencies are stored as indices initially. The caller is responsible
        for resolving these indices to UUIDs after all tasks are created.

        Args:
            conversation_id: The conversation this task belongs to
            task: Task schema from planning agent
            task_order: The order/sequence of this task
            task_id: Optional pre-generated UUID for this task

        Returns:
            Dictionary ready for database insertion
        """
        metadata = {}
        if task.estimated_complexity:
            metadata["complexity"] = task.estimated_complexity

        # Store dependency indices temporarily - will be resolved to UUIDs by caller
        metadata["dependency_indices"] = task.dependencies

        return {
            "id": task_id or uuid4(),
            "conversation_id": conversation_id,
            "task_order": task_order,
            "description": task.description,
            "status": TaskStatus.pending,
            "dependencies": [],  # Will be populated after all tasks are created
            "task_metadata": metadata,
            "created_at": TimestampUtils.now(),
            "updated_at": TimestampUtils.now(),
            "completed_at": None,
        }

    @staticmethod
    def create_batch_from_plan(
        conversation_id: UUID, plan: Plan
    ) -> List[Dict[str, Any]]:
        """Create multiple TaskPlan entities from a Plan.

        This method handles dependency resolution by:
        1. Pre-generating UUIDs for all tasks
        2. Mapping task indices to their UUIDs
        3. Resolving dependency indices to UUIDs

        Args:
            conversation_id: The conversation these tasks belong to
            plan: Plan schema containing list of tasks

        Returns:
            List of dictionaries ready for database insertion
        """
        if not plan.tasks:
            return []

        # Pre-generate UUIDs for all tasks
        task_ids = [uuid4() for _ in plan.tasks]

        task_plans = []
        for idx, task in enumerate(plan.tasks):
            task_data = TaskPlanFactory.create_from_plan_task(
                conversation_id=conversation_id,
                task=task,
                task_order=idx,
                task_id=task_ids[idx],
            )

            # Resolve dependency indices to UUIDs
            resolved_dependencies = []
            for dep_idx in task.dependencies:
                if 0 <= dep_idx < len(task_ids):
                    resolved_dependencies.append(task_ids[dep_idx])

            task_data["dependencies"] = TaskPlanFactory._serialize_dependencies(
                resolved_dependencies
            )

            # Store overall_goal in first task's metadata
            if idx == 0 and plan.overall_goal:
                task_data["task_metadata"]["overall_goal"] = plan.overall_goal

            task_plans.append(task_data)

        return task_plans

    @staticmethod
    def _serialize_dependencies(dependencies: List[UUID | str]) -> List[str]:
        """Convert dependency identifiers to strings for JSON storage."""
        serialized: List[str] = []
        for dep in dependencies:
            if isinstance(dep, UUID):
                serialized.append(str(dep))
            elif dep:
                serialized.append(str(dep))
        return serialized

    @staticmethod
    def create_from_descriptions(
        conversation_id: UUID,
        descriptions: List[str],
    ) -> List[Dict[str, Any]]:
        """Create TaskPlan entities from a list of descriptions.

        Creates tasks with sequential ordering and no dependencies.

        Args:
            conversation_id: The conversation these tasks belong to
            descriptions: List of task descriptions

        Returns:
            List of dictionaries ready for database insertion
        """
        task_plans = []
        for idx, description in enumerate(descriptions):
            task_plans.append(
                {
                    "id": uuid4(),
                    "conversation_id": conversation_id,
                    "task_order": idx,
                    "description": description,
                    "status": TaskStatus.pending,
                    "dependencies": [],
                    "task_metadata": {},
                    "created_at": TimestampUtils.now(),
                    "updated_at": TimestampUtils.now(),
                    "completed_at": None,
                }
            )

        return task_plans
