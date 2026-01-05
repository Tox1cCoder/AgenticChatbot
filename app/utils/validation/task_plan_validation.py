"""
TaskPlan validation utilities.
"""

from typing import List, Set, Optional
from uuid import UUID

from app.repositories.task_plan import TaskPlanRepository
from app.repositories.conversation import ConversationRepository
from app.core.exceptions import ResourceNotFoundException, ValidationException


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

    def validate_dependencies(
        self, conversation_id: UUID, dependencies: List[UUID]
    ) -> None:
        """Verify all dependency task IDs exist in the same conversation.

        Args:
            conversation_id: The conversation ID for context
            dependencies: List of dependency task IDs to validate

        Raises:
            ValidationException: If any dependency is invalid
        """
        if not dependencies:
            return

        for dep_id in dependencies:
            task = self.task_plan_repository.get_by_id_and_conversation(
                dep_id, conversation_id
            )
            if not task:
                raise ValidationException(
                    detail=f"Invalid dependency: Task {dep_id} not found in this conversation",
                    error_code="INVALID_DEPENDENCY",
                )

    def validate_no_circular_dependencies(
        self,
        conversation_id: UUID,
        task_id: UUID,
        new_dependencies: List[UUID],
    ) -> None:
        """Check for circular dependency chains.

        Uses DFS-based cycle detection to ensure adding the new dependencies
        won't create a circular dependency chain.

        Args:
            conversation_id: The conversation ID for context
            task_id: The task being updated
            new_dependencies: The new dependencies to add

        Raises:
            ValidationException: If circular dependency detected
        """
        if not new_dependencies:
            return

        # Build dependency graph for all tasks in conversation
        all_tasks = self.task_plan_repository.get_by_conversation_id(
            conversation_id, include_completed=True
        )

        # Create adjacency list (task_id -> list of tasks that depend on it)
        graph: dict[UUID, List[UUID]] = {}
        for task in all_tasks:
            deps = task.dependencies or []
            for dep in deps:
                dep_id = self._coerce_uuid(dep)
                if not dep_id:
                    continue
                if dep_id not in graph:
                    graph[dep_id] = []
                graph[dep_id].append(task.id)

        # Update graph with new dependencies for the task being modified
        for dep_id in new_dependencies:
            if dep_id not in graph:
                graph[dep_id] = []
            if task_id not in graph[dep_id]:
                graph[dep_id].append(task_id)

        # DFS-based cycle detection
        visited: Set[UUID] = set()
        rec_stack: Set[UUID] = set()

        def has_cycle(node: UUID) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        # Check for cycles starting from each new dependency
        for dep_id in new_dependencies:
            if dep_id not in visited:
                if has_cycle(dep_id):
                    raise ValidationException(
                        detail="Circular dependency detected. Tasks cannot depend on each other in a cycle.",
                        error_code="CIRCULAR_DEPENDENCY",
                    )

        # Also check if the new dependencies would make task_id depend on itself
        # (through a chain of dependencies)
        def can_reach(start: UUID, target: UUID, visited_local: Set[UUID]) -> bool:
            """Check if target is reachable from start through dependencies."""
            if start == target:
                return True
            if start in visited_local:
                return False
            visited_local.add(start)

            for neighbor in graph.get(start, []):
                if can_reach(neighbor, target, visited_local):
                    return True
            return False

        for dep_id in new_dependencies:
            if can_reach(task_id, dep_id, set()):
                raise ValidationException(
                    detail=f"Circular dependency detected: Task would indirectly depend on itself",
                    error_code="CIRCULAR_DEPENDENCY",
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
