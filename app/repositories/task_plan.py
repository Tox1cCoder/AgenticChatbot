"""
TaskPlan repository for database operations.
"""

from typing import List, Optional, Any, Dict
from uuid import UUID
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import select, asc, func

from app.models.task_plan import TaskPlan
from app.models.enums import TaskStatus
from app.repositories.command_strategy import DefaultCommandStrategy
from app.repositories.query_strategy import DefaultQueryStrategy
from app.schemas.task_plan import TaskPlanCreate, TaskPlanUpdate


class TaskPlanCRUDStrategy(
    DefaultCommandStrategy[TaskPlan, TaskPlanCreate, TaskPlanUpdate],
    DefaultQueryStrategy[TaskPlan],
):
    """Custom CRUD strategy for TaskPlan operations."""

    def __init__(self, model: type[TaskPlan]):
        DefaultCommandStrategy.__init__(self, model)
        DefaultQueryStrategy.__init__(self, model)

    def get_by_id(self, db: Session, id: UUID) -> Optional[TaskPlan]:
        """Get a task plan by ID (TaskPlan doesn't have soft delete)."""
        statement = select(self.model).where(self.model.id == id)
        return db.execute(statement).scalar_one_or_none()

    def get_by_conversation_id(
        self,
        db: Session,
        conversation_id: UUID,
        include_completed: bool = True,
    ) -> List[TaskPlan]:
        """Get all tasks for a conversation, optionally filter out completed tasks.

        Args:
            db: Database session
            conversation_id: The conversation ID to filter by
            include_completed: If False, exclude completed and skipped tasks

        Returns:
            List of TaskPlan ordered by task_order ASC
        """
        statement = select(self.model).where(
            self.model.conversation_id == conversation_id
        )

        if not include_completed:
            statement = statement.where(
                self.model.status.notin_([TaskStatus.completed, TaskStatus.skipped])
            )

        statement = statement.order_by(asc(self.model.task_order))
        return list(db.execute(statement).scalars().all())

    def get_pending_tasks(self, db: Session, conversation_id: UUID) -> List[TaskPlan]:
        """Get tasks with status=pending, ordered by task_order.

        Args:
            db: Database session
            conversation_id: The conversation ID to filter by

        Returns:
            List of pending TaskPlan ordered by task_order ASC
        """
        statement = (
            select(self.model)
            .where(
                self.model.conversation_id == conversation_id,
                self.model.status == TaskStatus.pending,
            )
            .order_by(asc(self.model.task_order))
        )
        return list(db.execute(statement).scalars().all())

    def get_next_task(self, db: Session, conversation_id: UUID) -> Optional[TaskPlan]:
        """Get the first pending task by task_order.

        Args:
            db: Database session
            conversation_id: The conversation ID to filter by

        Returns:
            The next TaskPlan to work on, or None if all tasks are complete
        """
        pending_tasks = self.get_pending_tasks(db, conversation_id)
        return pending_tasks[0] if pending_tasks else None

    def mark_completed(self, db: Session, task_id: UUID) -> Optional[TaskPlan]:
        """Update task status to completed and set completed_at timestamp.

        Args:
            db: Database session
            task_id: The task ID to mark as completed

        Returns:
            The updated TaskPlan, or None if not found
        """
        statement = select(self.model).where(self.model.id == task_id)
        task = db.execute(statement).scalar_one_or_none()

        if task:
            task.status = TaskStatus.completed
            task.completed_at = datetime.now(timezone.utc)
            task.updated_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(task)

        return task

    def get_by_id_and_conversation(
        self, db: Session, task_id: UUID, conversation_id: UUID
    ) -> Optional[TaskPlan]:
        """Get task by ID with conversation ownership check.

        Args:
            db: Database session
            task_id: The task ID to retrieve
            conversation_id: The conversation ID for ownership check

        Returns:
            The TaskPlan if found and belongs to conversation, None otherwise
        """
        statement = select(self.model).where(
            self.model.id == task_id,
            self.model.conversation_id == conversation_id,
        )
        return db.execute(statement).scalar_one_or_none()

    def count_by_conversation(
        self, db: Session, conversation_id: UUID, status: Optional[TaskStatus] = None
    ) -> int:
        """Count tasks, optionally filtered by status.

        Args:
            db: Database session
            conversation_id: The conversation ID to filter by
            status: Optional TaskStatus to filter by

        Returns:
            Count of matching tasks
        """
        statement = select(func.count(self.model.id)).where(
            self.model.conversation_id == conversation_id
        )

        if status is not None:
            statement = statement.where(self.model.status == status)

        return db.execute(statement).scalar() or 0

    def delete(self, db: Session, id: UUID) -> bool:
        """Hard delete a task plan (TaskPlan doesn't have soft delete).

        Args:
            db: Database session
            id: The task ID to delete

        Returns:
            True if deleted, False if not found
        """
        statement = select(self.model).where(self.model.id == id)
        task = db.execute(statement).scalar_one_or_none()

        if task:
            db.delete(task)
            db.commit()
            return True
        return False

    def exists(self, db: Session, id: UUID) -> bool:
        """Check if a task plan exists by ID.

        Args:
            db: Database session
            id: The task ID to check

        Returns:
            True if exists, False otherwise
        """
        statement = select(self.model.id).where(self.model.id == id)
        return db.execute(statement).scalar() is not None


class TaskPlanRepository:
    """Repository for TaskPlan model using session factory pattern."""

    def __init__(self, session_factory: callable):
        """Initialize repository with session factory for dependency injection."""
        self.session_factory = session_factory
        self._crud_strategy = TaskPlanCRUDStrategy(TaskPlan)

    def create(self, input_data: dict) -> TaskPlan:
        """Create a new task plan.

        Args:
            input_data: Dictionary with task plan data

        Returns:
            Created TaskPlan
        """
        payload = dict(input_data)
        if "dependencies" in payload:
            payload["dependencies"] = self._serialize_dependencies(
                payload.get("dependencies")
            )

        with self.session_factory() as session:
            return self._crud_strategy.create(session, payload)

    def get_by_id(self, id: UUID) -> Optional[TaskPlan]:
        """Get task plan by ID.

        Args:
            id: The task ID to retrieve

        Returns:
            TaskPlan if found, None otherwise
        """
        with self.session_factory() as session:
            return self._crud_strategy.get_by_id(session, id)

    def get_by_conversation_id(
        self,
        conversation_id: UUID,
        include_completed: bool = True,
    ) -> List[TaskPlan]:
        """Get all tasks for a conversation.

        Args:
            conversation_id: The conversation ID to filter by
            include_completed: If False, exclude completed and skipped tasks

        Returns:
            List of TaskPlan ordered by task_order ASC
        """
        with self.session_factory() as session:
            return self._crud_strategy.get_by_conversation_id(
                session, conversation_id, include_completed
            )

    def get_pending_tasks(self, conversation_id: UUID) -> List[TaskPlan]:
        """Get tasks with status=pending.

        Args:
            conversation_id: The conversation ID to filter by

        Returns:
            List of pending TaskPlan ordered by task_order ASC
        """
        with self.session_factory() as session:
            return self._crud_strategy.get_pending_tasks(session, conversation_id)

    def get_next_task(self, conversation_id: UUID) -> Optional[TaskPlan]:
        """Get the first pending task whose dependencies are all completed.

        Args:
            conversation_id: The conversation ID to filter by

        Returns:
            The next TaskPlan to work on, or None if all tasks are complete
        """
        with self.session_factory() as session:
            return self._crud_strategy.get_next_task(session, conversation_id)

    def mark_completed(self, task_id: UUID) -> Optional[TaskPlan]:
        """Update task status to completed and set completed_at timestamp.

        Args:
            task_id: The task ID to mark as completed

        Returns:
            The updated TaskPlan, or None if not found
        """
        with self.session_factory() as session:
            return self._crud_strategy.mark_completed(session, task_id)

    def get_by_id_and_conversation(
        self, task_id: UUID, conversation_id: UUID
    ) -> Optional[TaskPlan]:
        """Get task by ID with conversation ownership check.

        Args:
            task_id: The task ID to retrieve
            conversation_id: The conversation ID for ownership check

        Returns:
            The TaskPlan if found and belongs to conversation, None otherwise
        """
        with self.session_factory() as session:
            return self._crud_strategy.get_by_id_and_conversation(
                session, task_id, conversation_id
            )

    def count_by_conversation(
        self, conversation_id: UUID, status: Optional[TaskStatus] = None
    ) -> int:
        """Count tasks, optionally filtered by status.

        Args:
            conversation_id: The conversation ID to filter by
            status: Optional TaskStatus to filter by

        Returns:
            Count of matching tasks
        """
        with self.session_factory() as session:
            return self._crud_strategy.count_by_conversation(
                session, conversation_id, status
            )

    def update(self, id: UUID, input_schema: TaskPlanUpdate) -> Optional[TaskPlan]:
        """Update task plan by ID.

        Args:
            id: The task ID to update
            input_schema: Update data

        Returns:
            Updated TaskPlan, or None if not found
        """
        update_payload: Dict[str, Any]
        if hasattr(input_schema, "model_dump"):
            update_payload = input_schema.model_dump(exclude_unset=True)
        else:
            update_payload = dict(getattr(input_schema, "__dict__", {}))

        if "dependencies" in update_payload and update_payload["dependencies"] is not None:
            update_payload["dependencies"] = self._serialize_dependencies(
                update_payload["dependencies"]
            )

        class _UpdateWrapper:
            def __init__(self, data: Dict[str, Any]):
                self._data = data

            def model_dump(self, *args, **kwargs):
                return self._data

        with self.session_factory() as session:
            db_obj = self._crud_strategy.get_by_id(session, id)
            if db_obj is None:
                return None
            return self._crud_strategy.update(
                session, db_obj, _UpdateWrapper(update_payload)
            )

    def delete(self, id: UUID) -> bool:
        """Delete task plan by ID.

        Args:
            id: The task ID to delete

        Returns:
            True if deleted, False if not found
        """
        with self.session_factory() as session:
            return self._crud_strategy.delete(session, id)

    def exists(self, id: UUID) -> bool:
        """Check if task plan exists.

        Args:
            id: The task ID to check

        Returns:
            True if exists, False otherwise
        """
        with self.session_factory() as session:
            return self._crud_strategy.exists(session, id)

    @staticmethod
    def _serialize_dependencies(dependencies: Optional[List[Any]]) -> List[str]:
        """Convert dependency identifiers to strings for JSON storage."""
        if not dependencies:
            return []

        serialized: List[str] = []
        for dep in dependencies:
            if isinstance(dep, UUID):
                serialized.append(str(dep))
            elif dep is not None:
                serialized.append(str(dep))
        return serialized
