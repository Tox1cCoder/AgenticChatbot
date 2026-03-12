from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from uuid import UUID

from app.ai.agents.planning_agent import PlanningAgent
from app.ai.schemas import AgentMessage, MessageRole
from app.core.exceptions import ResourceNotFoundException
from app.interfaces.task_plan_service_interface import ITaskPlanService
from app.models.conversation import Conversation
from app.models.enums import TaskStatus
from app.models.task_plan import TaskPlan
from app.repositories.conversation import ConversationRepository
from app.repositories.task_plan import TaskPlanRepository
from app.schemas.conversation import ConversationUpdate
from app.schemas.task_plan import (
    PlanningStatusResponse,
    TaskPlanRead,
    TaskPlanUpdate,
)
from app.utils.validation.conversation_validation import ConversationValidationUtils
from app.utils.validation.task_plan_validation import TaskPlanValidationUtils

class TaskPlanService(ITaskPlanService):
    def __init__(
        self,
        task_plan_repository: TaskPlanRepository,
        conversation_validation_utils: ConversationValidationUtils,
        task_plan_validation_utils: TaskPlanValidationUtils,
        planning_agent: PlanningAgent,
        conversation_repository: ConversationRepository,
    ):
        self.task_plan_repository = task_plan_repository
        self.conversation_validation_utils = conversation_validation_utils
        self.task_plan_validation_utils = task_plan_validation_utils
        self.planning_agent = planning_agent
        self.conversation_repository = conversation_repository

    def _ensure_planning_mode_enabled(self, conversation_id: UUID) -> None:
        conversation = self.conversation_repository.get_by_id(conversation_id)
        if conversation and not getattr(conversation, "planning_mode_enabled", False):
            update = ConversationUpdate(planning_mode_enabled=True)
            self.conversation_repository.update(conversation_id, update)

    @staticmethod
    def _normalize_description(description: Any) -> str:
        text = str(description or "")
        return " ".join(text.split())

    def _normalize_descriptions(self, descriptions: List[str]) -> List[str]:
        normalized = [
            self._normalize_description(description) for description in descriptions or []
        ]
        normalized = [description for description in normalized if description]
        if not normalized:
            raise ValueError("At least one non-empty task description is required")
        return normalized

    @staticmethod
    def _coerce_status(
        raw_status: Any, default: TaskStatus = TaskStatus.pending
    ) -> TaskStatus:
        if isinstance(raw_status, TaskStatus):
            return raw_status
        if hasattr(raw_status, "value"):
            raw_status = raw_status.value
        if isinstance(raw_status, str):
            normalized = raw_status.strip().lower()
            try:
                return TaskStatus(normalized)
            except ValueError:
                return default
        return default

    def _normalize_todos(self, todos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        active_task_seen = False

        sortable = todos if isinstance(todos, list) else []
        sortable = sorted(
            sortable,
            key=lambda item: int(item.get("order", item.get("task_order", len(normalized))))
            if str(item.get("order", item.get("task_order", ""))).strip("-").isdigit()
            else len(normalized),
        )

        for raw_todo in sortable:
            if not isinstance(raw_todo, dict):
                continue

            description = self._normalize_description(raw_todo.get("description"))
            if not description:
                continue

            todo_id = str(raw_todo.get("id") or "").strip()
            if todo_id and todo_id in seen_ids:
                continue
            if todo_id:
                seen_ids.add(todo_id)

            status = self._coerce_status(raw_todo.get("status"), TaskStatus.pending)
            if status == TaskStatus.in_progress:
                if active_task_seen:
                    status = TaskStatus.pending
                else:
                    active_task_seen = True

            normalized.append(
                {
                    "id": todo_id or None,
                    "description": description,
                    "status": status,
                    "order": len(normalized),
                }
            )

        return normalized

    @staticmethod
    def _task_to_agent_dict(task: TaskPlanRead | TaskPlan) -> Dict[str, Any]:
        return {
            "id": str(task.id),
            "description": task.description,
            "status": task.status.value if hasattr(task.status, "value") else str(task.status),
            "task_order": task.task_order,
            "order": task.task_order,
        }

    @staticmethod
    def _set_task_status(
        task: TaskPlan, status: TaskStatus, *, now: datetime
    ) -> None:
        task.status = status
        task.updated_at = now
        if status == TaskStatus.completed:
            task.completed_at = now
        else:
            task.completed_at = None

    def _append_descriptions(
        self, conversation_id: UUID, descriptions: List[str]
    ) -> List[TaskPlanRead]:
        normalized = self._normalize_descriptions(descriptions)
        now = datetime.now(timezone.utc)

        with self.task_plan_repository.session_factory() as session:
            max_task_order = (
                session.query(TaskPlan.task_order)
                .filter(TaskPlan.conversation_id == conversation_id)
                .order_by(TaskPlan.task_order.desc())
                .limit(1)
                .scalar()
            )
            next_order = 0 if max_task_order is None else max_task_order + 1

            created_tasks: List[TaskPlan] = []
            for offset, description in enumerate(normalized):
                task = TaskPlan(
                    conversation_id=conversation_id,
                    task_order=next_order + offset,
                    description=description,
                    status=TaskStatus.pending,
                    task_metadata={},
                    created_at=now,
                    updated_at=now,
                    completed_at=None,
                )
                session.add(task)
                created_tasks.append(task)

            session.commit()

            for task in created_tasks:
                session.refresh(task)

            return [TaskPlanRead.model_validate(task) for task in created_tasks]

    def _sync_todo_snapshot(
        self,
        conversation_id: UUID,
        todos: List[Dict[str, Any]],
        *,
        preserve_existing_status: bool,
    ) -> List[TaskPlanRead]:
        normalized_todos = self._normalize_todos(todos)
        now = datetime.now(timezone.utc)

        with self.task_plan_repository.session_factory() as session:
            existing_tasks = (
                session.query(TaskPlan)
                .filter(TaskPlan.conversation_id == conversation_id)
                .order_by(TaskPlan.task_order.asc())
                .all()
            )

            existing_by_id = {str(task.id): task for task in existing_tasks}
            seen_task_ids: set[str] = set()
            active_task_seen = False

            for order, todo in enumerate(normalized_todos):
                todo_id = todo.get("id")
                task = existing_by_id.get(todo_id) if todo_id else None

                if task is not None:
                    seen_task_ids.add(str(task.id))
                    task.task_order = order
                    task.description = todo["description"]
                    task.updated_at = now

                    status = task.status if preserve_existing_status else todo["status"]
                    if status == TaskStatus.in_progress:
                        if active_task_seen:
                            status = TaskStatus.pending
                        else:
                            active_task_seen = True
                    self._set_task_status(task, status, now=now)
                    continue

                status = TaskStatus.pending if preserve_existing_status else todo["status"]
                if status == TaskStatus.in_progress:
                    if active_task_seen:
                        status = TaskStatus.pending
                    else:
                        active_task_seen = True

                task = TaskPlan(
                    conversation_id=conversation_id,
                    task_order=order,
                    description=todo["description"],
                    status=status,
                    task_metadata={},
                    created_at=now,
                    updated_at=now,
                    completed_at=now if status == TaskStatus.completed else None,
                )
                session.add(task)

            for task in existing_tasks:
                if str(task.id) not in seen_task_ids:
                    session.delete(task)

            session.commit()

            stored_tasks = (
                session.query(TaskPlan)
                .filter(TaskPlan.conversation_id == conversation_id)
                .order_by(TaskPlan.task_order.asc())
                .all()
            )
            return [TaskPlanRead.model_validate(task) for task in stored_tasks]

    @staticmethod
    def _extract_agent_todos(response: Any) -> List[Dict[str, Any]]:
        metadata = getattr(response, "metadata", None) or {}
        todos = metadata.get("todos")
        if not isinstance(todos, list):
            raise ValueError("Planning agent did not return a todo payload")
        return todos

    async def create_task_plan(
        self,
        conversation_id: UUID,
        user_message: str,
        user_id: UUID,
    ) -> List[TaskPlanRead]:
        self.conversation_validation_utils.validate_conversation_access(
            user_id, conversation_id
        )

        existing_tasks = self.task_plan_repository.get_by_conversation_id(
            conversation_id, include_completed=True
        )
        if existing_tasks:
            return await self.modify_task_plan(conversation_id, user_message, user_id)

        agent_message = AgentMessage(
            role=MessageRole.USER,
            content=user_message,
            metadata={},
        )

        response = await self.planning_agent.generate_plan(
            message=agent_message,
            conversation_id=str(conversation_id),
        )

        if response.error:
            raise ValueError(response.error)

        todos = self._extract_agent_todos(response)
        if not self._normalize_todos(todos):
            raise ValueError("Failed to generate task plan from the request")

        created_tasks = self.sync_todos_from_agent(
            conversation_id=conversation_id,
            todos=todos,
            user_id=user_id,
            preserve_existing_status=False,
        )
        self._ensure_planning_mode_enabled(conversation_id)
        return created_tasks

    async def modify_task_plan(
        self,
        conversation_id: UUID,
        user_message: str,
        user_id: UUID,
    ) -> List[TaskPlanRead]:
        self.conversation_validation_utils.validate_conversation_access(
            user_id, conversation_id
        )

        existing_tasks = self.task_plan_repository.get_by_conversation_id(
            conversation_id, include_completed=True
        )

        if not existing_tasks:
            return await self.create_task_plan(conversation_id, user_message, user_id)

        existing_tasks_dict = [
            self._task_to_agent_dict(task) for task in existing_tasks
        ]

        agent_message = AgentMessage(
            role=MessageRole.USER,
            content=user_message,
            metadata={"existing_tasks": existing_tasks_dict},
        )

        response = await self.planning_agent.modify_plan(
            message=agent_message,
            existing_tasks=existing_tasks_dict,
            conversation_id=str(conversation_id),
        )

        if response.error:
            raise ValueError(response.error)

        todos = self._extract_agent_todos(response)
        if not self._normalize_todos(todos):
            raise ValueError("Failed to modify task plan from the request")

        updated_tasks = self.sync_todos_from_agent(
            conversation_id=conversation_id,
            todos=todos,
            user_id=user_id,
            preserve_existing_status=True,
        )
        self._ensure_planning_mode_enabled(conversation_id)
        return updated_tasks

    def sync_todos_from_agent(
        self,
        conversation_id: UUID,
        todos: List[Dict[str, Any]],
        user_id: UUID,
        preserve_existing_status: bool = False,
    ) -> List[TaskPlanRead]:
        self.conversation_validation_utils.validate_conversation_access(
            user_id, conversation_id
        )
        tasks = self._sync_todo_snapshot(
            conversation_id=conversation_id,
            todos=todos,
            preserve_existing_status=preserve_existing_status,
        )
        self._ensure_planning_mode_enabled(conversation_id)
        return tasks

    def create_task_plan_from_list(
        self,
        conversation_id: UUID,
        task_descriptions: List[str],
        user_id: UUID,
    ) -> List[TaskPlanRead]:
        self.conversation_validation_utils.validate_conversation_access(
            user_id, conversation_id
        )
        created_tasks = self._append_descriptions(conversation_id, task_descriptions)
        self._ensure_planning_mode_enabled(conversation_id)
        return created_tasks

    def get_by_id(self, task_id: UUID, user_id: UUID) -> TaskPlanRead:
        self.task_plan_validation_utils.validate_task_access(user_id, task_id)
        task = self.task_plan_repository.get_by_id(task_id)
        return TaskPlanRead.model_validate(task)

    def get_conversation_tasks(
        self,
        conversation_id: UUID,
        user_id: UUID,
        include_completed: bool = False,
    ) -> List[TaskPlanRead]:
        self.conversation_validation_utils.validate_conversation_access(
            user_id, conversation_id
        )

        tasks = self.task_plan_repository.get_by_conversation_id(
            conversation_id, include_completed=include_completed
        )
        return [TaskPlanRead.model_validate(task) for task in tasks]

    def get_active_or_next_task(
        self, conversation_id: UUID, user_id: UUID
    ) -> Optional[TaskPlanRead]:
        self.conversation_validation_utils.validate_conversation_access(
            user_id, conversation_id
        )

        task = self.task_plan_repository.get_active_or_next_task(conversation_id)
        if task:
            return TaskPlanRead.model_validate(task)
        return None

    def update_task(
        self,
        task_id: UUID,
        user_id: UUID,
        task_update_data: TaskPlanUpdate,
    ) -> TaskPlanRead:
        self.task_plan_validation_utils.validate_task_access(user_id, task_id)

        updated_task = self.task_plan_repository.update(task_id, task_update_data)
        if not updated_task:
            raise ResourceNotFoundException(
                detail="Task plan not found",
                error_code="TASK_PLAN_NOT_FOUND",
            )

        return TaskPlanRead.model_validate(updated_task)

    def mark_task_completed(self, task_id: UUID, user_id: UUID) -> TaskPlanRead:
        self.task_plan_validation_utils.validate_task_access(user_id, task_id)

        updated_task = self.task_plan_repository.mark_completed(task_id)
        if not updated_task:
            raise ResourceNotFoundException(
                detail="Task plan not found",
                error_code="TASK_PLAN_NOT_FOUND",
            )

        return TaskPlanRead.model_validate(updated_task)

    def mark_task_in_progress(self, task_id: UUID, user_id: UUID) -> TaskPlanRead:
        self.task_plan_validation_utils.validate_task_access(user_id, task_id)

        now = datetime.now(timezone.utc)
        with self.task_plan_repository.session_factory() as session:
            task = session.query(TaskPlan).filter(TaskPlan.id == task_id).first()
            if not task:
                raise ResourceNotFoundException(
                    detail="Task plan not found",
                    error_code="TASK_PLAN_NOT_FOUND",
                )

            (
                session.query(TaskPlan)
                .filter(
                    TaskPlan.conversation_id == task.conversation_id,
                    TaskPlan.id != task_id,
                    TaskPlan.status == TaskStatus.in_progress,
                )
                .update(
                    {
                        TaskPlan.status: TaskStatus.pending,
                        TaskPlan.completed_at: None,
                        TaskPlan.updated_at: now,
                    },
                    synchronize_session=False,
                )
            )

            self._set_task_status(task, TaskStatus.in_progress, now=now)
            session.commit()
            session.refresh(task)
            return TaskPlanRead.model_validate(task)

    def delete_task(self, task_id: UUID, user_id: UUID) -> bool:
        self.task_plan_validation_utils.validate_task_access(user_id, task_id)
        return self.task_plan_repository.delete(task_id)

    def get_planning_status(
        self, conversation_id: UUID, user_id: UUID
    ) -> PlanningStatusResponse:
        self.conversation_validation_utils.validate_conversation_access(
            user_id, conversation_id
        )

        conversation: Optional[Conversation] = self.conversation_repository.get_by_id(
            conversation_id
        )
        planning_mode_enabled = (
            conversation.planning_mode_enabled if conversation else False
        )

        total_tasks = self.task_plan_repository.count_by_conversation(conversation_id)
        pending_tasks = self.task_plan_repository.count_by_conversation(
            conversation_id, status=TaskStatus.pending
        )
        in_progress_tasks = self.task_plan_repository.count_by_conversation(
            conversation_id, status=TaskStatus.in_progress
        )
        completed_tasks = self.task_plan_repository.count_by_conversation(
            conversation_id, status=TaskStatus.completed
        )
        skipped_tasks = self.task_plan_repository.count_by_conversation(
            conversation_id, status=TaskStatus.skipped
        )

        if not planning_mode_enabled and total_tasks > 0:
            self._ensure_planning_mode_enabled(conversation_id)
            planning_mode_enabled = True

        progress_percentage = 0.0
        if total_tasks > 0:
            progress_percentage = (completed_tasks / total_tasks) * 100

        next_task = self.task_plan_repository.get_active_or_next_task(conversation_id)
        next_task_read = TaskPlanRead.model_validate(next_task) if next_task else None

        return PlanningStatusResponse(
            planning_mode_enabled=planning_mode_enabled,
            total_tasks=total_tasks,
            pending_tasks=pending_tasks,
            in_progress_tasks=in_progress_tasks,
            completed_tasks=completed_tasks,
            skipped_tasks=skipped_tasks,
            progress_percentage=round(progress_percentage, 2),
            next_task=next_task_read,
        )
