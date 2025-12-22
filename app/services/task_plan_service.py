from typing import Optional, List, Dict, Any, TYPE_CHECKING
from uuid import UUID

from app.repositories.task_plan import TaskPlanRepository
from app.repositories.conversation import ConversationRepository
from app.utils.validation.conversation_validation import ConversationValidationUtils
from app.utils.validation.task_plan_validation import TaskPlanValidationUtils
from app.interfaces.task_plan_service_interface import ITaskPlanService
from app.schemas.conversation import ConversationUpdate
from app.schemas.task_plan import (
    TaskPlanUpdate,
    TaskPlanRead,
    PlanningStatusResponse,
)
from app.factories.task_plan_factory import TaskPlanFactory
from app.ai.agents.planning_agent import PlanningAgent
from app.ai.schemas import AgentMessage, MessageRole
from app.models.enums import TaskStatus
from app.core.exceptions import ResourceNotFoundException
from app.ai.schemas import Plan, Task

if TYPE_CHECKING:
    from app.ai.schemas import Plan


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

    def _build_plan_from_payload(self, plan_payload: Dict[str, Any]) -> "Plan":
        tasks = [Task(**t) for t in plan_payload.get("tasks", [])]
        return Plan(tasks=tasks, overall_goal=plan_payload.get("overall_goal"))

    def _persist_plan(
        self,
        conversation_id: UUID,
        plan: "Plan",
        replace_existing: bool,
    ) -> List[TaskPlanRead]:
        if replace_existing:
            existing_tasks = self.task_plan_repository.get_by_conversation_id(
                conversation_id, include_completed=True
            )
            for old_task in existing_tasks:
                self.task_plan_repository.delete(old_task.id)

        task_entities = TaskPlanFactory.create_batch_from_plan(conversation_id, plan)

        created_tasks = []
        for task_data in task_entities:
            created_task = self.task_plan_repository.create(task_data)
            created_tasks.append(TaskPlanRead.model_validate(created_task))

        return created_tasks

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

        if response.error or not response.metadata.get("plan"):
            raise ValueError(
                response.error or "Failed to generate task plan from the request"
            )

        plan_data = response.metadata.get("plan", {})

        plan = self._build_plan_from_payload(plan_data)
        created_tasks = self._persist_plan(
            conversation_id, plan, replace_existing=False
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
            {
                "id": str(task.id),
                "description": task.description,
                "status": (
                    task.status.value
                    if hasattr(task.status, "value")
                    else str(task.status)
                ),
                "task_order": task.task_order,
                "dependencies": [str(d) for d in (task.dependencies or [])],
            }
            for task in existing_tasks
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

        if response.error or not response.metadata.get("plan"):
            raise ValueError(
                response.error or "Failed to modify task plan from the request"
            )

        plan_data = response.metadata.get("plan", {})

        modified_plan = self._build_plan_from_payload(plan_data)
        created_tasks = self._persist_plan(
            conversation_id, modified_plan, replace_existing=True
        )
        self._ensure_planning_mode_enabled(conversation_id)

        return created_tasks

    def sync_plan_from_agent(
        self,
        conversation_id: UUID,
        plan_payload: Dict[str, Any],
        user_id: UUID,
        replace_existing: bool = True,
    ) -> List[TaskPlanRead]:
        if not plan_payload:
            return []

        self.conversation_validation_utils.validate_conversation_access(
            user_id, conversation_id
        )

        plan = self._build_plan_from_payload(plan_payload)
        tasks = self._persist_plan(
            conversation_id, plan, replace_existing=replace_existing
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

        task_entities = TaskPlanFactory.create_from_descriptions(
            conversation_id, task_descriptions
        )

        created_tasks = []
        for task_data in task_entities:
            created_task = self.task_plan_repository.create(task_data)
            created_tasks.append(TaskPlanRead.model_validate(created_task))

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

    def get_next_task(
        self, conversation_id: UUID, user_id: UUID
    ) -> Optional[TaskPlanRead]:
        self.conversation_validation_utils.validate_conversation_access(
            user_id, conversation_id
        )

        task = self.task_plan_repository.get_next_task(conversation_id)
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

        task = self.task_plan_repository.get_by_id(task_id)

        if task_update_data.dependencies is not None:
            self.task_plan_validation_utils.validate_dependencies(
                task.conversation_id, task_update_data.dependencies
            )
            self.task_plan_validation_utils.validate_no_circular_dependencies(
                task.conversation_id, task_id, task_update_data.dependencies
            )

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

    def delete_task(self, task_id: UUID, user_id: UUID) -> bool:
        self.task_plan_validation_utils.validate_task_access(user_id, task_id)
        return self.task_plan_repository.delete(task_id)

    def get_planning_status(
        self, conversation_id: UUID, user_id: UUID
    ) -> PlanningStatusResponse:
        self.conversation_validation_utils.validate_conversation_access(
            user_id, conversation_id
        )

        conversation = self.conversation_repository.get_by_id(conversation_id)
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

        next_task = self.task_plan_repository.get_next_task(conversation_id)
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
