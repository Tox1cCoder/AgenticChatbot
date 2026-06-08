from typing import Any

from app.ai.agents.planning_agent import PlanningAgent
from app.ai.schemas import AgentMessage, MessageRole
from app.interfaces.planning_runtime_interface import IPlanningRuntimeService
from app.schemas.task_plan import PlanningRuntimeRequest, PlanningRuntimeResult


class PlanningRuntimeAdapter(IPlanningRuntimeService):
    """AI-side adapter that translates service planning requests into planning-agent calls."""

    def __init__(self, planning_agent: PlanningAgent):
        self.planning_agent = planning_agent

    @staticmethod
    def _build_agent_message(request: PlanningRuntimeRequest) -> AgentMessage:
        metadata: dict[str, Any] = {}
        if request.user_id:
            metadata["user_id"] = request.user_id
        if request.existing_tasks:
            metadata["existing_tasks"] = request.existing_tasks

        return AgentMessage(
            role=MessageRole.USER,
            content=request.user_message,
            metadata=metadata,
        )

    @staticmethod
    def _extract_todos(response: Any) -> list[dict[str, Any]]:
        metadata = getattr(response, "metadata", None) or {}
        todos = metadata.get("todos")
        if not isinstance(todos, list):
            raise ValueError("Planning agent did not return a todo payload")
        return todos

    async def generate_plan(self, request: PlanningRuntimeRequest) -> PlanningRuntimeResult:
        response = await self.planning_agent.generate_plan(
            message=self._build_agent_message(request),
            conversation_id=request.conversation_id,
        )
        if response.error:
            raise ValueError(response.error)
        return PlanningRuntimeResult(
            todos=self._extract_todos(response),
            metadata=dict(getattr(response, "metadata", None) or {}),
        )

    async def modify_plan(self, request: PlanningRuntimeRequest) -> PlanningRuntimeResult:
        response = await self.planning_agent.modify_plan(
            message=self._build_agent_message(request),
            existing_tasks=request.existing_tasks,
            conversation_id=request.conversation_id,
        )
        if response.error:
            raise ValueError(response.error)
        return PlanningRuntimeResult(
            todos=self._extract_todos(response),
            metadata=dict(getattr(response, "metadata", None) or {}),
        )
