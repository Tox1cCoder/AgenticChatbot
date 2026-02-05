from __future__ import annotations

import uuid
from textwrap import dedent
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from ...core.config import settings
from ..planning_tools import create_write_todos_tool
from ..prompts import PLANNING_EXECUTION_PROMPT
from ..schemas import (
    AgentMessage,
    AgentResponse,
    AgentType,
    MessageRole,
    Plan,
    Task,
    TodoStatus,
)
from ..todo_actions import apply_write_todos_action
from ..utils import coerce_response_text, normalize_tool_call
from .base_agent import BaseAgent


class PlanningAgent(BaseAgent):
    """Planning agent that manages task plans using ReAct-style tool-calling."""

    def __init__(self, model_name: Optional[str] = None):
        """Initialize Planning Agent with 'planning' config key."""
        super().__init__(model_name=model_name, agent_config_key="planning")

    @property
    def agent_type(self) -> AgentType:
        return AgentType.PLANNING

    @property
    def agent_id(self) -> str:
        return "planning_agent"

    def _get_base_system_prompt(self) -> str:
        return PLANNING_EXECUTION_PROMPT

    def _get_llm_with_tools(
        self,
        model: Any = None,
        conversation_id: Optional[str] = None,
        internal_tools: Optional[List[BaseTool]] = None,
    ) -> Any:
        """
        Override to ensure write_todos is always included as an internal tool.

        This guarantees write_todos is available even in deferred tool loading mode,
        where only tool_search + pinned tools + loaded tools would normally be bound.
        """
        write_todos_tool = create_write_todos_tool()
        combined_internal = [write_todos_tool]

        if internal_tools:
            # Add any additional internal tools, avoiding duplicates
            for tool in internal_tools:
                if tool.name != write_todos_tool.name:
                    combined_internal.append(tool)

        return super()._get_llm_with_tools(
            model=model,
            conversation_id=conversation_id,
            internal_tools=combined_internal,
        )

    async def _init_tools(self):
        # Initialize MCP tools from parent.
        await super()._init_tools()

        # Ensure write_todos is available (and first).
        write_todos_tool = create_write_todos_tool()
        tool_names = {tool.name for tool in self.tools}
        if write_todos_tool.name not in tool_names:
            self.tools.insert(0, write_todos_tool)

    def _build_system_prompt(
        self,
        persona: Optional[str] = None,
        has_tool_context: bool = False,
        todos: Optional[List[Dict[str, Any]]] = None,
        current_task_index: Optional[int] = None,
        planning_phase: Optional[str] = None,
        should_describe_plan: bool = False,
    ) -> str:
        base_prompt = super()._build_system_prompt(persona, has_tool_context)

        phase = planning_phase or "planning"
        if phase == "planning":
            phase_prompt = dedent(
                """
                # CURRENT PHASE: PLANNING
                - Create or modify the task plan using: set_todos, add_todo, update_todo, remove_todo
                - Do NOT execute tasks: no start_todo or complete_todo
                - Ask for confirmation before starting execution

                When the user explicitly asks to start/execute/implement:
                - Begin execution by calling start_todo for the next task
                """
            ).strip()
        else:
            phase_prompt = dedent(
                """
                # CURRENT PHASE: EXECUTING
                Work through tasks autonomously:
                1) Find the next pending task
                2) Call start_todo
                3) Do the work
                4) Call complete_todo
                5) Continue to the next task

                Stop only when all tasks are completed (then summarize), you need user clarification,
                or an unresolvable error occurs.
                """
            ).strip()

        prompt = f"{base_prompt}\n\n{phase_prompt}"

        if should_describe_plan:
            prompt += (
                "\n\n"
                + dedent(
                    """
                # IMPORTANT: You just created or modified the plan.
                Now respond with text only:
                - Summarize the tasks you created/modified
                - Ask if the user wants changes before starting execution
                - Do NOT make any tool calls in this response
                """
                ).strip()
            )

        if todos:
            prompt += "\n\n" + self._format_todos_context(todos, current_task_index)

        return prompt

    def _format_todos_context(
        self, todos: List[Dict[str, Any]], current_task_index: Optional[int] = None
    ) -> str:
        if not todos:
            return ""

        lines = ["CURRENT TASK PLAN:"]
        lines.append(
            "(Use the ID shown in brackets when calling start_todo/complete_todo)"
        )
        lines.append("")

        for i, todo in enumerate(todos):
            raw_status = todo.get("status", TodoStatus.PENDING.value)
            status = raw_status.value if hasattr(raw_status, "value") else raw_status

            desc = todo.get("description", "No description")
            todo_id = todo.get("id", str(i + 1))

            if status == TodoStatus.COMPLETED.value:
                indicator = "[x]"
            elif status == TodoStatus.IN_PROGRESS.value:
                indicator = "[~]"
            elif status == TodoStatus.SKIPPED.value:
                indicator = "[-]"
            else:
                indicator = "[ ]"

            current_marker = " ← CURRENT TASK" if i == current_task_index else ""
            lines.append(
                f"{indicator} Task {i + 1} [ID: {todo_id}]: {desc} ({status}){current_marker}"
            )

        completed = sum(
            1
            for t in todos
            if (
                getattr(t.get("status"), "value", t.get("status"))
                == TodoStatus.COMPLETED.value
            )
        )
        lines.append(f"\nProgress: {completed}/{len(todos)} tasks completed")

        return "\n".join(lines)

    async def generate_plan(
        self, message: AgentMessage, conversation_id: Optional[str] = None
    ) -> AgentResponse:
        """Generate a plan payload for persistence (used by TaskPlanService)."""
        return await self._generate_or_modify_plan(
            message=message,
            existing_tasks=None,
            conversation_id=conversation_id,
            plan_modified=False,
        )

    async def modify_plan(
        self,
        message: AgentMessage,
        existing_tasks: List[Dict[str, Any]],
        conversation_id: Optional[str] = None,
    ) -> AgentResponse:
        """Modify an existing plan payload for persistence (used by TaskPlanService)."""
        return await self._generate_or_modify_plan(
            message=message,
            existing_tasks=existing_tasks,
            conversation_id=conversation_id,
            plan_modified=True,
        )

    async def _generate_or_modify_plan(
        self,
        *,
        message: AgentMessage,
        existing_tasks: Optional[List[Dict[str, Any]]],
        conversation_id: Optional[str],
        plan_modified: bool,
    ) -> AgentResponse:
        if not self.langchain_model:
            return self._build_error_response(
                "Planning service is not properly configured.",
                conversation_id,
            )

        persona = message.metadata.get("persona")
        conversation_history = message.metadata.get("history", [])

        # Convert existing tasks into todo-like dicts for prompt context.
        todos: List[Dict[str, Any]] = []
        current_task_index: Optional[int] = None
        if existing_tasks:
            for i, task in enumerate(existing_tasks):
                todos.append(
                    {
                        "id": task.get("id", str(i)),
                        "description": task.get("description", ""),
                        "status": task.get("status", TodoStatus.PENDING.value),
                        "order": task.get("order", task.get("task_order", i)),
                    }
                )
            current_task_index = next(
                (
                    i
                    for i, todo in enumerate(todos)
                    if todo.get("status")
                    in (TodoStatus.PENDING.value, TodoStatus.IN_PROGRESS.value)
                ),
                None,
            )

        system_prompt = self._build_system_prompt(
            persona=persona,
            has_tool_context=False,
            todos=todos or None,
            current_task_index=current_task_index,
            planning_phase="planning",
        )

        # Force a tool call so we can reliably extract a machine-readable plan.
        write_todos_tool = create_write_todos_tool()
        llm = self.langchain_model.bind_tools(
            [write_todos_tool],
            tool_config={"function_calling_config": {"mode": "ANY"}},
        )

        langchain_messages = [SystemMessage(content=system_prompt)]
        if conversation_history:
            langchain_messages.extend(
                self._convert_history_to_langchain_messages(conversation_history)
            )
        langchain_messages.append(HumanMessage(content=message.content))

        raw_response = await llm.ainvoke(langchain_messages)

        tool_calls = getattr(raw_response, "tool_calls", None) or []
        updated_todos = self._apply_write_todos_calls(
            base_todos=todos, tool_calls=tool_calls
        )

        if not updated_todos:
            fallback_text = coerce_response_text(getattr(raw_response, "content", ""))
            fallback_text = fallback_text.strip() if fallback_text else ""
            if not fallback_text:
                return self._build_error_response(
                    "I couldn't generate a valid plan from your request.",
                    conversation_id,
                )
            updated_todos = [
                {
                    "id": str(uuid.uuid4()),
                    "description": fallback_text,
                    "status": TodoStatus.PENDING.value,
                    "order": 0,
                }
            ]

        plan = self._todos_to_plan(
            todos=updated_todos,
            overall_goal=(
                (message.content.strip() or None) if not plan_modified else None
            ),
        )

        response_text = self._format_plan_summary(plan, plan_modified=plan_modified)

        metadata: Dict[str, Any] = {
            "model": self.model_name,
            "conversation_id": conversation_id,
            "plan": plan.model_dump(),
            "task_count": len(plan.tasks),
            "todos": updated_todos,
        }
        if plan_modified:
            metadata["plan_modified"] = True

        return AgentResponse(
            agent_type=self.agent_type,
            agent_id=self.agent_id,
            message=AgentMessage(
                role=MessageRole.ASSISTANT,
                content=response_text,
            ),
            metadata=metadata,
        )

    def _apply_write_todos_calls(
        self, *, base_todos: List[Dict[str, Any]], tool_calls: Iterable[Any]
    ) -> List[Dict[str, Any]]:
        max_todos = getattr(settings, "max_todos_per_plan", 50)
        todos = list(base_todos)

        for raw_tool_call in tool_calls:
            tool_call = normalize_tool_call(raw_tool_call)
            if tool_call.get("name") != "write_todos":
                continue

            try:
                todos, _, _, _ = apply_write_todos_action(
                    todos=todos,
                    current_task_index=None,
                    tool_args=tool_call.get("args"),
                    max_todos=max_todos,
                )
            except Exception:
                continue

        if len(todos) > max_todos:
            todos = todos[:max_todos]

        return todos

    def _todos_to_plan(
        self, *, todos: List[Dict[str, Any]], overall_goal: Optional[str]
    ) -> Plan:
        def sort_key(item: Dict[str, Any]) -> int:
            order = item.get("order")
            try:
                return int(order)
            except (TypeError, ValueError):
                return 0

        ordered = sorted(todos, key=sort_key) if todos else []

        seen: set[str] = set()
        tasks: List[Task] = []
        for item in ordered:
            desc = str(item.get("description", "")).strip()
            if not desc or desc in seen:
                continue
            seen.add(desc)
            tasks.append(Task(description=desc))

        return Plan(tasks=tasks, overall_goal=overall_goal)

    def _format_plan_summary(self, plan: Plan, *, plan_modified: bool) -> str:
        if not plan.tasks:
            return "I couldn't generate any actionable tasks."

        parts = [
            "I've updated the plan:" if plan_modified else "I've created a plan:",
            "",
        ]
        if plan.overall_goal:
            parts.append(f"Goal: {plan.overall_goal}")
            parts.append("")
        for i, task in enumerate(plan.tasks, start=1):
            parts.append(f"{i}. {task.description}")
        return "\n".join(parts)
