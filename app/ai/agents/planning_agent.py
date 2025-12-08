from typing import Optional, List, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI

from ..schemas import AgentMessage, AgentResponse, AgentType, MessageRole, Task, Plan
from ..prompts import build_planning_prompt
from ..utils import coerce_response_text
from ...core.config import settings


PLAN_MODIFICATION_PROMPT = """You are a planning assistant. The user wants to modify an existing task plan.

CURRENT PLAN:
{current_plan}

USER REQUEST:
{user_request}

INSTRUCTIONS:
1. Analyze the user's request to understand what changes they want
2. Apply the requested modifications to the plan
3. Supported operations:
   - Add new tasks (place them in appropriate order with correct dependencies)
   - Remove tasks (update dependencies of tasks that depended on removed tasks)
   - Modify task descriptions
   - Reorder tasks (update dependencies accordingly)
   - Update task dependencies
4. Ensure the modified plan is valid (no circular dependencies, valid dependency indices)
5. Return the complete modified plan

Return the updated plan with all tasks."""


class PlanningAgent:
    def __init__(self):
        self.model_name = settings.chat_agent_model
        self.langchain_model = None
        self._init_model()

    def _init_model(self):
        api_key = settings.gemini_api_key
        if not api_key:
            return

        if api_key.startswith("GEMINI_API_KEY="):
            api_key = api_key.split("=", 1)[-1].strip()

        try:
            self.langchain_model = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=api_key,
                temperature=0.7,
            )
        except Exception:
            pass

    async def generate_plan(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
    ) -> AgentResponse:
        if not self.langchain_model:
            return AgentResponse(
                agent_type=AgentType.PLANNING,
                agent_id="planning_agent",
                message=AgentMessage(
                    role=MessageRole.ASSISTANT,
                    content="I'm sorry, but I'm unable to generate a plan at the moment. The planning service is not properly configured.",
                ),
                metadata={"error": "Model not initialized"},
                error="Model not initialized",
            )

        conversation_history = message.metadata.get("history", [])
        persona = message.metadata.get("persona")

        prompt = build_planning_prompt(
            message.content, conversation_history, persona=persona
        )

        try:
            structured_llm = self.langchain_model.with_structured_output(Plan)
            plan: Plan = await structured_llm.ainvoke(prompt)

            validated_tasks = self._validate_and_order_tasks(plan.tasks)
            plan.tasks = validated_tasks

            formatted_response = self._format_plan_response(plan)

            return AgentResponse(
                agent_type=AgentType.PLANNING,
                agent_id="planning_agent",
                message=AgentMessage(
                    role=MessageRole.ASSISTANT,
                    content=coerce_response_text(formatted_response),
                ),
                metadata={
                    "model": self.model_name,
                    "conversation_id": conversation_id,
                    "plan": plan.model_dump(),
                    "task_count": len(plan.tasks),
                },
            )

        except ValueError as e:
            return AgentResponse(
                agent_type=AgentType.PLANNING,
                agent_id="planning_agent",
                message=AgentMessage(
                    role=MessageRole.ASSISTANT,
                    content=f"I encountered an issue while creating the plan: {str(e)}. Please try rephrasing your request.",
                ),
                metadata={
                    "model": self.model_name,
                    "conversation_id": conversation_id,
                    "error": str(e),
                },
                error=str(e),
            )

        except Exception as e:
            return AgentResponse(
                agent_type=AgentType.PLANNING,
                agent_id="planning_agent",
                message=AgentMessage(
                    role=MessageRole.ASSISTANT,
                    content=f"I encountered an error while generating the plan: {str(e)}",
                ),
                metadata={
                    "model": self.model_name,
                    "conversation_id": conversation_id,
                    "error": str(e),
                },
                error=str(e),
            )

    async def modify_plan(
        self,
        message: AgentMessage,
        existing_tasks: List[Dict[str, Any]],
        conversation_id: Optional[str] = None,
    ) -> AgentResponse:
        if not self.langchain_model:
            return AgentResponse(
                agent_type=AgentType.PLANNING,
                agent_id="planning_agent",
                message=AgentMessage(
                    role=MessageRole.ASSISTANT,
                    content="I'm sorry, but I'm unable to modify the plan at the moment. The planning service is not properly configured.",
                ),
                metadata={"error": "Model not initialized"},
                error="Model not initialized",
            )

        current_plan_str = self._format_existing_tasks(existing_tasks)

        prompt = PLAN_MODIFICATION_PROMPT.format(
            current_plan=current_plan_str,
            user_request=message.content,
        )

        try:
            structured_llm = self.langchain_model.with_structured_output(Plan)
            modified_plan: Plan = await structured_llm.ainvoke(prompt)

            validated_tasks = self._validate_and_order_tasks(modified_plan.tasks)
            modified_plan.tasks = validated_tasks

            formatted_response = self._format_modification_response(
                existing_tasks, modified_plan
            )

            return AgentResponse(
                agent_type=AgentType.PLANNING,
                agent_id="planning_agent",
                message=AgentMessage(
                    role=MessageRole.ASSISTANT,
                    content=coerce_response_text(formatted_response),
                ),
                metadata={
                    "model": self.model_name,
                    "conversation_id": conversation_id,
                    "plan": modified_plan.model_dump(),
                    "task_count": len(modified_plan.tasks),
                    "plan_modified": True,
                },
            )

        except ValueError as e:
            return AgentResponse(
                agent_type=AgentType.PLANNING,
                agent_id="planning_agent",
                message=AgentMessage(
                    role=MessageRole.ASSISTANT,
                    content=f"I encountered an issue while modifying the plan: {str(e)}. Please try rephrasing your request.",
                ),
                metadata={
                    "model": self.model_name,
                    "conversation_id": conversation_id,
                    "error": str(e),
                },
                error=str(e),
            )

        except Exception as e:
            return AgentResponse(
                agent_type=AgentType.PLANNING,
                agent_id="planning_agent",
                message=AgentMessage(
                    role=MessageRole.ASSISTANT,
                    content=f"I encountered an error while modifying the plan: {str(e)}",
                ),
                metadata={
                    "model": self.model_name,
                    "conversation_id": conversation_id,
                    "error": str(e),
                },
                error=str(e),
            )

    def _format_existing_tasks(self, existing_tasks: List[Dict[str, Any]]) -> str:
        if not existing_tasks:
            return "No existing tasks."

        lines = []
        for i, task in enumerate(existing_tasks):
            desc = task.get("description", "No description")
            status = task.get("status", "pending")
            deps = task.get("dependencies", [])
            dep_str = (
                f" (depends on: {', '.join(str(d) for d in deps)})" if deps else ""
            )
            lines.append(f"Task {i + 1}: {desc} [{status}]{dep_str}")

        return "\n".join(lines)

    def _format_modification_response(
        self, original_tasks: List[Dict[str, Any]], modified_plan: Plan
    ) -> str:
        parts = []

        original_count = len(original_tasks)
        new_count = len(modified_plan.tasks)

        if new_count > original_count:
            parts.append(
                f"I've updated the plan (added {new_count - original_count} task(s)):"
            )
        elif new_count < original_count:
            parts.append(
                f"I've updated the plan (removed {original_count - new_count} task(s)):"
            )
        else:
            parts.append("I've updated the plan:")

        parts.append("")

        if modified_plan.overall_goal:
            parts.append(f"**Goal:** {modified_plan.overall_goal}")
            parts.append("")

        for i, task in enumerate(modified_plan.tasks):
            task_line = f"**Task {i + 1}:** {task.description}"

            details = []
            if task.estimated_complexity:
                details.append(f"Complexity: {task.estimated_complexity}")
            if task.dependencies:
                dep_str = ", ".join(f"Task {d + 1}" for d in task.dependencies)
                details.append(f"Depends on: {dep_str}")

            if details:
                task_line += f" ({', '.join(details)})"

            parts.append(task_line)
            parts.append("")

        return "\n".join(parts)

    def _validate_and_order_tasks(self, tasks: List[Task]) -> List[Task]:
        if not tasks:
            return tasks

        num_tasks = len(tasks)

        for i, task in enumerate(tasks):
            for dep_idx in task.dependencies:
                if dep_idx < 0 or dep_idx >= num_tasks:
                    raise ValueError(
                        f"Task {i} has invalid dependency index {dep_idx}. "
                        f"Valid indices are 0 to {num_tasks - 1}."
                    )
                if dep_idx >= i:
                    raise ValueError(
                        f"Task {i} depends on task {dep_idx}, but dependencies "
                        f"must reference earlier tasks (lower indices)."
                    )

        visited = [False] * num_tasks
        rec_stack = [False] * num_tasks

        def has_cycle(node: int) -> bool:
            visited[node] = True
            rec_stack[node] = True

            for dep_idx in tasks[node].dependencies:
                if not visited[dep_idx]:
                    if has_cycle(dep_idx):
                        return True
                elif rec_stack[dep_idx]:
                    return True

            rec_stack[node] = False
            return False

        for i in range(num_tasks):
            if not visited[i]:
                if has_cycle(i):
                    raise ValueError(
                        "Circular dependencies detected in the task plan. "
                        "Please ensure tasks are ordered correctly."
                    )

        return tasks

    def _format_plan_response(self, plan: Plan) -> str:
        parts = []

        if plan.overall_goal:
            parts.append(f"**Goal:** {plan.overall_goal}")
            parts.append("")

        parts.append(f"I've created a plan with {len(plan.tasks)} tasks:")
        parts.append("")

        for i, task in enumerate(plan.tasks):
            task_line = f"**Task {i + 1}:** {task.description}"

            details = []
            if task.estimated_complexity:
                details.append(f"Complexity: {task.estimated_complexity}")
            if task.dependencies:
                dep_str = ", ".join(f"Task {d + 1}" for d in task.dependencies)
                details.append(f"Depends on: {dep_str}")

            if details:
                task_line += f" ({', '.join(details)})"

            parts.append(task_line)
            parts.append("")

        return "\n".join(parts)
