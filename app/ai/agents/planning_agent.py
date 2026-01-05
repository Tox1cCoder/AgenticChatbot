import logging
import uuid
from typing import Optional, List, Dict, Any

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage

from .base_agent import BaseAgent

from ..schemas import (
    AgentMessage,
    AgentResponse,
    AgentType,
    MessageRole,
    Task,
    Plan,
    TodoAction,
    TodoStatus,
    WriteTodosInput,
)

from ..prompts import build_planning_prompt, PLANNING_EXECUTION_PROMPT
from ..utils import coerce_response_text
from ...core.config import settings

logger = logging.getLogger(__name__)


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


def create_write_todos_tool():
    """Create the write_todos tool for task management."""

    @tool(args_schema=WriteTodosInput)
    def write_todos(
        action: TodoAction,
        todos: Optional[List[dict]] = None,
        todo: Optional[dict] = None,
        todo_id: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> str:
        """
        Manage the todo list for task planning. Use this tool to:
        - SET_TODOS: Replace all todos with a new list (when creating a plan)
        - ADD_TODO: Add a single new todo to the list
        - UPDATE_TODO: Update an existing todo's description or other properties
        - COMPLETE_TODO: Mark a todo as completed when done working on it
        - START_TODO: Mark a todo as in progress when starting work on it
        - REMOVE_TODO: Remove a todo from the list

        The tool returns a confirmation message. The actual state update
        is handled by the graph's tool execution node.
        """
        # This is a placeholder - actual execution happens in the graph's _planning_tools_node
        # The tool just validates and returns a confirmation
        if action == TodoAction.SET_TODOS:
            if not todos:
                return "Error: todos list is required for SET_TODOS action"
            return f"Setting {len(todos)} todos in the plan."

        elif action == TodoAction.ADD_TODO:
            if not todo:
                return "Error: todo is required for ADD_TODO action"
            return f"Adding todo: {todo.get('description', 'unknown')}"

        elif action == TodoAction.COMPLETE_TODO:
            if not todo_id:
                return "Error: todo_id is required for COMPLETE_TODO action"
            msg = f"Marking todo {todo_id} as completed."
            if reason:
                msg += f" Reason: {reason}"
            return msg

        elif action == TodoAction.START_TODO:
            if not todo_id:
                return "Error: todo_id is required for START_TODO action"
            return f"Starting work on todo {todo_id}."

        elif action == TodoAction.UPDATE_TODO:
            if not todo:
                return "Error: todo is required for UPDATE_TODO action"
            return f"Updating todo: {todo.get('id', 'unknown')}"

        elif action == TodoAction.REMOVE_TODO:
            if not todo_id:
                return "Error: todo_id is required for REMOVE_TODO action"
            return f"Removing todo {todo_id}."

        return f"Unknown action: {action}"

    return write_todos


class PlanningAgent(BaseAgent):
    """Planning agent that manages task plans using ReAct-style tool-calling."""

    @property
    def agent_type(self) -> AgentType:
        return AgentType.PLANNING

    @property
    def agent_id(self) -> str:
        return "planning_agent"

    def _get_base_system_prompt(self) -> str:
        return PLANNING_EXECUTION_PROMPT

    async def _init_tools(self):
        # First call parent to initialize MCP tools
        await super()._init_tools()

        # Add write_todos tool to the beginning of the tools list
        write_todos_tool = create_write_todos_tool()

        # Deduplicate by checking if write_todos already exists
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
        # Call parent to get base prompt with persona and tool context
        base_prompt = super()._build_system_prompt(persona, has_tool_context)

        # Add phase-specific instructions
        phase = planning_phase or "planning"
        
        if phase == "planning":
            phase_prompt = """
# CURRENT PHASE: PLANNING
You are in the PLANNING phase. Your role is to:
- Create or modify the task plan using SET_TODOS, ADD_TODO, UPDATE_TODO, or REMOVE_TODO
- DO NOT start executing tasks (no START_TODO or COMPLETE_TODO)
- Wait for the user to explicitly ask to implement/execute/start before working on tasks

When the user asks to "implement", "execute", "start working", "work on this", or similar:
- You should switch to EXECUTION phase and begin working through tasks
- Set planning_phase: "executing" in your response metadata"""
        else:  # executing
            phase_prompt = """
# CURRENT PHASE: EXECUTING
You are in the EXECUTION phase. Work through tasks autonomously:
1. Find the next pending task
2. Use START_TODO to mark it in progress
3. Complete the task work
4. Use COMPLETE_TODO to mark it done
5. Continue to the next task

Continue until all tasks are complete or you need user clarification."""

        base_prompt = f"{base_prompt}\n{phase_prompt}"

        # Add instruction to describe the plan after creation/modification
        if should_describe_plan:
            base_prompt += """

# IMPORTANT: You just created or modified the plan.
Now provide a clear, helpful response that:
1. Describes the tasks you created/modified
2. Lists the key tasks in the plan
3. Asks if the user wants to make any changes before starting execution
DO NOT make any tool calls - just respond with text describing the plan."""

        # Append todo context if provided
        if todos:
            task_context = self._format_todos_context(todos, current_task_index)
            return f"{base_prompt}\n\n{task_context}"

        return base_prompt

    def _format_todos_context(
        self,
        todos: List[Dict[str, Any]],
        current_task_index: Optional[int] = None,
    ) -> str:
        if not todos:
            return ""

        lines = ["CURRENT TASK PLAN:"]
        lines.append(
            "(Use the ID shown in brackets when calling COMPLETE_TODO or START_TODO)"
        )
        lines.append("")

        for i, todo in enumerate(todos):
            status = todo.get("status", "pending")
            desc = todo.get("description", "No description")
            todo_id = todo.get("id", str(i + 1))

            # Status indicator
            if status == "completed":
                indicator = "✅"
            elif status == "in_progress":
                indicator = "🔄"
            elif status == "skipped":
                indicator = "⏭️"
            else:
                indicator = "⬜"

            # Highlight current task
            current_marker = " ← CURRENT TASK" if i == current_task_index else ""
            lines.append(
                f"{indicator} Task {i + 1} [ID: {todo_id}]: {desc} ({status}){current_marker}"
            )

        # Add summary
        completed = sum(1 for t in todos if t.get("status") == "completed")
        total = len(todos)
        lines.append(f"\nProgress: {completed}/{total} tasks completed")

        return "\n".join(lines)

    async def invoke_model(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
        todos: Optional[List[Dict[str, Any]]] = None,
        current_task_index: Optional[int] = None,
    ) -> AgentResponse:
        if not self.langchain_model:
            return self._build_error_response(
                "Planning service is not properly configured.",
                conversation_id,
            )

        # Initialize MCP tools if not done yet
        if self.mcp_manager is None:
            await self._init_tools()

        llm_with_tools = self._get_llm_with_tools()
        system_prompt = self._build_system_prompt(
            None, False, todos, current_task_index
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=message.content),
        ]

        response = await llm_with_tools.ainvoke(messages)

        tool_calls = []
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_calls = response.tool_calls
            return AgentResponse(
                agent_type=AgentType.PLANNING,
                agent_id="planning_agent",
                message=AgentMessage(
                    role=MessageRole.ASSISTANT,
                    content=coerce_response_text(response.content or ""),
                    tool_calls=tool_calls,
                ),
                metadata={
                    "model": self.model_name,
                    "conversation_id": conversation_id,
                    "has_tool_calls": True,
                },
            )

        # No tool calls - just a regular response
        return AgentResponse(
            agent_type=AgentType.PLANNING,
            agent_id="planning_agent",
            message=AgentMessage(
                role=MessageRole.ASSISTANT,
                content=coerce_response_text(response.content or ""),
            ),
            metadata={
                "model": self.model_name,
                "conversation_id": conversation_id,
            },
        )

    # === Legacy methods for backward compatibility ===

    async def generate_plan(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
    ) -> AgentResponse:
        """Legacy method: Generate plan using structured output."""
        if not self.langchain_model:
            return self._build_error_response(
                "Planning service is not properly configured.",
                conversation_id,
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

            # Also generate todos for the new format
            todos = self._plan_to_todos(plan)

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
                    "todos": todos,  # Include for state sync
                },
            )

        except Exception as e:
            logger.error(f"Error generating plan: {e}", exc_info=True)
            error_msg = (
                f"Issue while creating the plan: {str(e)}. Please try rephrasing your request."
                if isinstance(e, ValueError)
                else f"Error while generating the plan: {str(e)}"
            )
            return self._build_error_response(
                error_msg,
                conversation_id,
                error=str(e),
            )

    async def modify_plan(
        self,
        message: AgentMessage,
        existing_tasks: List[Dict[str, Any]],
        conversation_id: Optional[str] = None,
    ) -> AgentResponse:
        """Legacy method: Modify plan using structured output."""
        if not self.langchain_model:
            return self._build_error_response(
                "Planning service is not properly configured.",
                conversation_id,
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

            # Generate todos for state sync
            todos = self._plan_to_todos(modified_plan)

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
                    "todos": todos,
                },
            )

        except ValueError as e:
            logger.error(f"Error modifying plan: {e}", exc_info=True)
            error_msg = (
                f"Issue while modifying the plan: {str(e)}. Please try rephrasing your request."
                if isinstance(e, ValueError)
                else f"Error while modifying the plan: {str(e)}"
            )
            return self._build_error_response(
                error_msg,
                conversation_id,
                error=str(e),
            )

    def _plan_to_todos(self, plan: Plan) -> List[Dict[str, Any]]:
        todos = []
        for i, task in enumerate(plan.tasks):
            todos.append(
                {
                    "id": str(uuid.uuid4()),
                    "description": task.description,
                    "status": TodoStatus.PENDING.value,
                    "order": i,
                    "dependencies": [str(d) for d in task.dependencies],
                    "complexity": task.estimated_complexity,
                }
            )
        return todos

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

    async def stream_message(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
        todos: Optional[List[Dict[str, Any]]] = None,
        current_task_index: Optional[int] = None,
    ):
        """Stream planning agent responses with proper token streaming."""
        if not self.langchain_model:
            yield {
                "type": "error",
                "error": "Planning service is not properly configured.",
            }
            return

        try:
            # Initialize MCP tools if not done yet
            if self.mcp_manager is None:
                await self._init_tools()

            llm_with_tools = self._get_llm_with_tools()
            system_prompt = self._build_system_prompt(
                None, False, todos, current_task_index
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=message.content),
            ]

            accumulated_content = ""
            accumulated_thinking = ""
            tool_calls = []

            # Stream using astream_events
            async for event in llm_with_tools.astream_events(messages, version="v2"):
                event_type = event.get("event")

                if event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if not chunk:
                        continue

                    # Handle content blocks
                    if hasattr(chunk, "content_blocks") and chunk.content_blocks:
                        for block in chunk.content_blocks:
                            block_type = block.get("type")

                            if block_type == "text":
                                token = block.get("text", "")
                                if token:
                                    accumulated_content += token
                                    yield {"type": "token", "content": token}

                            elif block_type in ("thinking", "reasoning"):
                                thinking_token = block.get(block_type, "") or block.get(
                                    "text", ""
                                )
                                if thinking_token:
                                    accumulated_thinking += thinking_token
                                    yield {
                                        "type": "thinking",
                                        "content": thinking_token,
                                    }

                    # Handle string content
                    elif hasattr(chunk, "content") and isinstance(chunk.content, str):
                        token = coerce_response_text(chunk.content)
                        if token:
                            accumulated_content += token
                            yield {"type": "token", "content": token}

                    # Handle tool calls
                    if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                        for tc_chunk in chunk.tool_call_chunks:
                            if tc_chunk:
                                tool_calls.append(tc_chunk)

            # Build final response
            response = AgentResponse(
                agent_type=AgentType.PLANNING,
                agent_id="planning_agent",
                message=AgentMessage(
                    role=MessageRole.ASSISTANT,
                    content=accumulated_content,
                    tool_calls=tool_calls if tool_calls else None,
                ),
                metadata={
                    "model": self.model_name,
                    "conversation_id": conversation_id,
                    "has_tool_calls": bool(tool_calls),
                },
            )

            if accumulated_thinking:
                response.metadata["thinking"] = accumulated_thinking

            yield {"type": "complete", "response": response}

        except Exception as e:
            logger.error(f"Error streaming planning response: {e}", exc_info=True)
            yield {"type": "error", "error": str(e)}

    async def cleanup(self):
        await super().cleanup()
