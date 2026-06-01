from __future__ import annotations

import uuid
from collections.abc import Iterable
from textwrap import dedent
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from ...core.config import settings
from ...interfaces.runtime_model_resolver_interface import IRuntimeModelResolver
from ..model_factory import ModelFactory
from ..planning_tools import create_write_todos_tool
from ..prompts import PLANNING_EXECUTION_PROMPT
from ..schemas import (
    AgentMessage,
    AgentResponse,
    AgentType,
    MessageRole,
    TodoStatus,
)
from ..text_normalization import tokenize_text
from ..todo_actions import apply_write_todos_action
from ..utils import coerce_response_text, normalize_tool_call
from .base_agent import BaseAgent


class PlanningAgent(BaseAgent):
    """Planning agent that manages task plans using ReAct-style tool-calling."""

    def __init__(
        self,
        model_name: str | None = None,
        runtime_model_resolver: IRuntimeModelResolver | None = None,
    ):
        """Initialize Planning Agent with 'planning' config key."""
        super().__init__(
            model_name=model_name,
            agent_config_key="planning",
            runtime_model_resolver=runtime_model_resolver,
        )

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
        conversation_id: str | None = None,
        internal_tools: list[BaseTool] | None = None,
        user_id: str | None = None,
        device_id: str | None = None,
        include_hand_off: bool | None = None,
    ) -> Any:
        """
        Override to ensure write_todos is always included as an internal tool.

        This guarantees write_todos is available even in deferred tool loading mode,
        where only tool_search + pinned tools + loaded tools would normally be bound.
        """
        return super()._get_llm_with_tools(
            model=model,
            conversation_id=conversation_id,
            internal_tools=self._combine_with_write_todos(internal_tools),
            user_id=user_id,
            device_id=device_id,
            include_hand_off=include_hand_off,
        )

    def _get_tools_for_binding(
        self,
        conversation_id: str | None = None,
        internal_tools: list[BaseTool] | None = None,
        user_id: str | None = None,
        device_id: str | None = None,
        tool_scope: str | None = None,
        include_hand_off: bool | None = None,
    ) -> list[BaseTool]:
        """
        Ensure write_todos is always present in both binding and execution maps.
        """
        return super()._get_tools_for_binding(
            conversation_id=conversation_id,
            internal_tools=self._combine_with_write_todos(internal_tools),
            user_id=user_id,
            device_id=device_id,
            tool_scope=tool_scope,
            include_hand_off=include_hand_off,
        )

    @staticmethod
    def _combine_with_write_todos(
        internal_tools: list[BaseTool] | None,
    ) -> list[BaseTool]:
        """Merge optional internal tools with the mandatory write_todos tool."""
        write_todos_tool = create_write_todos_tool()
        combined: list[BaseTool] = [write_todos_tool]
        if not internal_tools:
            return combined
        for tool in internal_tools:
            if tool.name != write_todos_tool.name:
                combined.append(tool)
        return combined

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
        persona: str | None = None,
        has_tool_context: bool = False,
        todos: list[dict[str, Any]] | None = None,
        current_task_index: int | None = None,
        planning_phase: str | None = None,
        should_describe_plan: bool = False,
        **kwargs: Any,
    ) -> str:
        base_prompt = super()._build_system_prompt(persona, has_tool_context, **kwargs)

        phase = planning_phase or "planning"
        if phase == "planning":
            phase_prompt = dedent(
                """
                # CURRENT PHASE: PLANNING
                - Create or modify the task plan using: set_todos, add_todo, update_todo, remove_todo
                - Do NOT execute tasks: no start_todo or complete_todo
                - Ask for confirmation before starting execution
                - When modifying an existing plan, preserve task IDs and statuses for unchanged tasks

                When the user explicitly asks to start/execute/implement:
                - Begin execution by calling start_todo for the next task

                ## Delegation tools (planning phase)
                Two delegation primitives exist; do NOT confuse them.
                - `dispatch_subagents` — fan out *independent* worker tasks in
                  parallel and wait for results. When the user explicitly
                  asks to delegate, fan out, parallelize, run subagents, or
                  test the subagent feature, CALL `dispatch_subagents`
                  directly with concrete worker tasks. Do not narrate it.
                - `hand_off` — transfer the *entire* conversation to a more
                  suitable top-level agent (e.g. the user asks a follow-up
                  that's outside planning, like "actually just answer this
                  question normally"). After hand_off the next agent owns the
                  reply; the planning loop ends for this turn.
                Otherwise, plan creation/editing uses `write_todos` only.
                """
            ).strip()
        else:
            phase_prompt = dedent(
                """
                # CURRENT PHASE: EXECUTING

                ## Decision order on every turn
                1. If the user has clearly switched topic away from this plan
                   (asks an off-plan question, wants normal chat, wants
                   document search, etc.) — CALL `hand_off` to the right
                   top-level agent with a one-line reason. Do not keep working
                   the plan against the user's intent.
                2. If the user explicitly asked to delegate, dispatch, fan
                   out, parallelize, or run subagents — CALL
                   `dispatch_subagents` now with concrete worker tasks. Do not
                   narrate it.
                3. If the next pending todo can be split into 2+ INDEPENDENT
                   sub-tasks (research X while building Y; check A while
                   drafting B), prefer `dispatch_subagents` over doing them
                   yourself serially.
                4. Otherwise, work the next pending todo directly: start_todo
                   → do the work → complete_todo → continue.

                ## `dispatch_subagents` rules
                - Targets: chat_agent, rag_agent, search_agent,
                  image_generator_agent, canvas_agent. `planning_agent` is
                  forbidden.
                - Tasks in one call MUST be independent. Serial work stays
                  sequential.
                - Keep dispatch calls focused: include only the independent
                  worker tasks needed for the current step.
                - Workers CANNOT mutate todos. After the call returns, read
                  each `answer` (the full worker answer; `summary` is only
                  the compact activity preview) and call `write_todos`
                  (complete or update) for related todos. If a result is
                  `failed`, `timeout`, or `requires_approval`, leave the todo
                  pending and explain the blocker in your reply.
                - You are the only actor allowed to call `write_todos`.

                ### Task fields — required shapes
                - `task`: a string. Put the worker instructions here.
                - `context`: optional, MUST be a JSON object (dict). For
                  arbitrary text like a crawled page or a document excerpt,
                  wrap it: `{"text": "...long blob..."}`. For a list of
                  references/citations: `{"items": [...]}`. Never pass a raw
                  string or array as `context` — the call will be rejected.
                - `related_todo_ids`: optional list of todo id strings.

                ## Subagent model choice (optional `model_override`)
                - If the user names a model, pass it in `model_override`.
                - Otherwise use the worker's default model for normal tasks.
                - Use faster/lower-cost models for simple extraction,
                  formatting, search summaries, and high-volume parallel
                  checks.
                - Use frontier/high-reasoning models only for hard coding,
                  architecture, debugging, ambiguous synthesis, or tasks
                  where a cheap retry would cost more time than one strong
                  call.

                ### Allowed `model_override.model` values (use these EXACT ids)
                OpenAI (`provider: "openai"`):
                - `gpt-5.5` — frontier coding/professional reasoning.
                - `gpt-5.4` — frontier, lower cost than 5.5.
                - `gpt-5.4-mini` — fast mini model for subagents / high-volume
                  parallel checks.

                Gemini (`provider: "gemini"`):
                - `gemini-3.1-pro-preview` — complex agentic / vibe-coding
                  (Pro supports `low`/`high` reasoning_effort only).
                - `gemini-3-flash-preview` — lower-cost frontier; supports
                  `minimal`/`low`/`medium`/`high` reasoning_effort.

                ### `reasoning_effort` is a SEPARATE field
                Allowed values: `none`, `minimal`, `low`, `medium`, `high`,
                `xhigh`. Pass it as its own key — `reasoning_effort` is a
                separate field. NEVER append it to the `model` id.

                Correct:
                ```json
                {"provider": "openai", "model": "gpt-5.5",
                 "reasoning_effort": "medium"}
                ```
                Wrong (these are invalid model ids and the worker will
                fail with `model_not_found`):
                - `"model": "gpt-5.5-medium"`
                - `"model": "gemini-3-flash"` (missing `-preview`)
                - `"model": "gpt-5"` or `"model": "gemini"` (not a real id)

                ## `hand_off` rules
                - Use ONLY when the conversation should leave planning
                  entirely. Do not use to do parallel research — that is
                  `dispatch_subagents`.
                - The receiving agent takes over the user-facing reply; the
                  planning loop will not run again this turn.

                ## Anti-narration rule
                Never describe a delegation in prose without emitting the
                tool call. Either dispatch, hand off, or do the work yourself
                via `write_todos`. Pure prose like "I'll hand this to the
                search agent" is a bug.

                Stop only when all tasks are completed (then summarize), you
                need user clarification, or an unresolvable error occurs.
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

        custom_workers = kwargs.get("custom_workers")
        if custom_workers:
            worker_lines = []
            for worker in custom_workers:
                runtime_id = worker.get("runtime_agent_id")
                name = worker.get("name") or runtime_id
                description = (worker.get("description") or "").strip()
                worker_lines.append(f'- {runtime_id} (name: "{name}"): {description}')
            prompt += (
                "\n\n## Custom worker agents available to dispatch_subagents\n"
                "In addition to base workers, you may dispatch to these attached "
                "custom agents by using their runtime id as the task `agent`:\n"
                + "\n".join(worker_lines)
            )

        return prompt

    def _format_todos_context(
        self, todos: list[dict[str, Any]], current_task_index: int | None = None
    ) -> str:
        if not todos:
            return ""

        lines = ["CURRENT TASK PLAN:"]
        lines.append("(Use the ID shown in brackets when calling start_todo/complete_todo)")
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
            if (getattr(t.get("status"), "value", t.get("status")) == TodoStatus.COMPLETED.value)
        )
        lines.append(f"\nProgress: {completed}/{len(todos)} tasks completed")

        return "\n".join(lines)

    async def generate_plan(
        self, message: AgentMessage, conversation_id: str | None = None
    ) -> AgentResponse:
        """Generate a canonical todo payload for persistence."""
        return await self._generate_or_modify_plan(
            message=message,
            existing_tasks=None,
            conversation_id=conversation_id,
            plan_modified=False,
        )

    async def modify_plan(
        self,
        message: AgentMessage,
        existing_tasks: list[dict[str, Any]],
        conversation_id: str | None = None,
    ) -> AgentResponse:
        """Modify an existing canonical todo payload for persistence."""
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
        existing_tasks: list[dict[str, Any]] | None,
        conversation_id: str | None,
        plan_modified: bool,
    ) -> AgentResponse:
        persona = message.metadata.get("persona")
        conversation_history = message.metadata.get("history", [])
        request_user_id = message.metadata.get("user_id")
        model_request = message.metadata.get("model_request")
        history_summary = message.metadata.get("history_summary")

        # Convert existing tasks into todo-like dicts for prompt context.
        todos: list[dict[str, Any]] = []
        current_task_index: int | None = None
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
            history_summary=history_summary,
        )

        # Force a tool call so we can reliably extract a machine-readable plan.
        write_todos_tool = create_write_todos_tool()

        langchain_messages = [SystemMessage(content=system_prompt)]
        if conversation_history:
            langchain_messages.extend(
                self._convert_history_to_langchain_messages(conversation_history)
            )
        message_content = message.content or ""
        langchain_messages.append(HumanMessage(content=message_content))

        runtime_config = self._resolve_runtime_model_config(request_user_id, model_request)

        while True:
            try:
                llm, _ = self._create_langchain_model_from_runtime(
                    runtime_config,
                    user_id=request_user_id,
                    enable_reasoning_summary=False,
                )
                bound_llm = ModelFactory.bind_tools_to_model(
                    llm,
                    [write_todos_tool],
                    tool_choice="write_todos",
                )
                raw_response = await self._ainvoke_with_retries(bound_llm, langchain_messages)
                break
            except Exception:
                fallback_runtime = self._create_fallback_runtime_config(
                    runtime_config.fallback_config,
                    reason="provider_error",
                    from_provider=runtime_config.provider,
                    inherited_warnings=runtime_config.warnings,
                )
                if not fallback_runtime or fallback_runtime.provider == runtime_config.provider:
                    return self._build_error_response(
                        "Planning service is not properly configured.",
                        conversation_id,
                    )
                runtime_config = fallback_runtime

        tool_calls = getattr(raw_response, "tool_calls", None) or []
        updated_todos = self._apply_write_todos_calls(base_todos=todos, tool_calls=tool_calls)

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

        canonical_todos = self._canonicalize_todos(updated_todos)

        # Quality gate: reject plans that contain under-specified task descriptions.
        quality_issues = self._validate_task_descriptions(canonical_todos)
        if quality_issues:
            problem_lines = "; ".join(f"task {i + 1}: {reason}" for i, reason in quality_issues)
            return self._build_error_response(
                f"Generated plan contains under-specified tasks ({problem_lines}). "
                "Please provide a more detailed request so each task can be "
                "described with a concrete action and sufficient detail.",
                conversation_id,
            )

        overall_goal = (message_content.strip() or None) if not plan_modified else None

        response_text = self._format_plan_summary(
            canonical_todos,
            overall_goal=overall_goal,
            plan_modified=plan_modified,
        )

        metadata: dict[str, Any] = {
            "conversation_id": conversation_id,
            "task_count": len(canonical_todos),
            "todos": canonical_todos,
        }
        self._apply_runtime_metadata(metadata, runtime_config)
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
        self, *, base_todos: list[dict[str, Any]], tool_calls: Iterable[Any]
    ) -> list[dict[str, Any]]:
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

    def _canonicalize_todos(self, todos: list[dict[str, Any]]) -> list[dict[str, Any]]:
        def sort_key(item: dict[str, Any]) -> int:
            order = item.get("order")
            try:
                return int(order)
            except (TypeError, ValueError):
                return 0

        ordered = sorted(todos, key=sort_key) if todos else []

        seen_ids: set[str] = set()
        active_task_seen = False
        canonical: list[dict[str, Any]] = []
        for item in ordered:
            desc = str(item.get("description", "")).strip()
            if not desc:
                continue
            todo_id = str(item.get("id") or uuid.uuid4())
            if todo_id in seen_ids:
                continue
            seen_ids.add(todo_id)

            raw_status = item.get("status", TodoStatus.PENDING.value)
            status = raw_status.value if hasattr(raw_status, "value") else str(raw_status)
            if status not in {
                TodoStatus.PENDING.value,
                TodoStatus.IN_PROGRESS.value,
                TodoStatus.COMPLETED.value,
                TodoStatus.SKIPPED.value,
            }:
                status = TodoStatus.PENDING.value
            if status == TodoStatus.IN_PROGRESS.value:
                if active_task_seen:
                    status = TodoStatus.PENDING.value
                else:
                    active_task_seen = True

            canonical.append(
                {
                    "id": todo_id,
                    "description": desc,
                    "status": status,
                    "order": len(canonical),
                }
            )

        return canonical

    # ---- Task-description quality ----------------------------------------

    # Minimum character length a task description must meet.
    _MIN_DESC_LEN: int = 20

    def _validate_task_descriptions(self, todos: list[dict[str, Any]]) -> list[tuple[int, str]]:
        """Return (0-based index, reason) for every invalid task description.

        A description is considered invalid if it is shorter than
        ``_MIN_DESC_LEN`` characters or does not contain enough substance
        to stand alone as an actionable task.
        """
        issues: list[tuple[int, str]] = []
        for i, todo in enumerate(todos):
            desc = str(todo.get("description", "")).strip()
            if len(desc) < self._MIN_DESC_LEN:
                issues.append((i, f"too short ({len(desc)} chars, min {self._MIN_DESC_LEN})"))
                continue

            if len(tokenize_text(desc)) < 3:
                issues.append((i, "too little detail for an actionable task"))
        return issues

    def _format_plan_summary(
        self,
        todos: list[dict[str, Any]],
        *,
        overall_goal: str | None,
        plan_modified: bool,
    ) -> str:
        if not todos:
            return "I couldn't generate any actionable tasks."

        parts = [
            "I've updated the plan:" if plan_modified else "I've created a plan:",
            "",
        ]
        if overall_goal:
            parts.append(f"Goal: {overall_goal}")
            parts.append("")
        for i, todo in enumerate(todos, start=1):
            parts.append(f"{i}. {todo['description']}")
        return "\n".join(parts)
