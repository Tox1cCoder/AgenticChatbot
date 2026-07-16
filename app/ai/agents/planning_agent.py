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
from ..planning_rubric import (
    FALLBACK_PLANNING_RUBRIC,
    PlanningRubricAttempt,
    PlanningRubricContract,
    PlanningRubricEvaluation,
    build_planning_rubric_author_prompt,
    build_planning_rubric_feedback,
    build_planning_rubric_grader_prompt,
    build_planning_rubric_revision_prompt,
    parse_planning_rubric_contract,
    parse_planning_rubric_evaluation,
)
from ..planning_tools import create_write_todos_tool
from ..prompts import PLANNING_EXECUTION_PROMPT
from ..request_budget import ContextBudgetExceededError
from ..schemas import (
    AgentMessage,
    AgentResponse,
    AgentType,
    MessageRole,
    TodoStatus,
)
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
        handoff_enabled = bool(kwargs.get("include_hand_off"))
        handoff_planning_guidance = (
            dedent(
                """
                - `hand_off` — transfer the *entire* conversation to a more
                  suitable top-level agent (e.g. the user asks a follow-up
                  that's outside planning, like "actually just answer this
                  question normally"). After hand_off the next agent owns the
                  reply; the planning loop ends for this turn.
                """
            ).strip()
            if handoff_enabled
            else ""
        )
        handoff_execution_guidance = (
            dedent(
                """
                ## `hand_off` rules
                - Before working the plan, use `hand_off` if the user has
                  clearly switched topic away from it (normal chat, document
                  search, and similar requests). The receiving agent takes
                  over the user-facing reply; the planning loop will not run
                  again this turn.
                - Do not use it to do parallel research — that is
                  `dispatch_subagents`.
                """
            ).strip()
            if handoff_enabled
            else ""
        )

        phase = planning_phase or "planning"
        if phase == "planning":
            phase_prompt = (
                dedent(
                    """
                # CURRENT PHASE: PLANNING
                - Create or modify the task plan using: set_todos, add_todo,
                  update_todo, remove_todo
                - Do NOT execute tasks: no start_todo or complete_todo
                - Ask for confirmation before starting execution
                - When modifying an existing plan, preserve task IDs and
                  statuses for unchanged tasks

                When the user explicitly asks to start/execute/implement:
                - Begin execution by calling start_todo for the next task

                ## Delegation tool (planning phase)
                - `dispatch_subagents` — fan out *independent* worker tasks in
                  parallel and wait for results. When the user explicitly
                  asks to delegate, fan out, parallelize, run subagents, or
                  test the subagent feature, CALL `dispatch_subagents`
                  directly with concrete worker tasks. Do not narrate it.
                {handoff_planning_guidance}
                Otherwise, plan creation/editing uses `write_todos` only.
                """
                )
                .strip()
                .replace("{handoff_planning_guidance}", handoff_planning_guidance)
            )
        else:
            phase_prompt = (
                dedent(
                    """
                # CURRENT PHASE: EXECUTING

                ## Decision order on every turn
                1. If the user explicitly asked to delegate, dispatch, fan
                   out, parallelize, or run subagents — CALL
                   `dispatch_subagents` now with concrete worker tasks. Do not
                   narrate it.
                2. If the next pending todo can be split into 2+ INDEPENDENT
                   sub-tasks (research X while building Y; check A while
                   drafting B), prefer `dispatch_subagents` over doing them
                   yourself serially.
                3. Otherwise, work the next pending todo directly: start_todo
                   → do the work → complete_todo → continue.

                {handoff_execution_guidance}

                ## `dispatch_subagents` rules
                - Targets: chat_agent, rag_agent, search_agent,
                  image_generator_agent, canvas_agent. `planning_agent` is
                  forbidden.
                - Tasks in one call MUST be independent. Serial work stays
                  sequential.
                - Keep dispatch calls focused: include only the independent
                  worker tasks needed for the current step.
                - Workers CANNOT mutate todos. After the call returns, read
                  each `answer` (the full worker answer) and call
                  `write_todos` (complete or update) for related todos. If a
                  result is `failed`, `timeout`, or `requires_approval`,
                  leave the todo pending and explain the blocker in your
                  reply.
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

                ## Anti-narration rule
                Never describe a delegation in prose without emitting the
                tool call. Either dispatch, hand off, or do the work yourself
                via `write_todos`. Pure prose like "I'll hand this to the
                search agent" is a bug.

                Stop only when all tasks are completed (then summarize), you
                need user clarification, or an unresolvable error occurs.
                """
                )
                .strip()
                .replace("{handoff_execution_guidance}", handoff_execution_guidance)
            )

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

        planning_rubric_feedback = kwargs.get("planning_rubric_feedback")
        if isinstance(planning_rubric_feedback, str) and planning_rubric_feedback.strip():
            prompt += (
                "\n\n# PLANNING RUBRIC FEEDBACK\n"
                "Revise the plan by calling write_todos. Do not answer in prose until "
                "the rubric feedback is resolved.\n" + planning_rubric_feedback.strip()
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
        budget_result = None
        planning_history = langchain_messages[1:-1]
        planning_current = langchain_messages[-1:]

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
                budget_result = await self._preflight_model_request(
                    runtime_config,
                    system_messages=langchain_messages[:1],
                    history_messages=planning_history,
                    current_messages=planning_current,
                    tools=[write_todos_tool],
                    conversation_id=conversation_id,
                    user_id=request_user_id,
                )
                request_messages = (
                    list(budget_result.envelope.messages)
                    if budget_result is not None
                    else langchain_messages
                )
                raw_response = await self._ainvoke_with_retries(bound_llm, request_messages)
                break
            except ContextBudgetExceededError as exc:
                return self._build_error_response(
                    (
                        "This planning request is too large for the selected model's "
                        "context window. Please shorten the request."
                    ),
                    conversation_id,
                    error=exc.code,
                )
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

        # Plan-quality judgment is rubric/LLM-driven, not a fixed-length rejection.
        # The rubric loop enforces only structural invariants in code (valid shape,
        # unique ids, allowed statuses) and lets the grader revise vague tasks
        # before persistence. Disabling the rubric skips quality review entirely.
        rubric_attempt: PlanningRubricAttempt | None = None
        if getattr(settings, "planning_rubric_enabled", True):
            canonical_todos, rubric_attempt = await self._run_planning_rubric_loop(
                llm=llm,
                write_todos_tool=write_todos_tool,
                system_prompt=system_prompt,
                message=message,
                message_content=message_content,
                canonical_todos=canonical_todos,
                existing_todos=todos,
                plan_modified=plan_modified,
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
        request_budget_metadata = self._request_budget_metadata(budget_result)
        if request_budget_metadata is not None:
            metadata["request_budget"] = request_budget_metadata
        if plan_modified:
            metadata["plan_modified"] = True
        if rubric_attempt is not None:
            metadata["planning_rubric"] = rubric_attempt.metadata()

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

    # ---- Planning rubric grading -----------------------------------------

    async def _resolve_planning_rubric_contract(
        self,
        *,
        llm: Any,
        caller_rubric: str | None,
        user_message: str,
        candidate_todos: list[dict[str, Any]],
        existing_todos: list[dict[str, Any]] | None,
        plan_modified: bool,
        lifecycle: str | None = None,
        prior_feedback: str | None = None,
    ) -> PlanningRubricContract:
        if isinstance(caller_rubric, str) and caller_rubric.strip():
            return PlanningRubricContract(
                rubric=caller_rubric.strip(),
                source="caller",
                rationale="Caller supplied the planning rubric.",
            )

        prompt = build_planning_rubric_author_prompt(
            user_message=user_message,
            candidate_todos=candidate_todos,
            existing_todos=existing_todos,
            plan_modified=plan_modified,
            lifecycle=lifecycle,
            prior_feedback=prior_feedback,
        )
        try:
            raw_response = await self._ainvoke_with_retries(
                llm,
                [SystemMessage(content=prompt)],
            )
            raw_text = coerce_response_text(getattr(raw_response, "content", ""))
            return parse_planning_rubric_contract(raw_text)
        except Exception:
            return PlanningRubricContract(
                rubric=FALLBACK_PLANNING_RUBRIC,
                source="fallback",
                rationale="Rubric authoring failed; using minimal invariant fallback.",
            )

    async def _grade_planning_todos(
        self,
        *,
        llm: Any,
        rubric: str,
        user_message: str,
        candidate_todos: list[dict[str, Any]],
        existing_todos: list[dict[str, Any]] | None,
        plan_modified: bool,
        iteration: int,
    ) -> PlanningRubricEvaluation:
        prompt = build_planning_rubric_grader_prompt(
            rubric=rubric,
            user_message=user_message,
            candidate_todos=candidate_todos,
            existing_todos=existing_todos,
            plan_modified=plan_modified,
        )
        try:
            raw_response = await self._ainvoke_with_retries(
                llm,
                [SystemMessage(content=prompt)],
            )
            raw_text = coerce_response_text(getattr(raw_response, "content", ""))
            return parse_planning_rubric_evaluation(raw_text, iteration=iteration)
        except Exception as exc:
            return PlanningRubricEvaluation(
                iteration=iteration,
                result="grader_error",
                explanation=f"Planning rubric grader failed: {exc}",
                criteria=[],
            )

    async def _revise_todos_from_rubric_feedback(
        self,
        *,
        llm: Any,
        write_todos_tool: BaseTool,
        base_system_prompt: str,
        user_message: str,
        base_todos: list[dict[str, Any]],
        candidate_todos: list[dict[str, Any]],
        feedback: str,
    ) -> list[dict[str, Any]]:
        revision_prompt = build_planning_rubric_revision_prompt(
            feedback=feedback,
            candidate_todos=candidate_todos,
        )
        bound_llm = ModelFactory.bind_tools_to_model(
            llm,
            [write_todos_tool],
            tool_choice="write_todos",
        )
        raw_response = await self._ainvoke_with_retries(
            bound_llm,
            [
                SystemMessage(content=base_system_prompt),
                HumanMessage(content=user_message),
                HumanMessage(content=revision_prompt),
            ],
        )
        return self._canonicalize_todos(
            self._apply_write_todos_calls(
                base_todos=base_todos,
                tool_calls=getattr(raw_response, "tool_calls", None) or [],
            )
        )

    def _build_planning_structural_feedback(
        self,
        todos: list[dict[str, Any]],
    ) -> str | None:
        if not todos:
            return "The plan must contain at least one todo."
        seen_ids: set[str] = set()
        for index, todo in enumerate(todos, start=1):
            todo_id = str(todo.get("id") or "").strip()
            if not todo_id:
                return f"Task {index} is missing a stable id."
            if todo_id in seen_ids:
                return f"Task {index} reuses duplicate id {todo_id}."
            seen_ids.add(todo_id)
            status = str(todo.get("status") or "").strip().lower()
            if status not in {
                TodoStatus.PENDING.value,
                TodoStatus.IN_PROGRESS.value,
                TodoStatus.COMPLETED.value,
                TodoStatus.SKIPPED.value,
            }:
                return f"Task {index} has unsupported status {status!r}."
        return None

    async def _run_planning_rubric_loop(
        self,
        *,
        llm: Any,
        write_todos_tool: BaseTool,
        system_prompt: str,
        message: AgentMessage,
        message_content: str,
        canonical_todos: list[dict[str, Any]],
        existing_todos: list[dict[str, Any]] | None,
        plan_modified: bool,
    ) -> tuple[list[dict[str, Any]], PlanningRubricAttempt]:
        """Grade candidate todos against a contextual rubric, revising on demand.

        Returns the (possibly revised) canonical todos and the terminal attempt.
        Structural guardrails stay in code; plan-quality judgment stays LLM-driven.
        """
        caller_rubric = message.metadata.get("planning_rubric")
        caller_rubric = caller_rubric if isinstance(caller_rubric, str) else None
        max_iterations = max(1, int(getattr(settings, "planning_rubric_max_iterations", 3)))
        evaluations: list[PlanningRubricEvaluation] = []
        feedback: str | None = None
        status = "satisfied"

        contract = await self._resolve_planning_rubric_contract(
            llm=llm,
            caller_rubric=caller_rubric,
            user_message=message_content,
            candidate_todos=canonical_todos,
            existing_todos=existing_todos,
            plan_modified=plan_modified,
            lifecycle="planning",
        )

        for iteration in range(max_iterations):
            structural_feedback = self._build_planning_structural_feedback(canonical_todos)
            if structural_feedback:
                evaluation = PlanningRubricEvaluation(
                    iteration=iteration,
                    result="needs_revision",
                    explanation="Structural planning guardrail requires revision.",
                    criteria=[
                        {
                            "name": "structural_validity",
                            "passed": False,
                            "gap": structural_feedback,
                        }
                    ],
                )
            else:
                evaluation = await self._grade_planning_todos(
                    llm=llm,
                    rubric=contract.rubric,
                    user_message=message_content,
                    candidate_todos=canonical_todos,
                    existing_todos=existing_todos,
                    plan_modified=plan_modified,
                    iteration=iteration,
                )
            evaluations.append(evaluation)

            if evaluation.result == "satisfied":
                status = "satisfied"
                break
            if evaluation.result in {"failed", "grader_error"}:
                status = evaluation.result
                feedback = evaluation.explanation
                break

            feedback = build_planning_rubric_feedback(evaluation)
            if iteration >= max_iterations - 1:
                status = "max_iterations_reached"
                break

            canonical_todos = await self._revise_todos_from_rubric_feedback(
                llm=llm,
                write_todos_tool=write_todos_tool,
                base_system_prompt=system_prompt,
                user_message=message_content,
                base_todos=canonical_todos,
                candidate_todos=canonical_todos,
                feedback=feedback,
            )

        rubric_attempt = PlanningRubricAttempt(
            status=status,
            iterations=len(evaluations),
            source="modify_plan" if plan_modified else "generate_plan",
            rubric=contract.rubric,
            rubric_source=contract.source,
            rubric_rationale=contract.rationale,
            evaluations=evaluations,
            feedback=feedback,
            error=feedback if status == "grader_error" else None,
        )
        return canonical_todos, rubric_attempt

    async def review_todos_with_planning_rubric(
        self,
        *,
        user_message: str,
        candidate_todos: list[dict[str, Any]],
        existing_todos: list[dict[str, Any]] | None,
        plan_modified: bool,
        source: str = "planning_tools",
        start_iteration: int = 0,
        user_id: str | None = None,
        model_request: dict[str, Any] | None = None,
    ) -> PlanningRubricAttempt:
        """Grade a single graph-level candidate plan (one pass per call).

        The graph drives revision by routing back to the planning agent, so this
        method grades once and reports ``needs_revision``/``satisfied`` with the
        running ``start_iteration`` counter rather than looping internally.
        """
        runtime_config = self._resolve_runtime_model_config(user_id, model_request)
        llm, _ = self._create_langchain_model_from_runtime(
            runtime_config,
            user_id=user_id,
            enable_reasoning_summary=False,
        )
        max_iterations = max(1, int(getattr(settings, "planning_rubric_max_iterations", 3)))
        evaluations: list[PlanningRubricEvaluation] = []
        feedback: str | None = None
        status = "satisfied"

        iteration = max(0, int(start_iteration or 0))
        if iteration >= max_iterations:
            return PlanningRubricAttempt(
                status="max_iterations_reached",
                iterations=iteration,
                source=source,
                rubric=FALLBACK_PLANNING_RUBRIC,
                rubric_source="fallback",
                rubric_rationale="Graph-level rubric pass cap was already reached.",
                evaluations=[],
                feedback=None,
            )

        contract = await self._resolve_planning_rubric_contract(
            llm=llm,
            caller_rubric=None,
            user_message=user_message,
            candidate_todos=candidate_todos,
            existing_todos=existing_todos,
            plan_modified=plan_modified,
            lifecycle="planning_tools",
        )
        evaluation = await self._grade_planning_todos(
            llm=llm,
            rubric=contract.rubric,
            user_message=user_message,
            candidate_todos=candidate_todos,
            existing_todos=existing_todos,
            plan_modified=plan_modified,
            iteration=iteration,
        )
        evaluations.append(evaluation)
        if evaluation.result == "satisfied":
            status = "satisfied"
        elif evaluation.result in {"failed", "grader_error"}:
            status = evaluation.result
            feedback = evaluation.explanation
        else:
            feedback = build_planning_rubric_feedback(evaluation)
            if iteration >= max_iterations - 1:
                status = "max_iterations_reached"
            else:
                status = "needs_revision"

        return PlanningRubricAttempt(
            status=status,
            iterations=iteration + 1,
            source=source,
            rubric=contract.rubric,
            rubric_source=contract.source,
            rubric_rationale=contract.rationale,
            evaluations=evaluations,
            feedback=feedback,
            error=feedback if status == "grader_error" else None,
        )

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
