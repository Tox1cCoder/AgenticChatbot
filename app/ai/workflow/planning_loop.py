"""Planning loop helpers extracted from MultiAgentWorkflow (Task 8, Step 6).

Methods are relocated verbatim; behavior is identical.
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import interrupt

from app.ai.schemas import GraphState, GraphStateView, TodoStatus
from app.ai.todo_actions import apply_write_todos_action
from app.ai.tool_execution import build_rejected_tool_artifacts, ensure_agent_tool_map
from app.ai.utils import apply_hitl_decisions, normalize_tool_call
from app.core.config import settings
from app.services.event_streaming.subagents import resolve_subagent_event_sink

logger = logging.getLogger("app.ai.graph")
_apply_decisions = apply_hitl_decisions


class PlanningLoopMixin:
    """Relocated planning_loop methods for :class:`MultiAgentWorkflow`."""

    def _build_planning_internal_tools(
        self,
        state: GraphState,
        *,
        executable: bool = False,
    ) -> list[Any]:
        """Return Planning-supervisor-only internal tools for this turn.

        ``dispatch_subagents`` is bound whenever the feature flag is on. The
        planning node only runs when a turn is routed (or handed off) to
        planning_agent, so an explicit "use subagents" request works without
        pre-enabling Planning mode — the mode flag flips on automatically when
        the resulting todos sync. ``planning_phase``, Planning mode, and plan
        presence are prompt-level guidance, not binding gates.
        """
        if not getattr(settings, "planning_subagents_enabled", False):
            return []

        from app.ai.planning_subagents import (
            PlanningSubagentDispatcher,
            create_dispatch_subagents_tool,
        )

        context = state.get("context") if isinstance(state.get("context"), dict) else {}
        event_sink = resolve_subagent_event_sink(context.get("subagent_event_sink_token"))
        dispatcher = (
            PlanningSubagentDispatcher(
                workflow=self,
                settings=settings,
                event_sink=event_sink,
            )
            if executable
            else None
        )
        dispatch_tool = create_dispatch_subagents_tool(
            dispatcher=dispatcher,
            parent_state_provider=(lambda: state) if executable else None,
        )
        return [dispatch_tool]

    async def _planning_node(self, state: GraphState) -> GraphState:
        """
        Planning agent node with tool-calling ReAct pattern.

        Uses invoke_model_with_history to get responses that may include tool calls
        for the write_todos tool. Supports both initial plan creation and
        ongoing task management.
        """
        messages = state.get("messages", [])
        if not messages:
            return state

        conversation_id = state.get("conversation_id")
        user_id = state.get("user_id")
        conversation_history = await self._get_conversation_history(
            conversation_id, user_id, agent_key="planning", state=state
        )

        context = state.get("context", {})

        # Get planning phase and check if we need to generate plan response
        planning_phase = state.get("planning_phase", "planning")
        should_generate_plan_response = context.get("generate_plan_response", False)

        # Clear the flag after reading
        if should_generate_plan_response:
            context["generate_plan_response"] = False
            state["context"] = context

        # Get current todos from state (may have been updated by planning_tools)
        todos = state.get("todos", [])
        current_task_index = state.get("current_task_index")

        # Increment planning call count for budget tracking
        planning_call_count = (state.get("planning_call_count") or 0) + 1
        state["planning_call_count"] = planning_call_count

        # Debug logging for observability
        logger.debug(
            f"[Planning Node] phase={planning_phase}, call_count={planning_call_count}, "
            f"generate_plan_response={should_generate_plan_response}, "
            f"todos_count={len(todos)}, current_task_index={current_task_index}"
        )

        persona = state.get("persona")

        # Get only current turn messages for the model
        current_turn_messages = self._messages_for_selected_agent(
            state,
            state.get("selected_agent") or "planning_agent",
            messages,
        )
        current_turn_messages, has_images = self._apply_current_turn_attachments(
            state,
            current_turn_messages,
        )

        # Build the Planning-mode subagent dispatch tool (only when allowed).
        internal_tools = self._build_planning_internal_tools(state)
        multi_agent_kwargs = self._multi_agent_kwargs(state, "planning_agent")
        internal_tools.extend(multi_agent_kwargs.get("internal_tools") or [])

        # Call the planning agent with history
        custom_workers = [
            {
                "runtime_agent_id": entry.get("runtime_agent_id") or runtime_id,
                "name": entry.get("name"),
                "description": entry.get("description"),
            }
            for runtime_id, entry in GraphStateView(state).custom_agents().items()
        ]

        response = await self.planning_agent.invoke_model_with_history(
            messages=current_turn_messages,
            conversation_history=conversation_history,
            persona=persona,
            conversation_id=conversation_id,
            user_id=user_id,
            device_id=state.get("device_id"),
            model_request=state.get("model_request"),
            history_summary=state.get("history_summary"),
            todos=todos,
            current_task_index=current_task_index,
            planning_phase=planning_phase,
            should_describe_plan=should_generate_plan_response,
            internal_tools=internal_tools or None,
            custom_workers=custom_workers or None,
            planning_rubric_feedback=context.get("planning_rubric_feedback"),
            handoff_target_descriptions=multi_agent_kwargs.get("handoff_target_descriptions"),
            multi_agent_activity=multi_agent_kwargs.get("multi_agent_activity"),
            **self._final_response_kwargs(state),
        )

        # Check if agent switched to executing phase via response metadata
        if response.metadata.get("planning_phase"):
            state["planning_phase"] = response.metadata["planning_phase"]

        if response.metadata.get("todos"):
            state["todos"] = response.metadata["todos"]

        # Mark that final summary was generated if this was a summary call
        if context.get("generate_final_summary") and not response.message.tool_calls:
            context["final_summary_generated"] = True
            state["context"] = context

        self._mark_response_has_images(response, has_images)

        response = self._finalize_forced_final_response(state, response)
        self._merge_tool_artifacts(state, response)
        return self._finalize_agent_response(state, response)

    async def _review_planning_todos_with_rubric(
        self,
        *,
        state: GraphState,
        todos: list[dict[str, Any]],
        source: str = "planning_tools",
    ) -> Any:
        if not getattr(settings, "planning_rubric_enabled", True):
            from app.ai.planning_rubric import FALLBACK_PLANNING_RUBRIC, PlanningRubricAttempt

            return PlanningRubricAttempt(
                status="disabled",
                iterations=0,
                source=source,
                rubric=FALLBACK_PLANNING_RUBRIC,
                rubric_source="fallback",
                evaluations=[],
            )

        # Reuse PlanningAgent's grader method so provider/model behavior stays centralized.
        context = GraphStateView(state).context_copy()
        previous = context.get("planning_rubric")
        previous_iterations = 0
        if (
            isinstance(previous, dict)
            and previous.get("source") == source
            and previous.get("status") == "needs_revision"
        ):
            try:
                previous_iterations = int(previous.get("iterations") or 0)
            except (TypeError, ValueError):
                previous_iterations = 0
        try:
            return await self.planning_agent.review_todos_with_planning_rubric(
                user_message=self._latest_user_text(state),
                candidate_todos=todos,
                existing_todos=state.get("all_tasks") or [],
                plan_modified=bool(state.get("has_existing_plan")),
                source=source,
                start_iteration=previous_iterations,
                user_id=state.get("user_id"),
                model_request=state.get("model_request"),
            )
        except Exception as exc:
            from app.ai.planning_rubric import FALLBACK_PLANNING_RUBRIC, PlanningRubricAttempt

            return PlanningRubricAttempt(
                status="grader_error",
                iterations=previous_iterations,
                source=source,
                rubric=FALLBACK_PLANNING_RUBRIC,
                rubric_source="fallback",
                evaluations=[],
                feedback=f"Planning rubric review failed: {exc}",
                error=f"Planning rubric review failed: {exc}",
            )

    async def _planning_tools_node(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            return state

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return state

        todos = list(state.get("todos", []))  # Make a copy
        current_task_index = state.get("current_task_index")
        tool_outputs = []
        write_todos_actions: list[str] = []

        # Track errors for circuit breaker
        context = GraphStateView(state).context_copy()
        had_error = False
        max_todos = getattr(settings, "max_todos_per_plan", 50)
        conversation_id = state.get("conversation_id")
        user_id = state.get("user_id")

        # Separate write_todos calls from external/MCP tool calls.
        normalized_calls = [normalize_tool_call(tc) for tc in last_message.tool_calls]
        external_tool_calls = [tc for tc in normalized_calls if tc.get("name") != "write_todos"]
        write_todos_calls = [tc for tc in normalized_calls if tc.get("name") == "write_todos"]

        tool_map: dict[str, Any] = {}
        handoff_tool = self._handoff_tool_for_agent(state, "planning_agent")
        scoped_internal_tools = [handoff_tool] if handoff_tool else None
        if external_tool_calls:
            tool_map = await ensure_agent_tool_map(
                self.planning_agent,
                conversation_id=conversation_id,
                user_id=user_id,
                device_id=state.get("device_id"),
                internal_tools=scoped_internal_tools,
            )

        # Make Planning-supervisor-only tools (e.g. dispatch_subagents)
        # callable from this turn. They are not part of the agent's MCP/server
        # tool registry, so ensure_agent_tool_map doesn't surface them.
        if any(tc.get("name") == "dispatch_subagents" for tc in external_tool_calls):
            for tool in self._build_planning_internal_tools(state, executable=True):
                tool_name = getattr(tool, "name", None)
                if tool_name and tool_name not in tool_map:
                    tool_map[tool_name] = tool

        tool_names_all = [tc.get("name") for tc in normalized_calls]
        logger.debug(f"[Planning Tools Node] Executing tools: {tool_names_all}")

        # --- HITL approval gate for external (non-write_todos) tool calls ---
        rejected_tool_ids: dict[str, str] = {}  # tool_call_id -> rejection reason
        rejected_feedback: dict[str, str] = {}
        approved_external_calls = list(external_tool_calls)

        if external_tool_calls and await self._needs_approval(
            state, external_tool_calls, agent=self.planning_agent, tool_map=tool_map
        ):
            # Label the stop reason before yielding to the human.
            _ctx = dict(state.get("context") or {})
            _ctx["pause_reason"] = "awaiting_approval"
            state["context"] = _ctx

            interrupt_payload = await self._prepare_interrupt_payload(
                state,
                tool_calls=external_tool_calls,
                agent=self.planning_agent,
                tool_map=tool_map,
            )
            human_decisions = interrupt(interrupt_payload)
            _ctx = GraphStateView(state).context_copy()
            _ctx.pop("pause_reason", None)
            state["context"] = _ctx

            if not human_decisions:
                approved_external_calls, rejected_feedback = _apply_decisions(
                    external_tool_calls, []
                )
            else:
                approved_external_calls, rejected_feedback = _apply_decisions(
                    external_tool_calls, human_decisions
                )

            for tc_id, feedback in rejected_feedback.items():
                rejected_tool_ids[tc_id] = feedback

        # Emit rejection ToolMessages for rejected external calls
        for tc in external_tool_calls:
            tc_id = tc.get("id")
            if tc_id in rejected_tool_ids:
                tool_outputs.append(
                    {
                        "tool_call_id": tc_id,
                        "name": tc.get("name"),
                        "content": rejected_tool_ids[tc_id],
                    }
                )

        # --- Artifact + image tracking (BP-1) ---
        tool_artifacts: list[dict[str, Any]] = []
        all_images: list[dict[str, str]] = []

        if rejected_feedback:
            tool_artifacts.extend(
                build_rejected_tool_artifacts(
                    tool_calls=external_tool_calls,
                    rejected_feedback=rejected_feedback,
                )
            )

        if approved_external_calls:
            (
                external_outputs,
                external_artifacts,
                external_images,
            ) = await self._execute_agent_tool_calls(
                state=state,
                agent=self.planning_agent,
                tool_calls=approved_external_calls,
                tool_map=tool_map,
                capture_images=True,
                internal_tools=scoped_internal_tools,
            )
            tool_outputs.extend(external_outputs)
            tool_artifacts.extend(external_artifacts)
            all_images.extend(external_images)
            had_error = had_error or any(
                artifact.get("status") == "error" for artifact in external_artifacts
            )

            # If the Planning Agent invoked hand_off, switch the selected agent so
            # the conditional edge from planning_tools can route to the target.
            self._apply_hand_off_if_present(state, external_outputs)

        # Execute write_todos calls (always permitted — internal state mutations)
        for tool_call_data in write_todos_calls:
            tool_name = tool_call_data.get("name")
            tool_id = tool_call_data.get("id")
            tool_args = tool_call_data.get("args", {})

            try:
                todos, current_task_index, result, action = apply_write_todos_action(
                    todos=todos,
                    current_task_index=current_task_index,
                    tool_args=tool_args,
                    max_todos=max_todos,
                )
                write_todos_actions.append(action)

                if action == "set_todos" and result.startswith("Error: Plan exceeds maximum"):
                    requested = len(tool_args.get("todos", []) or [])
                    logger.warning(
                        "Rejected plan with %d todos (max: %d)",
                        requested,
                        max_todos,
                    )

            except Exception as e:
                raw_action = tool_args.get("action")
                action = raw_action.value if hasattr(raw_action, "value") else raw_action
                result = f"Error executing {action}: {str(e)}"
                had_error = True  # Mark error for circuit breaker

            tool_outputs.append(
                {
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": result,
                }
            )

        # Update state with new todos
        state["todos"] = todos
        state["current_task_index"] = current_task_index

        # Check if plan was modified (SET_TODOS, ADD_TODO, UPDATE_TODO, REMOVE_TODO)
        # and mark context so we can return to agent for response generation
        context = state.get("context", {})
        plan_modifying_actions = {"set_todos", "add_todo", "update_todo", "remove_todo"}
        execution_actions = {"start_todo", "complete_todo"}

        for action in write_todos_actions:
            if action in plan_modifying_actions:
                context["plan_just_modified"] = True
            if action in execution_actions:
                state["planning_phase"] = "executing"
                logger.debug(f"Switched to executing phase due to {action} action")
        state["context"] = context

        # Grade plan-mutating write_todos output against the planning rubric.
        # On needs_revision, store actionable feedback and route back to the
        # planning agent instead of emitting the normal plan-summary pass.
        if any(action in plan_modifying_actions for action in write_todos_actions):
            rubric_attempt = await self._review_planning_todos_with_rubric(
                state=state,
                todos=todos,
                source="planning_tools",
            )
            rubric_status = getattr(rubric_attempt, "status", None)
            if rubric_status != "disabled":
                context["planning_rubric"] = rubric_attempt.metadata()
            if rubric_status == "needs_revision" and getattr(rubric_attempt, "feedback", None):
                context["planning_rubric_feedback"] = rubric_attempt.feedback
                context["plan_just_modified"] = False
                context.pop("generate_plan_response", None)
            elif rubric_status in {"satisfied", "max_iterations_reached"}:
                context.pop("planning_rubric_feedback", None)
            state["context"] = context

        self._apply_tool_outputs_to_state(
            state,
            tool_outputs=tool_outputs,
            tool_artifacts=tool_artifacts,
            all_images=all_images,
            mirror_to_response=True,
        )

        # Update consecutive_errors counter for circuit breaker. Increments
        # below the warn-threshold are routine retries and stay at INFO so the
        # WARNING level remains a useful "near the breaker" signal regardless
        # of how aggressively ``planning_consecutive_errors_limit`` is tuned.
        context = GraphStateView(state).context_copy()
        if had_error:
            context["consecutive_errors"] = context.get("consecutive_errors", 0) + 1
            current = context["consecutive_errors"]
            error_limit = settings.planning_consecutive_errors_limit
            # Require at least 2 errors before warning (a single transient
            # failure should never warn) AND warn only at the step before the
            # breaker fires. The breaker itself logs its own WARNING when it
            # actually trips, so we don't duplicate that here.
            warn_threshold = max(2, error_limit - 1)
            if current >= warn_threshold and current < error_limit:
                logger.warning("Planning consecutive errors: %d/%d", current, error_limit)
            else:
                logger.info("Planning consecutive errors: %d/%d", current, error_limit)
        else:
            # Reset on successful iteration
            context["consecutive_errors"] = 0
        state["context"] = context

        return state

    def _should_call_planning_tools(self, state: GraphState) -> str:
        messages = state.get("messages", [])
        if not messages:
            return "end"

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return "end"

        return "planning_tools"

    def _should_continue_planning(self, state: GraphState) -> str:
        # hand_off applied during this planning_tools turn re-routes the
        # conversation to a different top-level agent. Honor the new
        # selected_agent so the planning loop yields to the target node.
        delegated_agent = state.get("selected_agent")
        if isinstance(delegated_agent, str) and delegated_agent != "planning_agent":
            is_base_agent = delegated_agent in self.agents
            is_attached_custom = self._is_attached_custom_agent(state, delegated_agent)
            if is_base_agent or is_attached_custom:
                delegated_node = self._route_target_for(state, delegated_agent)
                logger.info(
                    "Planning hand_off detected: routing planning_tools to %s via %s",
                    delegated_agent,
                    delegated_node,
                )
                return delegated_node

        planning_call_count = state.get("planning_call_count", 0)
        max_iterations = int(getattr(settings, "planning_max_iterations", 0) or 0)
        planning_budget_enabled = max_iterations > 0
        context = GraphStateView(state).context_copy()
        planning_phase = state.get("planning_phase", "planning")
        last_message_is_tool_output = self._last_message_is_tool_output(state)

        # Planning rubric feedback wins over the normal prose summary pass: the
        # plan needs a forced write_todos revision before any user-facing reply.
        if context.get("planning_rubric_feedback"):
            logger.debug("[Should Continue Planning] Decision: planning_agent (rubric feedback)")
            return "planning_agent"

        # If plan state was just mutated, always give the Planning Agent one
        # response pass so the user does not get an empty tool-calling message.
        if context.get("plan_just_modified"):
            context["plan_just_modified"] = False
            context["generate_plan_response"] = True
            state["context"] = context
            if planning_budget_enabled and planning_call_count >= max_iterations:
                context["pause_reason"] = "max_iterations_reached"
                state["context"] = context
                self._mark_force_final_response(
                    state,
                    reason="max_iterations_reached",
                    scope="planning",
                    count=planning_call_count,
                    limit=max_iterations,
                )
            logger.debug(
                "[Should Continue Planning] Decision: planning_agent (plan_just_modified=True)"
            )
            return "planning_agent"

        # Hard budget: if the latest thing is a tool result, route back once
        # with tools disabled for a final synthesis instead of ending on the
        # empty intermediate tool-calling response.
        if planning_budget_enabled and planning_call_count >= max_iterations:
            context["pause_reason"] = "max_iterations_reached"
            state["context"] = context
            if last_message_is_tool_output:
                self._mark_force_final_response(
                    state,
                    reason="max_iterations_reached",
                    scope="planning",
                    count=planning_call_count,
                    limit=max_iterations,
                )
                logger.warning(
                    "Planning budget reached after tool output: %d >= %d; "
                    "routing to final synthesis",
                    planning_call_count,
                    max_iterations,
                )
                return "planning_agent"

            self._set_continuation_signal(
                state,
                should_continue=settings.auto_continue_enabled,
                reason="max_iterations_reached",
                scope="planning",
                count=planning_call_count,
                limit=max_iterations,
            )
            logger.warning(f"Planning budget exceeded: {planning_call_count} >= {max_iterations}")
            return "end"

        # Soft-limit: if auto-continue is enabled, trigger continuation at
        # a fraction of the planning budget.
        if planning_budget_enabled and settings.auto_continue_enabled:
            soft_limit = max(1, int(max_iterations * settings.auto_continue_soft_limit_ratio))
            if planning_call_count >= soft_limit:
                if last_message_is_tool_output:
                    logger.debug(
                        "[Should Continue Planning] Decision: planning_agent "
                        "(soft budget reached after tool output; reconcile before pausing)"
                    )
                    return "planning_agent"

                self._set_continuation_signal(
                    state,
                    should_continue=True,
                    reason="soft_budget",
                    scope="planning",
                    count=planning_call_count,
                    limit=soft_limit,
                )
                logger.info(
                    "Planning soft-limit reached: %d >= %d, requesting auto-continue",
                    planning_call_count,
                    soft_limit,
                )
                return "end"

        # Circuit breaker: check consecutive errors
        consecutive_errors = context.get("consecutive_errors", 0)
        max_consecutive_errors = settings.planning_consecutive_errors_limit

        if consecutive_errors >= max_consecutive_errors:
            context["pause_reason"] = "consecutive_errors_limit"
            state["context"] = context
            self._set_continuation_signal(
                state,
                should_continue=False,
                reason="consecutive_errors_limit",
                scope="planning",
                count=consecutive_errors,
                limit=max_consecutive_errors,
            )
            logger.warning(
                f"Planning circuit breaker triggered: {consecutive_errors} consecutive errors"
            )
            return "end"

        # In planning phase, always give the agent a chance to respond after tools
        if planning_phase == "planning":
            messages = state.get("messages", [])
            last_msg_type = type(messages[-1]).__name__ if messages else "None"

            # If last message is a ToolMessage, give agent a chance to process results
            if messages and isinstance(messages[-1], ToolMessage):
                logger.debug(
                    "[Should Continue Planning] Decision: planning_agent "
                    "(last_message=ToolMessage, agent needs to respond)"
                )
                return "planning_agent"

            # Agent already responded with text - end planning loop
            logger.debug(
                f"[Should Continue Planning] Decision: end (planning_phase=planning, "
                f"last_message_type={last_msg_type})"
            )
            return "end"

        # === EXECUTING PHASE LOGIC ===
        todos = state.get("todos", [])
        if todos:
            pending_count = sum(
                1
                for t in todos
                if t.get("status")
                in (
                    TodoStatus.PENDING.value,
                    "pending",
                    TodoStatus.IN_PROGRESS.value,
                    "in_progress",
                )
            )
            if pending_count == 0:
                context["all_tasks_completed"] = True
                state["context"] = context

                # Check if we've already generated the final summary
                if context.get("final_summary_generated"):
                    return "end"

                # Need one more iteration to generate completion summary
                context["generate_final_summary"] = True
                state["context"] = context
                return "planning_agent"

        return "planning_agent"
