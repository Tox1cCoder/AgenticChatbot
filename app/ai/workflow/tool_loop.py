"""Generic tool / HITL loop helpers extracted from MultiAgentWorkflow (Task 8, Step 4).

Methods are relocated verbatim; behavior is identical.
"""

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import interrupt

from app.ai.hand_off_tool import MAX_DELEGATION_DEPTH
from app.ai.hitl_config import (
    any_call_requires_approval,
    policy_from_context,
    redact_sensitive_args,
)
from app.ai.mcp_registry import get_global_mcp_manager
from app.ai.schemas import GraphState, GraphStateView
from app.ai.token_instrumentation import truncate_tool_result
from app.ai.tool_context import tool_execution_context
from app.ai.tool_execution import (
    build_rejected_tool_artifacts,
    ensure_agent_tool_map,
    execute_tool_calls,
)
from app.ai.utils import (
    apply_hitl_decisions,
    find_pending_tool_call_message,
    make_json_safe,
    normalize_tool_call,
)
from app.core.config import settings

logger = logging.getLogger("app.ai.graph")
_apply_decisions = apply_hitl_decisions


class ToolLoopMixin:
    """Relocated tool_loop methods for :class:`MultiAgentWorkflow`."""

    async def _tool_node(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        pending_tool_message = find_pending_tool_call_message(messages)
        if not pending_tool_message:
            return state

        pending_message_idx, pending_message = pending_tool_message

        # Skip tool calls that already have a ToolMessage (e.g. HITL rejections).
        # This keeps the AIMessage tool_calls intact (needed for a valid LLM message
        # sequence) while avoiding re-execution of calls that were already resolved.
        already_resolved_ids = {
            msg.tool_call_id
            for msg in messages[pending_message_idx + 1 :]
            if isinstance(msg, ToolMessage) and getattr(msg, "tool_call_id", None)
        }
        tool_calls_pending = [
            tc
            for tc in pending_message.tool_calls
            if normalize_tool_call(tc).get("id") not in already_resolved_ids
        ]
        if not tool_calls_pending:
            # All tool calls for this AI message are already resolved (all rejected).
            # Return early so _route_tool_output can send the agent back to re-respond.
            return state

        selected_agent_name = state.get("selected_agent")
        agent = self._resolve_runtime_agent(state, selected_agent_name)
        if not agent:
            logger.warning(
                "Skipping tool execution: selected_agent '%s' not in agent registry",
                selected_agent_name,
            )
            # Strip tool_calls from the pending AIMessage to prevent downstream
            # routing confusion when the tools node cannot execute anything.
            sanitized = AIMessage(content=pending_message.content or "")
            state["messages"] = (
                messages[:pending_message_idx] + [sanitized] + messages[pending_message_idx + 1 :]
            )
            return state

        tool_map = await ensure_agent_tool_map(
            agent,
            conversation_id=state.get("conversation_id"),
            user_id=state.get("user_id"),
            device_id=state.get("device_id"),
        )

        tool_outputs, tool_artifacts, all_images = await self._execute_agent_tool_calls(
            state=state,
            agent=agent,
            tool_calls=tool_calls_pending,
            tool_map=tool_map,
            capture_images=True,
        )
        self._apply_tool_outputs_to_state(
            state,
            tool_outputs=tool_outputs,
            tool_artifacts=tool_artifacts,
            all_images=all_images,
            truncate_outputs=True,
        )

        # ── Inter-agent delegation (hand_off tool) ──────────────────────
        state = self._apply_hand_off_if_present(state, tool_outputs)

        return state

    def _apply_hand_off_if_present(self, state: GraphState, tool_outputs: list) -> GraphState:
        """Detect a hand_off tool result and re-route to the target agent.

        If ``delegation_count`` exceeds ``MAX_DELEGATION_DEPTH`` the delegation
        is rejected and an explanatory ToolMessage is appended instead.
        """
        hand_off_output = None
        for output in tool_outputs:
            if output.get("name") == "hand_off":
                hand_off_output = output
                break
        if hand_off_output is None:
            return state

        try:
            payload = json.loads(hand_off_output["content"])
            target_agent = payload.get("hand_off")
            reason = payload.get("reason", "")
        except (json.JSONDecodeError, KeyError):
            logger.warning("Malformed hand_off tool output; ignoring delegation")
            return state

        # Validate target agent exists: a base agent or an attached custom agent.
        if target_agent not in self.agents and not self._is_attached_custom_agent(
            state, target_agent
        ):
            logger.warning("hand_off requested unknown/unattached agent '%s'", target_agent)
            tool_call_id = hand_off_output.get("tool_call_id")
            if tool_call_id:
                state.setdefault("messages", []).append(
                    ToolMessage(
                        content=(
                            f"Hand-off refused: '{target_agent}' is not a valid target. "
                            "It is not a base agent and not a custom agent attached to this "
                            "conversation. Answer the request yourself or hand off to a listed "
                            "target."
                        ),
                        tool_call_id=tool_call_id,
                        name="hand_off",
                    )
                )
            return state

        # Circuit-breaker: cap delegation depth
        delegation_count = state.get("delegation_count") or 0
        if delegation_count >= MAX_DELEGATION_DEPTH:
            logger.warning(
                "Delegation depth %d reached limit of %d; refusing hand_off to '%s'",
                delegation_count,
                MAX_DELEGATION_DEPTH,
                target_agent,
            )
            state.setdefault("messages", []).append(
                ToolMessage(
                    content=(
                        f"Delegation refused: maximum depth of {MAX_DELEGATION_DEPTH} reached. "
                        "Please answer the user's request directly."
                    ),
                    tool_call_id=hand_off_output["tool_call_id"],
                    name="hand_off",
                )
            )
            return state

        previous_agent = state.get("selected_agent")
        logger.info(
            "Delegating from '%s' → '%s' (reason: %s)",
            previous_agent,
            target_agent,
            reason,
        )
        state["selected_agent"] = target_agent
        state["delegation_count"] = delegation_count + 1

        # Control-plane handoff metadata — used by ``_messages_for_selected_agent``
        # to strip handoff control AIMessage/ToolMessage pairs out of the
        # delegated agent's prompt, and by the streamer to emit an
        # ``agent_selected`` event when the active agent changes.
        context = state.get("context") or {}
        if not isinstance(context, dict):
            context = {}
        context["handoff"] = {
            "active": True,
            "source_agent": previous_agent,
            "target_agent": target_agent,
            "reason": reason,
            "tool_call_id": hand_off_output.get("tool_call_id"),
        }
        state["context"] = context
        self._record_agent_invocation(state, target_agent, via="handoff", reason=reason)
        return state

    async def _needs_approval(
        self,
        state: GraphState,
        normalized_calls: list[dict[str, Any]],
        *,
        agent: Any | None = None,
        tool_map: dict[str, Any] | None = None,
    ) -> bool:
        """Resolve per-call provenance and apply the per-turn HITL policy."""
        policy = policy_from_context(state.get("context"))
        if not policy.get("master_enabled", True):
            return False

        mcp_manager = None
        if tool_map is None and agent is not None:
            view = GraphStateView(state)
            tool_map = await ensure_agent_tool_map(
                agent,
                conversation_id=view.conversation_id(),
                user_id=view.user_id(),
                device_id=view.device_id(),
            )
        if tool_map is not None:
            mcp_manager = await get_global_mcp_manager()

        return any_call_requires_approval(
            normalized_calls, policy=policy, tool_map=tool_map, mcp_manager=mcp_manager
        )

    async def _prepare_interrupt_payload(
        self,
        state: GraphState,
        *,
        tool_calls: list[Any],
        agent: Any | None,
        tool_map: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Attach device/runtime provenance to pending tool approvals."""
        state_view = GraphStateView(state)
        normalized_calls = [normalize_tool_call(tool_call) for tool_call in tool_calls]
        if tool_map is None and agent is not None:
            tool_map = await ensure_agent_tool_map(
                agent,
                conversation_id=state_view.conversation_id(),
                user_id=state_view.user_id(),
                device_id=state_view.device_id(),
            )

        device_id = state_view.device_id()
        provenance: dict[str, dict[str, Any]] = {}
        enriched_calls: list[dict[str, Any]] = []

        for tool_call in normalized_calls:
            enriched_call = dict(tool_call)
            # Redact sensitive-looking argument values from the human approval
            # prompt (and thus the downstream API InterruptResponse, which
            # re-parses this payload). redact_sensitive_args returns a fresh
            # dict, so the tool call that ACTUALLY executes on approval keeps
            # its real args untouched. Conservative key-name match, so normal
            # arguments stay visible for the approver.
            if isinstance(enriched_call.get("args"), dict):
                enriched_call["args"] = redact_sensitive_args(enriched_call["args"])
            tool_call_id = enriched_call.get("id") or enriched_call.get("tool_call_id")
            if tool_call_id and "tool_call_id" not in enriched_call:
                enriched_call["tool_call_id"] = tool_call_id

            tool_name = enriched_call.get("name")
            tool = tool_map.get(tool_name) if tool_map and tool_name else None
            tool_metadata = getattr(tool, "metadata", None) if tool is not None else None

            provenance_entry: dict[str, Any] = {}
            if device_id:
                provenance_entry["device_id"] = device_id
            if isinstance(tool_metadata, dict):
                for field_name in (
                    "tool_origin",
                    "server_name",
                    "qualified_tool_id",
                    "tool_instance_id",
                    "session_id",
                    "catalog_version",
                ):
                    if field_name not in tool_metadata:
                        continue
                    if tool_metadata[field_name] is None:
                        continue
                    if tool_metadata[field_name] == "":
                        continue
                    provenance_entry[field_name] = tool_metadata[field_name]

            if provenance_entry:
                provenance_key = str(tool_call_id or tool_name or len(provenance))
                provenance[provenance_key] = provenance_entry

            enriched_calls.append(enriched_call)

        interrupt_metadata: dict[str, Any] = {}
        if device_id:
            interrupt_metadata["device_id"] = device_id
        if provenance:
            interrupt_metadata["tool_provenance"] = provenance

        context = state_view.context_copy()
        context["pending_action_requests"] = enriched_calls
        if interrupt_metadata:
            context["interrupt_metadata"] = interrupt_metadata
        state["context"] = context

        payload: dict[str, Any] = {"action_requests": enriched_calls}
        if interrupt_metadata:
            payload["metadata"] = interrupt_metadata
        return payload

    async def _approval_node(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            return state

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return state

        selected_agent_name = state.get("selected_agent")
        agent = self.agents.get(selected_agent_name) if selected_agent_name else None
        interrupt_payload = await self._prepare_interrupt_payload(
            state,
            tool_calls=last_message.tool_calls,
            agent=agent,
        )
        interrupt_payload["action_requests"]

        # Label the stop reason before yielding to the human so callers can
        # distinguish approval-gate pauses from budget/error pauses.
        context = dict(state.get("context") or {})
        context["pause_reason"] = "awaiting_approval"
        state["context"] = context

        human_decisions = interrupt(interrupt_payload)
        context = GraphStateView(state).context_copy()
        context.pop("pause_reason", None)
        state["context"] = context

        if not human_decisions:
            _, rejected_feedback = _apply_decisions(last_message.tool_calls, [])
        else:
            _, rejected_feedback = _apply_decisions(last_message.tool_calls, human_decisions)

        rejection_messages = [
            ToolMessage(
                content=rejected_feedback[tc.get("id")],
                tool_call_id=tc.get("id"),
                name=tc.get("name"),
            )
            for tc in last_message.tool_calls
            if tc.get("id") in rejected_feedback
        ]

        if rejected_feedback:
            context = state.get("context", {})
            existing_artifacts = context.get("tool_artifacts", [])
            existing_artifacts.extend(
                build_rejected_tool_artifacts(
                    tool_calls=last_message.tool_calls,
                    rejected_feedback=rejected_feedback,
                )
            )
            context["tool_artifacts"] = existing_artifacts
            state["context"] = context

        state["messages"] = messages[:-1] + [last_message] + rejection_messages

        return state

    async def _execute_agent_tool_calls(
        self,
        *,
        state: GraphState,
        agent: Any,
        tool_calls: list[Any],
        tool_map: dict[str, Any] | None = None,
        capture_images: bool = True,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
        if not tool_calls:
            return [], [], []

        state_view = GraphStateView(state)
        conversation_id = state_view.conversation_id()
        user_id = state_view.user_id()
        device_id = state_view.device_id()
        agent_key = (
            getattr(agent, "tool_state_key", None)
            or getattr(agent, "agent_config_key", None)
            or getattr(agent, "agent_id", None)
            or "unknown"
        )

        self._hydrate_deferred_tool_snapshot_from_state(state, agent=agent)

        if tool_map is None:
            tool_map = await ensure_agent_tool_map(
                agent,
                conversation_id=conversation_id,
                user_id=user_id,
                device_id=device_id,
            )

        with tool_execution_context(
            conversation_id,
            user_id,
            agent_key,
            device_id,
        ):
            outputs, artifacts, images = await execute_tool_calls(
                tool_calls=tool_calls,
                tool_map=tool_map,
                capture_images=capture_images,
                device_id=device_id,
                agent=agent,
                conversation_id=conversation_id,
                user_id=user_id,
            )
        self._persist_deferred_tool_snapshot_to_state(state, agent=agent)
        return outputs, artifacts, images

    def _tool_end_events_from_node_state(
        self,
        *,
        node_state: dict[str, Any],
        last_state_values: dict[str, Any] | None,
        emitted_tool_result_ids: set[str],
    ):
        messages = node_state.get("messages", [])
        if not isinstance(messages, list):
            messages = [messages]

        messages_to_emit = messages
        if messages and isinstance(messages[-1], ToolMessage):
            first_trailing_index = len(messages) - 1
            while first_trailing_index > 0 and isinstance(
                messages[first_trailing_index - 1], ToolMessage
            ):
                first_trailing_index -= 1
            messages_to_emit = messages[first_trailing_index:]

        for message in messages_to_emit:
            if not isinstance(message, ToolMessage):
                continue

            tool_call_id = getattr(message, "tool_call_id", None)
            dedupe_key = str(tool_call_id or f"{getattr(message, 'name', 'unknown')}:{id(message)}")
            if dedupe_key in emitted_tool_result_ids:
                continue
            emitted_tool_result_ids.add(dedupe_key)

            event_payload = {
                "type": "tool_end",
                "name": getattr(message, "name", "unknown"),
                "tool_call_id": tool_call_id,
                "result": make_json_safe(message.content),
            }
            render_payload = self._lookup_tool_render_payload(
                last_state_values,
                tool_call_id,
            ) or self._lookup_tool_render_payload(node_state, tool_call_id)
            if render_payload:
                event_payload["render"] = render_payload
            yield event_payload

    @staticmethod
    def _tool_error_signature(artifact: dict[str, Any]) -> dict[str, str]:
        try:
            args_key = json.dumps(
                make_json_safe(artifact.get("args") or {}),
                sort_keys=True,
                separators=(",", ":"),
            )
        except Exception:
            args_key = "{}"
        return {
            "tool": str(artifact.get("tool") or "unknown"),
            "error_type": str(artifact.get("error_type") or "unknown"),
            "args": args_key[:500],
        }

    def _update_tool_error_streak(
        self,
        state: GraphState,
        tool_artifacts: list[dict[str, Any]] | None,
    ) -> None:
        context = GraphStateView(state).context_copy()
        errors = [
            artifact
            for artifact in tool_artifacts or []
            if isinstance(artifact, dict) and artifact.get("status") == "error"
        ]
        if not errors:
            context.pop("tool_error_streak", None)
            state["context"] = context
            return

        signature = self._tool_error_signature(errors[0])
        prior = context.get("tool_error_streak")
        prior_signature = prior.get("signature") if isinstance(prior, dict) else None
        try:
            prior_count = int(prior.get("count", 0)) if isinstance(prior, dict) else 0
        except Exception:
            prior_count = 0
        count = prior_count + 1 if prior_signature == signature else 1
        limit = max(1, int(getattr(settings, "tool_execution_consecutive_errors_limit", 3) or 3))
        context["tool_error_streak"] = {
            "count": count,
            "limit": limit,
            "signature": signature,
        }
        state["context"] = context

    def _apply_tool_outputs_to_state(
        self,
        state: GraphState,
        *,
        tool_outputs: list[dict[str, Any]],
        tool_artifacts: list[dict[str, Any]] | None = None,
        all_images: list[dict[str, str]] | None = None,
        truncate_outputs: bool = False,
        mirror_to_response: bool = False,
    ) -> None:
        max_chars = getattr(settings, "tool_result_max_chars", 0) or 0
        truncation_suffix = getattr(
            settings,
            "tool_result_truncation_suffix",
            "\n\n[Output truncated - full result available in tool artifacts]",
        )

        assistant_message_id = state.get("assistant_message_id")
        for output in tool_outputs:
            content = output["content"]
            if truncate_outputs and max_chars > 0:
                content, was_truncated = truncate_tool_result(
                    content,
                    max_chars=max_chars,
                    truncation_suffix=truncation_suffix,
                )
                if was_truncated:
                    logger.debug(
                        "Truncated tool output for %s from %d to %d chars",
                        output["name"],
                        len(output["content"]),
                        len(content),
                    )

            tool_kwargs: dict[str, Any] = {
                "content": content,
                "tool_call_id": output["tool_call_id"],
                "name": output["name"],
            }
            # Stamp ToolMessages with a derived id so terminal-turn compaction
            # can remove them. tool_call_id alone is unique within a turn but
            # langchain BaseMessage id is what RemoveMessage targets.
            if assistant_message_id and output.get("tool_call_id"):
                tool_kwargs["id"] = f"{assistant_message_id}-toolmsg-{output['tool_call_id']}"
            state.setdefault("messages", []).append(ToolMessage(**tool_kwargs))

        state["iteration_count"] = (state.get("iteration_count") or 0) + 1

        context = GraphStateView(state).context_copy()
        if tool_artifacts:
            existing_artifacts = list(context.get("tool_artifacts", []))
            existing_artifacts.extend(tool_artifacts)
            context["tool_artifacts"] = existing_artifacts
        render_results = dict(context.get("tool_render_results", {}))
        for output in tool_outputs:
            tool_call_id = output.get("tool_call_id")
            render = output.get("render")
            if tool_call_id and isinstance(render, dict):
                render_results[str(tool_call_id)] = make_json_safe(render)
        if render_results:
            context["tool_render_results"] = render_results
        if all_images:
            existing_images = list(context.get("tool_images", []))
            existing_images.extend(all_images)
            context["tool_images"] = existing_images

        # Lift artifact-attached rich-item candidates into turn-scoped context.
        # Each artifact may carry `_rich_item_candidates`; merge unique by id.
        if tool_artifacts:
            existing_candidates: list[dict[str, Any]] = list(
                context.get("rich_item_candidates", [])
            )
            seen_ids = {c.get("id") for c in existing_candidates if isinstance(c, dict)}
            for artifact in tool_artifacts:
                if not isinstance(artifact, dict):
                    continue
                candidates = artifact.get("_rich_item_candidates")
                if not isinstance(candidates, list):
                    continue
                for candidate in candidates:
                    if not isinstance(candidate, dict):
                        continue
                    cid = candidate.get("id")
                    if not cid or cid in seen_ids:
                        continue
                    existing_candidates.append(candidate)
                    seen_ids.add(cid)
            if existing_candidates:
                context["rich_item_candidates"] = existing_candidates
        state["context"] = context

        self._update_tool_error_streak(state, tool_artifacts)

        if mirror_to_response:
            response = state.get("response")
            if response and tool_outputs:
                if response.tool_artifacts is None:
                    response.tool_artifacts = []
                for output in tool_outputs:
                    response.tool_artifacts.append(
                        {
                            "tool": output["name"],
                            "result": output["content"],
                        }
                    )
                state["response"] = response
