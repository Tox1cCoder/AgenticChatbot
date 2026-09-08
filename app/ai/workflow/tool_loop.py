"""Generic tool / HITL loop helpers extracted from MultiAgentWorkflow (Task 8, Step 4).

Methods are relocated verbatim; behavior is identical.
"""

import json
import logging
from typing import Any

from langchain_core.messages import ToolMessage

from app.ai.hitl_config import (
    any_call_requires_approval,
    build_tool_interrupt_payload,
    policy_from_context,
)
from app.ai.mcp_registry import get_global_mcp_manager
from app.ai.rich_image_selection import apply_rich_image_selection
from app.ai.schemas import GraphState, GraphStateView
from app.ai.token_instrumentation import truncate_tool_result
from app.ai.tool_context import (
    rich_response_capable_from_context,
    tool_execution_context,
)
from app.ai.tool_execution import (
    ensure_agent_tool_map,
    execute_tool_calls,
)
from app.ai.utils import (
    apply_hitl_decisions,
    make_json_safe,
    normalize_tool_call,
)
from app.core.config import settings

logger = logging.getLogger("app.ai.graph")
_apply_decisions = apply_hitl_decisions


class ToolLoopMixin:
    """Relocated tool_loop methods for :class:`MultiAgentWorkflow`."""

    @staticmethod
    def _lift_rich_candidates(
        context: dict[str, Any],
        tool_artifacts: list[dict[str, Any]],
    ) -> None:
        """Move transient artifact candidates into turn context and sanitize artifacts."""
        existing_candidates: list[dict[str, Any]] = list(context.get("rich_item_candidates", []))
        seen_ids = {c.get("id") for c in existing_candidates if isinstance(c, dict)}
        for artifact in tool_artifacts:
            if not isinstance(artifact, dict):
                continue
            candidates = artifact.pop("_rich_item_candidates", None)
            if not isinstance(candidates, list):
                continue
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                candidate_id = candidate.get("id")
                if not candidate_id or candidate_id in seen_ids:
                    continue
                existing_candidates.append(candidate)
                seen_ids.add(candidate_id)
        if existing_candidates:
            context["rich_item_candidates"] = existing_candidates
            apply_rich_image_selection(context)

    def _apply_hand_off_if_present(self, state: GraphState, tool_outputs: list) -> GraphState:
        """Interpret one canonical handoff output before ToolMessages are persisted."""
        handoff_outputs = [
            output
            for output in tool_outputs
            if isinstance(output, dict) and output.get("name") == "hand_off"
        ]
        if not handoff_outputs:
            return state

        def reject(output: dict[str, Any], message: str) -> None:
            output["content"] = f"Hand-off refused: {message}"

        if len(handoff_outputs) != 1:
            for output in handoff_outputs:
                reject(output, "exactly one hand_off call is allowed per model message.")
            return state

        handoff_output = handoff_outputs[0]
        try:
            payload = json.loads(handoff_output.get("content", ""))
        except (TypeError, json.JSONDecodeError):
            reject(handoff_output, "the tool output must be canonical handoff JSON.")
            return state

        if (
            not isinstance(payload, dict)
            or set(payload) != {"hand_off"}
            or not isinstance(payload.get("hand_off"), str)
            or not payload["hand_off"].strip()
        ):
            reject(handoff_output, "the tool output must contain exactly one target agent.")
            return state

        source_agent = state.get("active_agent_id")
        if not isinstance(source_agent, str) or not source_agent:
            reject(handoff_output, "the active agent is unavailable.")
            return state

        target_agent = payload["hand_off"]
        allowed_targets = set(self._handoff_targets(state, source_agent))
        if target_agent not in allowed_targets:
            reject(
                handoff_output,
                f"'{target_agent}' is not a reachable target for {source_agent}.",
            )
            return state

        context = state.get("context")
        if not isinstance(context, dict):
            context = {}
        trail = context.get("agents_invoked")
        visited_agents = (
            {
                entry.get("id")
                for entry in trail
                if isinstance(entry, dict) and isinstance(entry.get("id"), str)
            }
            if isinstance(trail, list)
            else set()
        )
        if target_agent in visited_agents:
            reject(handoff_output, f"'{target_agent}' has already handled this turn.")
            return state

        context = GraphStateView(state).context()
        delegation_count = int(context.get("delegation_count") or 0)
        max_delegation_depth = settings.max_handoff_delegation_depth
        if delegation_count >= max_delegation_depth:
            reject(
                handoff_output,
                f"maximum delegation depth of {max_delegation_depth} reached; answer directly.",
            )
            return state

        logger.info("Delegating from '%s' to '%s'", source_agent, target_agent)
        state["active_agent_id"] = target_agent
        updated_context = dict(context)
        updated_context["delegation_count"] = delegation_count + 1
        state["context"] = updated_context
        context["handoff"] = {
            "active": True,
            "source_agent": source_agent,
            "target_agent": target_agent,
            "tool_call_id": handoff_output.get("tool_call_id"),
        }
        state["context"] = context
        self._record_agent_invocation(state, target_agent, via="handoff")
        return state

    async def _needs_approval(
        self,
        state: GraphState,
        normalized_calls: list[dict[str, Any]],
        *,
        agent: Any | None = None,
        tool_map: dict[str, Any] | None = None,
        internal_tools: list[Any] | None = None,
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
                internal_tools=internal_tools,
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
        internal_tools: list[Any] | None = None,
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
                internal_tools=internal_tools,
            )

        payload = build_tool_interrupt_payload(
            normalized_calls, tool_map=tool_map, device_id=state_view.device_id()
        )

        context = state_view.context_copy()
        context["pending_action_requests"] = payload["action_requests"]
        if payload.get("metadata"):
            context["interrupt_metadata"] = payload["metadata"]
        state["context"] = context

        return payload

    async def _execute_agent_tool_calls(
        self,
        *,
        state: GraphState,
        agent: Any,
        tool_calls: list[Any],
        tool_map: dict[str, Any] | None = None,
        capture_images: bool = True,
        internal_tools: list[Any] | None = None,
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
                internal_tools=internal_tools,
            )

        with tool_execution_context(
            conversation_id,
            user_id,
            agent_key,
            device_id,
            rich_response_capable=rich_response_capable_from_context(state.get("context")),
            logical_turn_id=state_view.logical_turn_id(),
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
            # Outputs flagged preserve_full_content (skill terminal errors)
            # must reach the model verbatim, never truncated.
            if truncate_outputs and max_chars > 0 and not output.get("preserve_full_content"):
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

        # Candidate records are turn-internal handoff data. Lift and remove
        # them before tool artifacts can reach persisted response metadata.
        if tool_artifacts:
            self._lift_rich_candidates(context, tool_artifacts)
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
