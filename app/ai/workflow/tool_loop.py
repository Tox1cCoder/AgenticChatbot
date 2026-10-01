"""Generic tool / HITL loop helpers extracted from MultiAgentWorkflow (Task 8, Step 4).

Methods are relocated verbatim; behavior is identical.
"""

import json
from typing import Any

from langchain_core.messages import ToolMessage

from app.ai.rich_image_selection import apply_rich_image_selection
from app.ai.schemas import GraphState, GraphStateView
from app.ai.utils import make_json_safe
from app.core.config import settings


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
