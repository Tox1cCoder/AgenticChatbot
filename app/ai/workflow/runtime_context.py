"""Typed runtime context handed to routing-v2 graph nodes.

Nodes read collaborators from here instead of closing over the workflow
instance, so a node can be tested with a plain object and the graph never
carries request-scoped services in checkpointed state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.ai.workflow.inventory import RoutingInventory, build_routing_inventory
from app.ai.workflow.routing import RoutingContextRequest

logger = logging.getLogger(__name__)

__all__ = ["WorkflowRuntimeContext", "build_runtime_inventory"]


def build_runtime_inventory(
    *,
    base_agent_ids: list[str],
    custom_agents: dict[str, Any] | None,
    max_custom_agents: int | None = None,
) -> RoutingInventory:
    """Snapshot the routable specialists for one request."""
    return build_routing_inventory(
        base_agent_ids=base_agent_ids,
        custom_agents=custom_agents,
        max_custom_agents=max_custom_agents,
    )


@dataclass
class WorkflowRuntimeContext:
    """Per-invocation collaborators for the parent graph.

    ``inventory`` is a snapshot taken before the turn starts, so the router and
    the transition resolver validate against the same immutable target set.
    """

    routing_service: Any
    inventory: RoutingInventory
    routing_context_builder: Any = None
    active_canvas: dict[str, Any] | None = None
    runtime_time: str | None = None
    locale: str | None = None

    async def build_routing_context(self, state: dict[str, Any], inventory: RoutingInventory):
        """Assemble the bounded routing context for this turn.

        Everything below is descriptive context. None of it selects an agent.
        """
        builder = self.routing_context_builder or getattr(
            self.routing_service, "context_builder", None
        )
        if builder is None:
            raise RuntimeError("no routing context builder is configured")

        messages = state.get("messages") or []
        message_text = ""
        for message in reversed(messages):
            if getattr(message, "type", None) == "human":
                content = getattr(message, "content", "")
                message_text = content if isinstance(content, str) else str(content)
                break

        planning = {
            "planning_mode_enabled": bool(state.get("planning_mode_enabled")),
            "has_existing_plan": bool(state.get("has_existing_plan")),
            "lifecycle": _lifecycle_value(state.get("plan_lifecycle")),
            "summary": _plan_summary(state),
        }

        context_data = state.get("context") or {}
        active_canvas = self.active_canvas
        if active_canvas is None and isinstance(context_data, dict):
            candidate = context_data.get("active_canvas")
            active_canvas = candidate if isinstance(candidate, dict) else None

        attachments = state.get("attachments") or []

        return await builder.build(
            RoutingContextRequest(
                message=message_text,
                inventory=inventory,
                conversation_id=state.get("conversation_id"),
                user_id=state.get("user_id"),
                device_id=state.get("device_id"),
                user_message_id=state.get("user_message_id"),
                persona=state.get("persona"),
                active_canvas=active_canvas,
                planning=planning,
                attachment_count=len(attachments) if isinstance(attachments, list) else 0,
                locale=self.locale,
                runtime_time=self.runtime_time,
            )
        )


def _lifecycle_value(lifecycle: Any) -> str | None:
    if lifecycle is None:
        return None
    return str(getattr(lifecycle, "value", lifecycle))


def _plan_summary(state: dict[str, Any]) -> str:
    todos = state.get("todos")
    if not isinstance(todos, list) or not todos:
        return ""
    open_count = sum(
        1 for todo in todos if isinstance(todo, dict) and todo.get("status") != "completed"
    )
    return f"{len(todos)} plan items, {open_count} not completed"
