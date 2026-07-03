"""Canonical v3 → Streamlit internal JSON SSE adapter.

The Streamlit/demo client consumes the legacy JSON-only ``data:`` event
vocabulary (``token``/``thinking``/``tool``/``rich_items``/``interrupt``/
``complete``/``error``/``title_updated``/``node_complete``/...). This adapter
projects canonical :class:`V3StreamEvent` events back to those legacy dicts.
No ``[DONE]`` sentinel — the Streamlit client expects JSON-only events.
"""

from __future__ import annotations

from typing import Any

from .events import SUBAGENT_PHASE_BY_EVENT, V3StreamEvent


def legacy_event_from_v3(event: V3StreamEvent) -> dict[str, Any] | None:
    if event.type == "message_delta":
        return {"type": "token", "content": event.data.get("text", "")}
    if event.type == "reasoning_delta":
        return {"type": "thinking", "content": event.data.get("text", "")}
    if event.type == "tool_call_available":
        return {
            "type": "tool",
            "phase": "start",
            "status": "start",
            "state": "queued",
            "name": event.tool_name or "unknown",
            "tool_call_id": event.tool_call_id,
            "args": event.data.get("args"),
        }
    if event.type == "tool_execution_end":
        error = event.data.get("error")
        payload = {
            "type": "tool",
            "phase": "end",
            "status": "end",
            "state": "error" if error else "completed",
            "name": event.tool_name or "unknown",
            "tool_call_id": event.tool_call_id,
            "result": event.data.get("output"),
            "render": event.data.get("render"),
        }
        duration_ms = event.data.get("duration_ms")
        if duration_ms is not None:
            payload["duration_ms"] = duration_ms
        return payload
    if event.type == "rich_items":
        return {
            "type": "rich_items",
            "operation": event.data.get("operation", "upsert"),
            "items": list(event.data.get("items") or []),
        }
    if event.type == "agent_selected":
        payload = {
            "type": "agent_selected",
            "agent": event.agent or event.data.get("agent"),
        }
        reason = event.data.get("reason")
        if reason is not None:
            payload["reason"] = reason
        agent_name = event.data.get("agent_name")
        if agent_name:
            payload["agent_name"] = agent_name
        return payload
    if event.type == "interrupt":
        payload = dict(event.data)
        payload["type"] = "interrupt"
        return payload
    if event.type == "complete":
        payload = dict(event.data)
        payload["type"] = "complete"
        return payload
    if event.type == "state_snapshot":
        legacy_type = event.data.get("legacy_type")
        if legacy_type in {"node_complete", "continuation_start"}:
            payload = {key: value for key, value in event.data.items() if key != "legacy_type"}
            payload["type"] = legacy_type
            if legacy_type == "node_complete" and "node" not in payload and event.node:
                payload["node"] = event.node
            return payload
        return None
    if event.type == "error":
        message = event.data.get("message")
        payload = {
            "type": "error",
            "error": event.data.get("error") or (message if isinstance(message, str) else None),
        }
        if isinstance(message, dict):
            # Streamlit renders the persisted error bot message directly.
            payload["message"] = message
        return payload
    if event.type == "title_updated":
        return {
            "type": "title_updated",
            "title": event.data.get("title"),
            "conversation_id": event.data.get("conversation_id"),
        }
    if event.type == "user_message_created":
        return {"type": "user_message_created", "message": event.data.get("message")}
    if event.type == "heartbeat":
        return {"type": "heartbeat"}
    if event.type in SUBAGENT_PHASE_BY_EVENT:
        payload: dict[str, Any] = {
            "type": "subagent",
            "phase": SUBAGENT_PHASE_BY_EVENT[event.type],
            "subagent": event.subagent.model_dump(mode="json") if event.subagent else None,
        }
        if event.tool_call_id:
            payload["tool_call_id"] = event.tool_call_id
        if event.tool_name:
            payload["tool_name"] = event.tool_name
        for key in (
            "task",
            "output",
            "summary",
            "thinking",
            "status",
            "error",
            "render",
            "text",
            "channel",
            "elapsed_ms",
            "requested_model",
            "resolved_model",
        ):
            value = event.data.get(key)
            if value is not None:
                payload[key] = value
        return payload
    return None
