"""Legacy-dict → canonical V3 coercion for the graph boundary.

The graph projector (``graph_public_projection.py::GraphPublicStreamProjector
.map_event``) still emits legacy public dicts (``token``/``thinking``/
``tool_start``/``complete``/...). The
service layer and public adapters consume canonical :class:`V3StreamEvent`, so
this bridge converts dicts on the way in at the AI-service boundary. Delete
once the graph emits ``V3StreamEvent`` directly.
"""

from __future__ import annotations

from typing import Any, cast

from app.services.event_streaming.events import (
    StreamEventType,
    SubagentRef,
    V3StreamEvent,
    make_event,
)

TOOL_PHASE_START = "start"
TOOL_PHASE_END = "end"

_ERROR_PREFIXES = (
    "error:",
    "error executing tool:",
    "tool not found:",
)


def normalize_tool_phase(value: Any) -> str | None:
    phase = str(value or "").strip().lower()
    if phase in {TOOL_PHASE_START, TOOL_PHASE_END}:
        return phase
    return None


def _looks_like_tool_error(value: Any) -> bool:
    if isinstance(value, dict):
        for key in ("error", "errors", "exception"):
            candidate = value.get(key)
            if candidate not in (None, "", [], {}):
                return True

    if isinstance(value, str):
        stripped = value.strip().lower()
        return any(stripped.startswith(prefix) for prefix in _ERROR_PREFIXES)

    return False


def infer_tool_state(
    *,
    phase: Any,
    result: Any = None,
) -> str:
    normalized_phase = normalize_tool_phase(phase)
    if normalized_phase == TOOL_PHASE_START:
        return "running"
    if normalized_phase != TOOL_PHASE_END:
        return "unknown"
    return "error" if _looks_like_tool_error(result) else "completed"


_DIRECT_EVENT_TYPES: set[str] = {
    "agent_selected",
    "rich_items",
    "interrupt",
    "complete",
    "error",
    "heartbeat",
    "user_message_created",
    "title_updated",
}

_SUBAGENT_EVENT_TYPES: set[str] = {
    "subagent_start",
    "subagent_message_delta",
    "subagent_tool_call_available",
    "subagent_tool_execution_start",
    "subagent_tool_execution_end",
    "subagent_end",
}


def coerce_legacy_event_to_v3(
    event: dict[str, Any] | V3StreamEvent,
    *,
    sequence: int,
) -> V3StreamEvent:
    if isinstance(event, V3StreamEvent):
        return event

    event_type = event.get("type")
    if event_type == "token":
        return make_event(
            "message_delta", sequence=sequence, data={"text": event.get("content", "")}
        )
    if event_type == "thinking":
        return make_event(
            "reasoning_delta", sequence=sequence, data={"text": event.get("content", "")}
        )
    if event_type in {"tool", "tool_start"}:
        phase = event.get("phase")
        if event_type == "tool_start" or phase == "start":
            return make_event(
                "tool_call_available",
                sequence=sequence,
                tool_call_id=event.get("tool_call_id"),
                tool_name=event.get("name"),
                data={"args": event.get("args")},
            )
        return make_event(
            "tool_execution_end",
            sequence=sequence,
            tool_call_id=event.get("tool_call_id"),
            tool_name=event.get("name"),
            data={
                "output": event.get("result"),
                "render": event.get("render"),
                "error": event.get("error"),
            },
        )
    if event_type == "tool_end":
        return make_event(
            "tool_execution_end",
            sequence=sequence,
            tool_call_id=event.get("tool_call_id"),
            tool_name=event.get("name"),
            data={
                "output": event.get("result"),
                "render": event.get("render"),
                "error": event.get("error"),
            },
        )
    if event_type in {"node_complete", "continuation_start"}:
        payload = dict(event)
        payload["legacy_type"] = event_type
        return make_event(
            "state_snapshot",
            sequence=sequence,
            node=event.get("node"),
            data=payload,
        )
    if event_type in _SUBAGENT_EVENT_TYPES:
        subagent_payload = event.get("subagent")
        subagent = (
            SubagentRef.model_validate(subagent_payload)
            if isinstance(subagent_payload, dict)
            else None
        )
        data = event.get("data")
        return make_event(
            cast(StreamEventType, event_type),
            sequence=sequence,
            subagent=subagent,
            tool_call_id=event.get("tool_call_id"),
            tool_name=event.get("tool_name"),
            data=dict(data) if isinstance(data, dict) else {},
        )
    if event_type in _DIRECT_EVENT_TYPES:
        return make_event(cast(StreamEventType, event_type), sequence=sequence, data=dict(event))
    return make_event("state_snapshot", sequence=sequence, data={"raw": dict(event)})
