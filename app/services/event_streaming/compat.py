"""Temporary legacy-dict → canonical V3 coercion used during migration.

Producers (the graph mapper, message_service) still emit legacy public dicts
(``token``/``thinking``/``tool``/``complete``/...). The public adapters consume
canonical :class:`V3StreamEvent`, so this bridge converts dicts on the way in.
Delete once all producers emit ``V3StreamEvent`` directly (plan Task 9).
"""

from __future__ import annotations

from typing import Any, cast

from app.services.event_streaming.events import (
    StreamEventType,
    V3StreamEvent,
    make_event,
)

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


def coerce_legacy_event_to_v3(
    event: dict[str, Any] | V3StreamEvent,
    *,
    sequence: int,
) -> V3StreamEvent:
    if isinstance(event, V3StreamEvent):
        return event

    event_type = event.get("type")
    if event_type == "token":
        return make_event("message_delta", sequence=sequence, data={"text": event.get("content", "")})
    if event_type == "thinking":
        return make_event("reasoning_delta", sequence=sequence, data={"text": event.get("content", "")})
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
    if event_type in _DIRECT_EVENT_TYPES:
        return make_event(cast(StreamEventType, event_type), sequence=sequence, data=dict(event))
    return make_event("state_snapshot", sequence=sequence, data={"raw": dict(event)})
