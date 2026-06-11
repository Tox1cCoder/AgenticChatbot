"""Canonical internal v3 stream event model.

This is the single internal streaming contract for the backend. It is
independent of LangChain's upstream ``astream_events(version="v3")`` API —
"v3" here names *our* schema. Public adapters (AI SDK v6, Streamlit internal
SSE) consume these events and project them to their respective wire formats.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

StreamSchemaVersion = Literal["v3"]

StreamEventType = Literal[
    "run_start",
    "message_start",
    "message_delta",
    "message_end",
    "reasoning_start",
    "reasoning_delta",
    "reasoning_end",
    "tool_call_delta",
    "tool_call_available",
    "tool_execution_start",
    "tool_execution_delta",
    "tool_execution_end",
    "agent_selected",
    "subagent_start",
    "subagent_message_delta",
    "subagent_tool_call_available",
    "subagent_tool_execution_start",
    "subagent_tool_execution_end",
    "subagent_end",
    "state_snapshot",
    "rich_items",
    "interrupt",
    "complete",
    "error",
    "heartbeat",
    "user_message_created",
    "title_updated",
]


class SubagentRef(BaseModel):
    id: str
    name: str
    path: list[str] = Field(default_factory=list)
    status: Literal["running", "completed", "failed", "timeout", "requires_approval"]


class V3StreamEvent(BaseModel):
    schema_version: StreamSchemaVersion = "v3"
    type: StreamEventType
    sequence: int
    run_id: str | None = None
    conversation_id: str | None = None
    message_id: str | None = None
    agent: str | None = None
    node: str | None = None
    namespace: list[str] = Field(default_factory=list)
    subagent: SubagentRef | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


def make_event(
    event_type: StreamEventType,
    *,
    sequence: int,
    run_id: str | None = None,
    conversation_id: str | None = None,
    message_id: str | None = None,
    agent: str | None = None,
    node: str | None = None,
    namespace: list[str] | None = None,
    subagent: SubagentRef | None = None,
    tool_call_id: str | None = None,
    tool_name: str | None = None,
    data: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> V3StreamEvent:
    return V3StreamEvent(
        type=event_type,
        sequence=sequence,
        run_id=run_id,
        conversation_id=conversation_id,
        message_id=message_id,
        agent=agent,
        node=node,
        namespace=list(namespace or []),
        subagent=subagent,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        data=dict(data or {}),
        metadata=dict(metadata or {}),
    )


# Shared by both public adapters (AI SDK `data-subagent`, Streamlit `subagent`)
# so the wire `phase` discriminator stays consistent across protocols.
SUBAGENT_PHASE_BY_EVENT: dict[str, str] = {
    "subagent_start": "start",
    "subagent_end": "end",
    "subagent_tool_call_available": "tool",
    "subagent_tool_execution_start": "tool",
    "subagent_tool_execution_end": "tool",
    "subagent_message_delta": "delta",
}
