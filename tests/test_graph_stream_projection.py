"""Unit tests for the graph public-event stream projector.

``GraphPublicStreamProjector`` is constructed directly here (no
``MultiAgentWorkflow`` involved) to pin the canonical v3 curation extracted
from ``app/ai/graph.py``. The projector now emits canonical ``V3StreamEvent``s
(the service layer no longer re-translates dicts). The tool-loop dependency
(``tool_end_events_from_node_state``) is injected as a fake to prove the
projector never reaches back into workflow internals.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from app.services.event_streaming.events import SubagentRef, make_event
from app.services.event_streaming.graph_public_projection import (
    GraphPublicStreamProjector,
    StreamProjectionContext,
)


def _no_tool_end_events(**_kwargs: Any) -> Iterator[dict[str, Any]]:
    return iter(())


def _make_projector(
    *,
    tool_end_events_from_node_state=None,
    suppress_internal_stream_chunks: bool = False,
) -> GraphPublicStreamProjector:
    return GraphPublicStreamProjector(
        tool_end_events_from_node_state=(tool_end_events_from_node_state or _no_tool_end_events),
        suppress_internal_stream_chunks=suppress_internal_stream_chunks,
    )


def test_message_delta_emits_cumulative_delta_only():
    """Gemini-style deltas resend the full text so far; only the new suffix
    should surface as a ``message_delta`` event, mirroring
    ``_consume_stream_text_chunk``."""
    projector = _make_projector()
    ctx = StreamProjectionContext()

    first = list(
        projector.map_event(make_event("message_delta", sequence=1, data={"text": "Hello"}), ctx)
    )
    second = list(
        projector.map_event(
            make_event("message_delta", sequence=2, data={"text": "Hello, world"}), ctx
        )
    )
    repeat = list(
        projector.map_event(
            make_event("message_delta", sequence=3, data={"text": "Hello, world"}), ctx
        )
    )

    assert [(e.type, e.data) for e in first] == [("message_delta", {"text": "Hello"})]
    assert [(e.type, e.data) for e in second] == [("message_delta", {"text": ", world"})]
    assert repeat == []
    assert ctx.accumulated_content == "Hello, world"


def test_tool_call_available_dedupes_by_tool_call_id():
    projector = _make_projector()
    ctx = StreamProjectionContext()
    event = make_event(
        "tool_call_available",
        sequence=1,
        tool_call_id="call-1",
        tool_name="search_documents",
        data={"args": {"query": "x"}},
    )

    first = list(projector.map_event(event, ctx))
    second = list(projector.map_event(event, ctx))

    assert len(first) == 1
    emitted = first[0]
    assert emitted.type == "tool_call_available"
    assert emitted.tool_name == "search_documents"
    assert emitted.tool_call_id == "call-1"
    assert emitted.data == {"args": {"query": "x"}}
    assert second == []


def test_updates_tuple_tool_message_delegates_to_injected_callable():
    """A ``ToolMessage``-terminated node update must route through the
    injected ``tool_end_events_from_node_state`` callable rather than any
    workflow attribute — this is the tool-end → workflow injection seam. The
    injected callable still yields legacy ``tool_end`` dicts; the projector
    converts them to canonical ``tool_execution_end`` events."""
    seen_kwargs: dict[str, Any] = {}

    def fake_tool_end(*, node_state, last_state_values, emitted_tool_result_ids):
        seen_kwargs["node_state"] = node_state
        seen_kwargs["last_state_values"] = last_state_values
        seen_kwargs["emitted_tool_result_ids"] = emitted_tool_result_ids
        yield {
            "type": "tool_end",
            "name": "search_documents",
            "tool_call_id": "call-1",
            "result": "result text",
        }

    projector = _make_projector(tool_end_events_from_node_state=fake_tool_end)
    ctx = StreamProjectionContext()
    node_state = {
        "messages": [
            ToolMessage(content="result text", tool_call_id="call-1", name="search_documents"),
        ],
    }
    event = make_event(
        "state_snapshot",
        sequence=1,
        node="search_agent",
        data={"kind": "updates_tuple", "node_state": node_state},
    )

    events = list(projector.map_event(event, ctx))

    assert len(events) == 1
    emitted = events[0]
    assert emitted.type == "tool_execution_end"
    assert emitted.tool_name == "search_documents"
    assert emitted.tool_call_id == "call-1"
    assert emitted.data == {"output": "result text"}
    assert seen_kwargs["node_state"] is node_state
    assert seen_kwargs["last_state_values"] == node_state
    assert seen_kwargs["emitted_tool_result_ids"] is ctx.emitted_tool_result_ids


def test_values_snapshot_selected_agent_change_emits_second_agent_selected():
    """A handoff surfaces as a second ``agent_selected`` with reason=handoff
    when the v3 ``values`` snapshot's ``selected_agent`` changes."""
    projector = _make_projector()
    ctx = StreamProjectionContext(last_emitted_agent="planning_agent")
    event = make_event(
        "state_snapshot",
        sequence=5,
        data={
            "kind": "values",
            "values": {"selected_agent": "search_agent"},
            "new_messages": [],
            "selected_agent": "search_agent",
        },
    )

    events = list(projector.map_event(event, ctx))

    assert len(events) == 1
    emitted = events[0]
    assert emitted.type == "agent_selected"
    assert emitted.agent == "search_agent"
    assert emitted.data == {"agent": "search_agent", "reason": "handoff"}
    assert ctx.last_emitted_agent == "search_agent"

    # No re-emission once the agent has already been announced.
    repeat = list(projector.map_event(event, ctx))
    assert repeat == []


def test_updates_tuple_planning_agent_tool_calls_emit_node_complete():
    projector = _make_projector()
    ctx = StreamProjectionContext()
    node_state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"id": "call-1", "name": "write_todos", "args": {"todos": []}}],
            )
        ]
    }
    event = make_event(
        "state_snapshot",
        sequence=1,
        node="planning_agent",
        data={"kind": "updates_tuple", "node_state": node_state},
    )

    events = list(projector.map_event(event, ctx))

    assert len(events) == 2
    node_complete, tool_start = events

    # node_complete is carried as a state_snapshot with a legacy_type discriminator.
    assert node_complete.type == "state_snapshot"
    assert node_complete.node == "planning_agent"
    assert node_complete.data == {
        "node": "planning_agent",
        "tool_calls": [{"name": "write_todos", "id": "call-1", "args": {"todos": []}}],
        "legacy_type": "node_complete",
    }

    assert tool_start.type == "tool_call_available"
    assert tool_start.tool_name == "write_todos"
    assert tool_start.tool_call_id == "call-1"
    assert tool_start.data == {"args": {"todos": []}}


def test_planning_node_complete_also_derived_from_values_snapshot():
    """The v3 path has no per-node ``updates`` channel, so ``node_complete``
    for planning is best-effort derived from newly added messages in a
    ``values`` snapshot instead."""
    projector = _make_projector()
    ctx = StreamProjectionContext(last_emitted_agent="planning_agent")
    planning_ai_message = AIMessage(
        content="",
        tool_calls=[{"id": "call-2", "name": "write_todos", "args": {"todos": ["a"]}}],
    )
    event = make_event(
        "state_snapshot",
        sequence=2,
        data={
            "kind": "values",
            "values": {"selected_agent": "planning_agent"},
            "new_messages": [planning_ai_message],
            "selected_agent": "planning_agent",
        },
    )

    events = list(projector.map_event(event, ctx))

    node_complete = next(
        e
        for e in events
        if e.type == "state_snapshot" and e.data.get("legacy_type") == "node_complete"
    )
    assert node_complete.node == "planning_agent"
    assert node_complete.data == {
        "node": "planning_agent",
        "tool_calls": [{"name": "write_todos", "id": "call-2", "args": {"todos": ["a"]}}],
        "legacy_type": "node_complete",
    }
    tool_start_events = [e for e in events if e.type == "tool_call_available"]
    assert len(tool_start_events) == 1
    tool_start = tool_start_events[0]
    assert tool_start.tool_name == "write_todos"
    assert tool_start.tool_call_id == "call-2"
    assert tool_start.data == {"args": {"todos": ["a"]}}


def test_subagent_events_pass_through_unchanged():
    projector = _make_projector()
    ctx = StreamProjectionContext()
    subagent = SubagentRef(
        id="worker-a",
        name="search_agent",
        path=["planning_agent", "worker-a"],
        status="running",
    )
    event = make_event(
        "subagent_start",
        sequence=1,
        subagent=subagent,
        tool_call_id=None,
        tool_name=None,
        data={"task": "look this up"},
    )

    events = list(projector.map_event(event, ctx))

    assert len(events) == 1
    emitted = events[0]
    assert emitted.type == "subagent_start"
    assert emitted.data == {"task": "look this up"}
    assert emitted.subagent == subagent
    assert emitted.subagent.model_dump(mode="json") == {
        "id": "worker-a",
        "name": "search_agent",
        "path": ["planning_agent", "worker-a"],
        "status": "running",
    }


def test_suppress_internal_stream_chunks_drops_internal_message_chunk():
    """Internal (tagged) message chunks are dropped only when the injected
    ``suppress_internal_stream_chunks`` flag is on. Routed through the public
    ``map_event`` dispatcher (v1/v2 tuple fallback path: ``messages_tuple``)."""
    from types import SimpleNamespace

    chunk = SimpleNamespace(content="internal text", content_blocks=None)
    metadata = {"tags": ["internal"]}
    event = make_event(
        "state_snapshot",
        sequence=1,
        data={"kind": "messages_tuple", "chunk": chunk, "metadata": metadata},
    )

    suppressed_events = list(
        _make_projector(suppress_internal_stream_chunks=True).map_event(
            event, StreamProjectionContext()
        )
    )
    passthrough_events = list(
        _make_projector(suppress_internal_stream_chunks=False).map_event(
            event, StreamProjectionContext()
        )
    )

    assert suppressed_events == []
    assert [(e.type, e.data) for e in passthrough_events] == [
        ("message_delta", {"text": "internal text"})
    ]
