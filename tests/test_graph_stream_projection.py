"""Unit tests for the graph public-event stream projector.

``GraphPublicStreamProjector`` is constructed directly here (no
``MultiAgentWorkflow`` involved) to pin the canonical v3 → legacy public dict
projection extracted from ``app/ai/graph.py``. The tool-loop dependency
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
    should surface as a ``token`` event, mirroring ``_consume_stream_text_chunk``."""
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

    assert first == [{"type": "token", "content": "Hello"}]
    assert second == [{"type": "token", "content": ", world"}]
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

    assert first == [
        {
            "type": "tool_start",
            "name": "search_documents",
            "tool_call_id": "call-1",
            "args": {"query": "x"},
        }
    ]
    assert second == []


def test_updates_tuple_tool_message_delegates_to_injected_callable():
    """A ``ToolMessage``-terminated node update must route through the
    injected ``tool_end_events_from_node_state`` callable rather than any
    workflow attribute — this is the tool-end → workflow injection seam."""
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

    assert events == [
        {
            "type": "tool_end",
            "name": "search_documents",
            "tool_call_id": "call-1",
            "result": "result text",
        }
    ]
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

    assert events == [{"type": "agent_selected", "agent": "search_agent", "reason": "handoff"}]
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

    assert events == [
        {
            "type": "node_complete",
            "node": "planning_agent",
            "tool_calls": [{"name": "write_todos", "id": "call-1", "args": {"todos": []}}],
        },
        {
            "type": "tool_start",
            "name": "write_todos",
            "tool_call_id": "call-1",
            "args": {"todos": []},
        },
    ]


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

    node_complete = next(e for e in events if e["type"] == "node_complete")
    assert node_complete == {
        "type": "node_complete",
        "node": "planning_agent",
        "tool_calls": [{"name": "write_todos", "id": "call-2", "args": {"todos": ["a"]}}],
    }
    tool_start_events = [e for e in events if e["type"] == "tool_start"]
    assert tool_start_events == [
        {
            "type": "tool_start",
            "name": "write_todos",
            "tool_call_id": "call-2",
            "args": {"todos": ["a"]},
        }
    ]


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

    assert events == [
        {
            "type": "subagent_start",
            "data": {"task": "look this up"},
            "subagent": {
                "id": "worker-a",
                "name": "search_agent",
                "path": ["planning_agent", "worker-a"],
                "status": "running",
            },
        }
    ]


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
    assert passthrough_events == [{"type": "token", "content": "internal text"}]
