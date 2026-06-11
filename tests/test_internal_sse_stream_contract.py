from __future__ import annotations

from app.services.event_streaming.events import SubagentRef, make_event
from app.services.event_streaming.internal_sse import legacy_event_from_v3


def test_message_delta_maps_to_streamlit_token_event():
    event = make_event("message_delta", sequence=1, data={"text": "hello"})

    legacy = legacy_event_from_v3(event)

    assert legacy == {"type": "token", "content": "hello"}


def test_reasoning_delta_maps_to_streamlit_thinking_event():
    event = make_event("reasoning_delta", sequence=1, data={"text": "plan"})

    legacy = legacy_event_from_v3(event)

    assert legacy == {"type": "thinking", "content": "plan"}


def test_tool_call_available_maps_to_single_legacy_tool_start():
    event = make_event(
        "tool_call_available",
        sequence=1,
        tool_call_id="call-1",
        tool_name="search_documents",
        data={"args": {"query": "x"}},
    )

    legacy = legacy_event_from_v3(event)

    assert legacy["type"] == "tool"
    assert legacy["phase"] == "start"
    assert legacy["state"] == "queued"
    assert legacy["name"] == "search_documents"
    assert legacy["args"] == {"query": "x"}


def test_tool_execution_end_maps_to_legacy_tool_end():
    event = make_event(
        "tool_execution_end",
        sequence=1,
        tool_call_id="call-1",
        tool_name="search_documents",
        data={"output": "result", "render": {"type": "text", "text": "result"}},
    )

    legacy = legacy_event_from_v3(event)

    assert legacy["type"] == "tool"
    assert legacy["phase"] == "end"
    assert legacy["state"] == "completed"
    assert legacy["result"] == "result"
    assert legacy["render"]["type"] == "text"


def test_title_update_is_non_terminal():
    event = make_event(
        "title_updated",
        sequence=1,
        data={"title": "New title", "conversation_id": "conv-1"},
    )

    legacy = legacy_event_from_v3(event)

    assert legacy == {
        "type": "title_updated",
        "title": "New title",
        "conversation_id": "conv-1",
    }


def test_state_snapshot_preserves_legacy_node_complete_for_streamlit():
    event = make_event(
        "state_snapshot",
        sequence=1,
        node="planning_agent",
        data={
            "legacy_type": "node_complete",
            "node": "planning_agent",
            "tool_calls": [{"name": "dispatch_subagents", "id": "call-1", "args": {}}],
        },
    )

    legacy = legacy_event_from_v3(event)

    assert legacy == {
        "type": "node_complete",
        "node": "planning_agent",
        "tool_calls": [{"name": "dispatch_subagents", "id": "call-1", "args": {}}],
    }


def test_state_snapshot_preserves_legacy_continuation_marker():
    event = make_event(
        "state_snapshot",
        sequence=1,
        data={
            "legacy_type": "continuation_start",
            "round": 2,
            "max_rounds": 4,
            "reason": "recursion_limit",
        },
    )

    legacy = legacy_event_from_v3(event)

    assert legacy == {
        "type": "continuation_start",
        "round": 2,
        "max_rounds": 4,
        "reason": "recursion_limit",
    }


def test_subagent_start_projects_to_subagent_event():
    event = make_event(
        "subagent_start",
        sequence=1,
        subagent=SubagentRef(
            id="w1", name="search_agent", path=["planning_agent", "w1"], status="running"
        ),
        data={"task": "Find sources"},
    )
    payload = legacy_event_from_v3(event)
    assert payload["type"] == "subagent"
    assert payload["phase"] == "start"
    assert payload["subagent"]["id"] == "w1"
    assert payload["task"] == "Find sources"


def test_subagent_end_projects_status_and_summary():
    event = make_event(
        "subagent_end",
        sequence=2,
        subagent=SubagentRef(
            id="w1", name="search_agent", path=["planning_agent", "w1"], status="completed"
        ),
        data={"summary": "done", "elapsed_ms": 12},
    )
    payload = legacy_event_from_v3(event)
    assert payload["phase"] == "end"
    assert payload["subagent"]["status"] == "completed"
    assert payload["summary"] == "done"
    assert payload["elapsed_ms"] == 12
