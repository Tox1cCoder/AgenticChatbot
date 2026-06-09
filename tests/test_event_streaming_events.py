from __future__ import annotations

from app.services.event_streaming.events import (
    SubagentRef,
    V3StreamEvent,
    make_event,
)


def test_make_event_assigns_schema_and_sequence():
    event = make_event(
        "message_delta",
        sequence=7,
        agent="chat_agent",
        node="chat_agent",
        data={"text": "hello"},
    )

    assert isinstance(event, V3StreamEvent)
    assert event.schema_version == "v3"
    assert event.type == "message_delta"
    assert event.sequence == 7
    assert event.agent == "chat_agent"
    assert event.data == {"text": "hello"}


def test_subagent_ref_serializes_deepagents_shape():
    event = make_event(
        "subagent_start",
        sequence=1,
        subagent=SubagentRef(
            id="worker-a",
            name="search_agent",
            path=["planning_agent", "worker-a"],
            status="running",
        ),
    )

    payload = event.model_dump(mode="json")

    assert payload["schema_version"] == "v3"
    assert payload["subagent"]["id"] == "worker-a"
    assert payload["subagent"]["name"] == "search_agent"
    assert payload["subagent"]["path"] == ["planning_agent", "worker-a"]
    assert payload["subagent"]["status"] == "running"
