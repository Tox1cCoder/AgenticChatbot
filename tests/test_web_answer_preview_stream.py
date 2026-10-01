"""Web answer drafts reach the UI without becoming authoritative answer text."""

import json

import pytest
from langchain_core.messages import ToolMessage

from app.services.event_streaming.ai_sdk_v6 import AISDKV6StreamAdapter, AISDKV6StreamState
from app.services.event_streaming.events import make_event
from app.services.event_streaming.graph_public_projection import (
    GraphPublicStreamProjector,
    StreamProjectionContext,
)


def _projector():
    return GraphPublicStreamProjector(
        tool_end_events_from_node_state=lambda **kwargs: iter(()),
        suppress_internal_stream_chunks=True,
    )


def test_web_answer_chunks_are_previewed_and_a_repair_replaces_the_draft():
    projector = _projector()
    context = StreamProjectionContext()
    raw = [
        make_event("tool_execution_end", sequence=1, tool_name="web_search", data={"output": {}}),
        make_event("message_delta", sequence=2, run_id="draft", data={"text": "First "}),
        make_event("message_delta", sequence=3, run_id="draft", data={"text": "draft"}),
        make_event("message_delta", sequence=4, run_id="repair", data={"text": "Cited "}),
        make_event("message_delta", sequence=5, run_id="repair", data={"text": "answer"}),
    ]

    public = [event for item in raw for event in projector.map_event(item, context)]
    previews = [event.data for event in public if event.type == "answer_preview"]

    assert previews == [
        {"operation": "replace", "text": "First "},
        {"operation": "append", "text": "draft"},
        {"operation": "replace", "text": "Cited "},
        {"operation": "append", "text": "answer"},
    ]
    assert all(event.type != "message_delta" for event in public)


def test_repair_with_same_prefix_replaces_the_whole_draft():
    projector = _projector()
    context = StreamProjectionContext()
    raw = [
        make_event("tool_execution_end", sequence=1, tool_name="web_search", data={"output": {}}),
        make_event("message_delta", sequence=2, run_id="draft", data={"text": "Hello"}),
        make_event("message_delta", sequence=3, run_id="repair", data={"text": "Hello there"}),
    ]
    public = [event for item in raw for event in projector.map_event(item, context)]
    assert [event.data for event in public if event.type == "answer_preview"] == [
        {"operation": "replace", "text": "Hello"},
        {"operation": "replace", "text": "Hello there"},
    ]


def test_web_answer_does_not_preview_before_search_completes():
    projector = _projector()
    context = StreamProjectionContext(requires_web=True)
    public = list(
        projector.map_event(
            make_event("message_delta", sequence=1, run_id="draft", data={"text": "Internal"}),
            context,
        )
    )
    assert public == []


def test_legacy_web_tool_completion_enables_answer_preview():
    projector = GraphPublicStreamProjector(
        tool_end_events_from_node_state=lambda **kwargs: iter(
            ({"tool_call_id": "web-1", "name": "web_search", "result": "Results"},)
        ),
        suppress_internal_stream_chunks=True,
    )
    context = StreamProjectionContext(requires_web=True)
    update = make_event(
        "state_snapshot",
        sequence=1,
        data={
            "kind": "updates_tuple",
            "node_state": {"messages": [ToolMessage(content="Results", tool_call_id="web-1")]},
        },
    )
    list(projector.map_event(update, context))
    public = list(
        projector.map_event(
            make_event("message_delta", sequence=2, run_id="answer", data={"text": "Hello"}),
            context,
        )
    )
    assert [event.data for event in public if event.type == "answer_preview"] == [
        {"operation": "replace", "text": "Hello"}
    ]


@pytest.mark.asyncio
async def test_ai_sdk_preview_is_transient_and_final_answer_remains_authoritative():
    async def source():
        yield make_event(
            "answer_preview", sequence=1, data={"operation": "replace", "text": "Uncited draft"}
        )
        yield make_event("message_delta", sequence=2, data={"text": "Grounded final answer"})
        yield make_event("complete", sequence=3, data={"message": {"id": "m-1"}})

    adapter = AISDKV6StreamAdapter(
        source,
        AISDKV6StreamState(message_id="m-1", text_id="t-1", reasoning_id="r-1"),
    )
    payloads = [
        json.loads(line[6:])
        for chunk in [part async for part in adapter.iter_sse()]
        for line in chunk.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]

    previews = [part for part in payloads if part.get("type") == "data-answer-preview"]
    final_text = [part["delta"] for part in payloads if part.get("type") == "text-delta"]
    assert previews == [
        {
            "type": "data-answer-preview",
            "data": {"operation": "replace", "text": "Uncited draft"},
            "transient": True,
        },
        {"type": "data-answer-preview", "data": {"operation": "clear"}, "transient": True},
    ]
    assert final_text == ["Grounded final answer"]


@pytest.mark.asyncio
async def test_ai_sdk_honors_explicit_preview_clear():
    async def source():
        yield make_event(
            "answer_preview", sequence=1, data={"operation": "replace", "text": "Draft"}
        )
        yield make_event("answer_preview", sequence=2, data={"operation": "clear"})
        yield make_event(
            "answer_preview", sequence=3, data={"operation": "replace", "text": "New draft"}
        )
        yield make_event("complete", sequence=4, data={"message": {"id": "m-1"}})

    adapter = AISDKV6StreamAdapter(
        source,
        AISDKV6StreamState(message_id="m-1", text_id="t-1", reasoning_id="r-1"),
    )
    payloads = [
        json.loads(line[6:])
        for chunk in [part async for part in adapter.iter_sse()]
        for line in chunk.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert [
        part.get("data", {}).get("operation")
        for part in payloads
        if part.get("type") == "data-answer-preview"
    ] == ["replace", "clear", "replace", "clear"]


@pytest.mark.asyncio
async def test_ai_sdk_clears_preview_when_stream_source_fails():
    async def source():
        yield make_event(
            "answer_preview", sequence=1, data={"operation": "replace", "text": "Draft"}
        )
        raise RuntimeError("provider failure")

    adapter = AISDKV6StreamAdapter(
        source,
        AISDKV6StreamState(message_id="m-1", text_id="t-1", reasoning_id="r-1"),
    )
    payloads = [
        json.loads(line[6:])
        for chunk in [part async for part in adapter.iter_sse()]
        for line in chunk.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]

    assert [
        part.get("data", {}).get("operation")
        for part in payloads
        if part.get("type") == "data-answer-preview"
    ] == ["replace", "clear"]


@pytest.mark.asyncio
async def test_opted_in_web_answer_streams_text_deltas_and_reconciles_final_message():
    async def source():
        yield make_event(
            "answer_preview", sequence=1, data={"operation": "replace", "text": "Draft "}
        )
        yield make_event(
            "answer_preview", sequence=2, data={"operation": "append", "text": "answer"}
        )
        yield make_event("message_delta", sequence=3, data={"text": "Grounded final answer"})
        yield make_event(
            "complete",
            sequence=4,
            data={"message": {"id": "m-1", "content": "Grounded final answer"}},
        )

    adapter = AISDKV6StreamAdapter(
        source,
        AISDKV6StreamState(
            message_id="m-1",
            text_id="t-1",
            reasoning_id="r-1",
            stream_web_answer_v1=True,
        ),
    )
    payloads = [
        json.loads(line[6:])
        for chunk in [part async for part in adapter.iter_sse()]
        for line in chunk.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]

    assert [part["delta"] for part in payloads if part.get("type") == "text-delta"] == [
        "Draft ",
        "answer",
    ]
    assert [part["data"] for part in payloads if part.get("type") == "data-answer-reconcile"] == [
        {"messageId": "m-1", "content": "Grounded final answer"}
    ]
    types = [part.get("type") for part in payloads]
    assert types.index("text-delta") < types.index("data-answer-reconcile") < types.index("finish")


@pytest.mark.asyncio
async def test_opted_in_web_answer_repair_is_sent_as_replacement_draft():
    async def source():
        yield make_event(
            "answer_preview", sequence=1, data={"operation": "replace", "text": "Uncited"}
        )
        yield make_event(
            "answer_preview", sequence=2, data={"operation": "replace", "text": "Cited "}
        )
        yield make_event(
            "answer_preview", sequence=3, data={"operation": "append", "text": "reply"}
        )
        yield make_event("message_delta", sequence=4, data={"text": "Cited [1](https://a.test)"})
        yield make_event(
            "complete",
            sequence=5,
            data={"message": {"id": "m-1", "content": "Cited [1](https://a.test)"}},
        )

    adapter = AISDKV6StreamAdapter(
        source,
        AISDKV6StreamState(
            message_id="m-1",
            text_id="t-1",
            reasoning_id="r-1",
            stream_web_answer_v1=True,
        ),
    )
    payloads = [
        json.loads(line[6:])
        for chunk in [part async for part in adapter.iter_sse()]
        for line in chunk.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]

    assert [part["delta"] for part in payloads if part.get("type") == "text-delta"] == [
        "Uncited"
    ]
    assert [part["data"] for part in payloads if part.get("type") == "data-answer-preview"] == [
        {"operation": "replace", "text": "Cited "},
        {"operation": "append", "text": "reply"},
    ]
    assert [
        part["data"]["content"]
        for part in payloads
        if part.get("type") == "data-answer-reconcile"
    ] == [
        "Cited [1](https://a.test)"
    ]


@pytest.mark.asyncio
async def test_opted_in_web_answer_error_clears_streamed_draft():
    async def source():
        yield make_event(
            "answer_preview", sequence=1, data={"operation": "replace", "text": "Draft"}
        )
        yield make_event("error", sequence=2, data={"error": "Generation failed"})

    adapter = AISDKV6StreamAdapter(
        source,
        AISDKV6StreamState(
            message_id="m-1",
            text_id="t-1",
            reasoning_id="r-1",
            stream_web_answer_v1=True,
        ),
    )
    payloads = [
        json.loads(line[6:])
        for chunk in [part async for part in adapter.iter_sse()]
        for line in chunk.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]

    assert [part["data"] for part in payloads if part.get("type") == "data-answer-reconcile"] == [
        {"messageId": "m-1", "content": ""}
    ]
