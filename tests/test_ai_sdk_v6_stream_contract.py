from __future__ import annotations

import json

import pytest

from app.services.event_streaming.ai_sdk_v6 import AISDKV6StreamAdapter, AISDKV6StreamState
from app.services.event_streaming.events import make_event


async def _collect_payloads(source):
    adapter = AISDKV6StreamAdapter(
        lambda: source(),
        AISDKV6StreamState(message_id="m-1", text_id="t-1", reasoning_id="r-1"),
    )
    chunks = [chunk async for chunk in adapter.iter_sse()]
    payloads = []
    for line in "".join(chunks).splitlines():
        if not line.startswith("data: "):
            continue
        raw = line[6:]
        payloads.append(raw if raw == "[DONE]" else json.loads(raw))
    return payloads


@pytest.mark.asyncio
async def test_text_stream_maps_to_ui_message_chunks():
    async def source():
        yield make_event("message_delta", sequence=1, data={"text": "hel"})
        yield make_event("message_delta", sequence=2, data={"text": "lo"})
        yield make_event("complete", sequence=3, data={"message": {"id": "m-1"}})

    payloads = await _collect_payloads(source)

    assert [payload["type"] for payload in payloads[:-1]] == [
        "start",
        "start-step",
        "text-start",
        "text-delta",
        "text-delta",
        "text-end",
        "finish-step",
        "finish",
    ]
    assert payloads[3]["delta"] == "hel"
    assert payloads[4]["delta"] == "lo"
    assert payloads[-1] == "[DONE]"


@pytest.mark.asyncio
async def test_reasoning_stream_maps_to_ai_sdk_reasoning_chunks():
    async def source():
        yield make_event("reasoning_delta", sequence=1, data={"text": "plan"})
        yield make_event("complete", sequence=2, data={"message": {"id": "m-1"}})

    payloads = await _collect_payloads(source)
    types = [payload["type"] for payload in payloads if payload != "[DONE]"]

    assert "reasoning-start" in types
    assert "reasoning-delta" in types
    assert "reasoning-end" in types


@pytest.mark.asyncio
async def test_tool_call_and_tool_output_map_to_ai_sdk_tool_chunks():
    async def source():
        yield make_event(
            "tool_call_available",
            sequence=1,
            tool_call_id="call-1",
            tool_name="search_documents",
            data={"args": {"query": "x"}},
        )
        yield make_event(
            "tool_execution_end",
            sequence=2,
            tool_call_id="call-1",
            tool_name="search_documents",
            data={"output": "result", "render": {"type": "text", "text": "result"}},
        )
        yield make_event("complete", sequence=3, data={"message": {"id": "m-1"}})

    payloads = await _collect_payloads(source)
    tool_payloads = [
        payload
        for payload in payloads
        if payload != "[DONE]" and str(payload.get("type", "")).startswith("tool-")
    ]

    assert [payload["type"] for payload in tool_payloads] == [
        "tool-input-start",
        "tool-input-available",
        "tool-output-available",
    ]
    assert tool_payloads[1]["input"] == {"query": "x"}
    assert tool_payloads[2]["output"] == "result"
    assert tool_payloads[2]["render"]["type"] == "text"


@pytest.mark.asyncio
async def test_interrupt_terminates_ui_stream():
    async def source():
        yield make_event(
            "interrupt",
            sequence=1,
            data={
                "thread_id": "thread-1",
                "pending_tool_calls": [],
                "interrupt": {"interrupt_id": "int-1"},
                "message": "Approval needed",
            },
        )
        yield make_event("message_delta", sequence=2, data={"text": "unreachable"})

    payloads = await _collect_payloads(source)

    assert any(
        payload != "[DONE]" and payload.get("type") == "data-interrupt" for payload in payloads
    )
    assert payloads[-1] == "[DONE]"
    assert not any(
        payload != "[DONE]" and payload.get("delta") == "unreachable" for payload in payloads
    )
