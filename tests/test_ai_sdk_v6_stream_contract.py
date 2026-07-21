from __future__ import annotations

import json

import pytest

from app.api.ai_sdk import _extract_user_attachments, _has_user_attachment_candidates
from app.core.exceptions import CustomHTTPException
from app.services.event_streaming.ai_sdk_v6 import AISDKV6StreamAdapter, AISDKV6StreamState
from app.services.event_streaming.events import SubagentRef, make_event


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


@pytest.mark.asyncio
async def test_v3_error_retains_canonical_status_and_code_in_ai_sdk_stream():
    async def source():
        yield make_event(
            "error",
            sequence=1,
            data={
                "error": "This interrupt has already been resolved.",
                "status_code": 409,
                "error_code": "INTERRUPT_ALREADY_RESOLVED",
            },
        )

    payloads = await _collect_payloads(source)

    assert next(
        payload for payload in payloads if payload != "[DONE]" and payload["type"] == "error"
    ) == {
        "type": "error",
        "errorText": "This interrupt has already been resolved.",
        "statusCode": 409,
        "errorCode": "INTERRUPT_ALREADY_RESOLVED",
    }
    assert payloads[-1] == "[DONE]"


@pytest.mark.asyncio
async def test_ai_sdk_stream_exception_retains_custom_http_metadata():
    async def source():
        raise CustomHTTPException(409, "Already resolved", "INTERRUPT_ALREADY_RESOLVED")
        yield  # pragma: no cover

    payloads = await _collect_payloads(source)

    assert next(
        payload for payload in payloads if payload != "[DONE]" and payload["type"] == "error"
    ) == {
        "type": "error",
        "errorText": "Already resolved",
        "statusCode": 409,
        "errorCode": "INTERRUPT_ALREADY_RESOLVED",
    }
    assert payloads[-1] == "[DONE]"


@pytest.mark.asyncio
async def test_interrupt_projects_paused_message_metadata_shape():
    async def source():
        yield make_event(
            "interrupt",
            sequence=1,
            data={
                "thread_id": "thread-1",
                "pending_tool_calls": [],
                "interrupt": {"interrupt_id": "int-1"},
                "message": {
                    "id": "paused-1",
                    "content": "",
                    "message_metadata": {"paused": True, "pause_reason": "tool_approval_required"},
                },
            },
        )

    payloads = await _collect_payloads(source)
    interrupt_payload = next(
        payload
        for payload in payloads
        if payload != "[DONE]" and payload["type"] == "data-interrupt"
    )
    message = interrupt_payload["data"]["message"]

    assert message["id"] == "paused-1"
    assert message["metadata"] == {"paused": True, "pause_reason": "tool_approval_required"}
    assert "messageMetadata" not in message
    assert "message_metadata" not in message
    assert "pendingToolCalls" not in interrupt_payload["data"]


@pytest.mark.asyncio
async def test_subagent_events_map_to_data_subagent_chunks():
    async def source():
        yield make_event(
            "subagent_start",
            sequence=1,
            subagent=SubagentRef(
                id="w1", name="search_agent", path=["planning_agent", "w1"], status="running"
            ),
            data={"task": "Find sources"},
        )
        yield make_event(
            "subagent_tool_execution_end",
            sequence=2,
            subagent=SubagentRef(
                id="w1", name="search_agent", path=["planning_agent", "w1"], status="running"
            ),
            tool_call_id="sub-call-1",
            tool_name="search_documents",
            data={"output": "hit", "status": "success"},
        )
        yield make_event(
            "subagent_end",
            sequence=3,
            subagent=SubagentRef(
                id="w1", name="search_agent", path=["planning_agent", "w1"], status="completed"
            ),
            data={"output": "answer", "summary": "done", "elapsed_ms": 12},
        )
        yield make_event("complete", sequence=4, data={"message": {"id": "m-1"}})

    payloads = await _collect_payloads(source)
    subagent = [p for p in payloads if p != "[DONE]" and p.get("type") == "data-subagent"]

    assert [p["data"]["phase"] for p in subagent] == ["start", "tool", "end"]
    assert all(p["transient"] is True for p in subagent)
    assert subagent[0]["data"]["subagent"]["id"] == "w1"
    assert subagent[1]["data"]["toolName"] == "search_documents"
    assert subagent[2]["data"]["subagent"]["status"] == "completed"
    assert subagent[2]["data"]["elapsedMs"] == 12


@pytest.mark.asyncio
async def test_subagent_message_delta_maps_to_delta_phase_with_thinking_channel():
    async def source():
        yield make_event(
            "subagent_message_delta",
            sequence=1,
            subagent=SubagentRef(
                id="w1", name="search_agent", path=["planning_agent", "w1"], status="running"
            ),
            data={"text": "weighing sources", "channel": "reasoning"},
        )
        yield make_event(
            "subagent_end",
            sequence=2,
            subagent=SubagentRef(
                id="w1", name="search_agent", path=["planning_agent", "w1"], status="completed"
            ),
            data={"summary": "done", "thinking": "weighed sources", "elapsed_ms": 5},
        )
        yield make_event("complete", sequence=3, data={"message": {"id": "m-1"}})

    payloads = await _collect_payloads(source)
    subagent = [p for p in payloads if p != "[DONE]" and p.get("type") == "data-subagent"]

    assert subagent[0]["data"]["phase"] == "delta"
    assert subagent[0]["data"]["text"] == "weighing sources"
    assert subagent[0]["data"]["channel"] == "reasoning"
    assert subagent[1]["data"]["phase"] == "end"
    assert subagent[1]["data"]["thinking"] == "weighed sources"


@pytest.mark.asyncio
async def test_terminal_assistant_message_projects_to_ui_message_metadata_shape():
    async def source():
        yield make_event(
            "complete",
            sequence=1,
            data={
                "message": {
                    "id": "m-1",
                    "sender": 2,
                    "conversation_id": "conversation-1",
                    "content": "Answer",
                    "created_at": "2026-06-22T01:00:00Z",
                    "updated_at": "2026-06-22T01:00:01Z",
                    "message_metadata": {"context_window": {"display_state": "ok"}},
                }
            },
        )

    payloads = await _collect_payloads(source)
    assistant_payload = next(
        payload
        for payload in payloads
        if payload != "[DONE]" and payload.get("type") == "data-assistant-message"
    )
    message = assistant_payload["data"]["message"]

    assert message["id"] == "m-1"
    assert message["role"] == "assistant"
    assert message["createdAt"] == "2026-06-22T01:00:00Z"
    assert message["metadata"] == {"context_window": {"display_state": "ok"}}
    assert "messageMetadata" not in message
    assert "message_metadata" not in message
    assert "content" not in message
    assert "sender" not in message
    assert "conversation_id" not in message
    assert "updated_at" not in message


@pytest.mark.asyncio
async def test_terminal_assistant_message_preserves_additive_context_usage_without_new_event():
    context_window = {
        "provider": "provider-a",
        "model": "model-a",
        "context_window_tokens": 128000,
        "max_input_tokens": 128000,
        "max_output_tokens": 16384,
        "limit_type": "shared_context",
        "source": "provider_api",
        "known": True,
        "input_tokens": 1200,
        "output_tokens": 300,
        "total_tokens": 1500,
        "usage_source": "provider_reported",
        "used_tokens": 1500,
        "used_token_source": "provider_reported_total",
        "input_usage_ratio": 0.009375,
        "output_usage_ratio": 0.00234375,
        "usage_ratio": 0.01171875,
        "usage_ratio_basis": "shared_context_total",
        "display_state": "ok",
        "future_limit_field": "ignored-by-old-clients",
    }

    async def source():
        yield make_event("message_delta", sequence=1, data={"text": "Done"})
        yield make_event(
            "complete",
            sequence=2,
            data={
                "message": {
                    "id": "m-1",
                    "sender": 2,
                    "content": "Done",
                    "message_metadata": {
                        "context_window": context_window,
                        "futureMetadata": True,
                    },
                }
            },
        )

    payloads = await _collect_payloads(source)
    wire_payloads = [payload for payload in payloads if payload != "[DONE]"]
    terminal = wire_payloads[-4:]
    assistant = terminal[0]

    assert [payload["type"] for payload in terminal] == [
        "data-assistant-message",
        "text-end",
        "finish-step",
        "finish",
    ]
    assert assistant["data"]["message"]["metadata"] == {
        "context_window": context_window,
        "futureMetadata": True,
    }
    assert not any(payload.get("type") == "data-model-usage" for payload in wire_payloads)
    assert payloads[-1] == "[DONE]"


@pytest.mark.asyncio
async def test_terminal_assistant_message_scrubs_legacy_renderer_fields():
    async def source():
        yield make_event(
            "complete",
            sequence=1,
            data={
                "message": {
                    "id": "m-1",
                    "sender": 2,
                    "content": "Answer",
                    "message_metadata": {
                        "provider": "openai",
                        "images": [{"url": "https://img.test/a.png", "mime": "image/png"}],
                        "has_images": True,
                        "images_count": 1,
                        "agentic_images_count": 1,
                        "live_widgets": [{"widget_id": "w-1"}],
                        "canvas_artifact": {"content": "<html></html>"},
                        "pending_tool_calls": [],
                        "_rich_item_candidates": [],
                        "conversation_id": "conversation-1",
                        "has_tool_calls": False,
                        "context_messages": 3,
                    },
                }
            },
        )

    payloads = await _collect_payloads(source)
    assistant_payload = next(
        payload
        for payload in payloads
        if payload != "[DONE]" and payload.get("type") == "data-assistant-message"
    )
    metadata = assistant_payload["data"]["message"]["metadata"]

    assert metadata == {"provider": "openai"}
    # Legacy images still surface as AI SDK file parts, extracted before the scrub.
    file_payloads = [
        payload for payload in payloads if payload != "[DONE]" and payload.get("type") == "file"
    ]
    assert [payload["url"] for payload in file_payloads] == ["https://img.test/a.png"]


@pytest.mark.asyncio
async def test_user_message_event_projects_wire_safe_payload():
    async def source():
        yield make_event(
            "user_message_created",
            sequence=1,
            data={
                "message": {
                    "id": "u-1",
                    "sender": 1,
                    "conversation_id": "conversation-1",
                    "content": "hi",
                    "created_at": "2026-07-02T01:00:00Z",
                    "updated_at": "2026-07-02T01:00:01Z",
                    "message_metadata": {},
                }
            },
        )
        yield make_event("complete", sequence=2, data={"message": {"id": "m-1"}})

    payloads = await _collect_payloads(source)
    user_payload = next(
        payload
        for payload in payloads
        if payload != "[DONE]" and payload.get("type") == "data-user-message"
    )
    message = user_payload["data"]["message"]

    assert message["id"] == "u-1"
    assert message["role"] == "user"
    assert message["content"] == "hi"
    assert message["createdAt"] == "2026-07-02T01:00:00Z"
    assert "sender" not in message
    assert "conversation_id" not in message
    assert "updated_at" not in message


def test_ai_sdk_extracts_file_part_data_url_attachment():
    payload = [
        {
            "role": "user",
            "parts": [
                {"type": "text", "text": "inspect"},
                {
                    "type": "file",
                    "name": "screen.png",
                    "mediaType": "image/png",
                    "url": "data:image/png;base64,abc",
                },
            ],
        }
    ]

    assert _extract_user_attachments(payload) == [
        {"name": "screen.png", "mime": "image/png", "data": "data:image/png;base64,abc"}
    ]


def test_ai_sdk_ignores_unusable_request_attachments():
    payload = [
        {
            "role": "user",
            "parts": [
                {"type": "text", "text": "inspect"},
                {
                    "type": "file",
                    "name": "screen.png",
                    "mediaType": "image/png",
                    "url": "blob:http://app.local/123",
                },
                {
                    "type": "file",
                    "name": "notes.pdf",
                    "mediaType": "application/pdf",
                    "data": "JVBERi0=",
                },
                {
                    "type": "file",
                    "name": "notes.txt",
                    "url": "data:text/plain;base64,aGVsbG8=",
                },
            ],
        }
    ]

    assert _extract_user_attachments(payload) == []
    assert _has_user_attachment_candidates(payload) is True


def test_ai_sdk_ignores_invalid_raw_base64_attachment():
    payload = [
        {
            "role": "user",
            "attachments": [
                {
                    "type": "image",
                    "name": "broken.png",
                    "mimeType": "image/png",
                    "data": "not base64!",
                }
            ],
        }
    ]

    assert _extract_user_attachments(payload) == []
    assert _has_user_attachment_candidates(payload) is True
