from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from contextvars import ContextVar

import pytest

from app.ai.tool_execution import execute_tool_calls
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
async def test_text_stream_preserves_source_context_until_cleanup():
    chat_scope = ContextVar("test_chat_scope", default=None)

    @contextmanager
    def bind_chat_scope():
        token = chat_scope.set("bound")
        try:
            yield
        finally:
            chat_scope.reset(token)

    async def source():
        with bind_chat_scope():
            yield make_event("message_delta", sequence=1, data={"text": "hello"})
        yield make_event("complete", sequence=2, data={"message": {"id": "m-1"}})

    payloads = await _collect_payloads(source)

    errors = [
        payload for payload in payloads if payload != "[DONE]" and payload.get("type") == "error"
    ]
    assert errors == []
    assert any(
        payload != "[DONE]"
        and payload.get("type") == "text-delta"
        and payload.get("delta") == "hello"
        for payload in payloads
    )
    assert payloads[-1] == "[DONE]"


@pytest.mark.asyncio
async def test_heartbeat_stream_closes_source_in_its_own_context():
    chat_scope = ContextVar("test_close_chat_scope", default=None)
    cleaned = asyncio.Event()
    block_forever = asyncio.Event()

    async def source():
        token = chat_scope.set("bound")
        try:
            yield make_event("message_delta", sequence=1, data={"text": "hello"})
            await block_forever.wait()
        finally:
            chat_scope.reset(token)
            cleaned.set()

    adapter = AISDKV6StreamAdapter(
        source,
        AISDKV6StreamState(message_id="m-1", text_id="t-1", reasoning_id="r-1"),
        heartbeat_interval_seconds=1,
    )
    events = adapter._events_with_heartbeats()

    first = await anext(events)
    await events.aclose()

    assert first.type == "message_delta"
    await asyncio.wait_for(cleaned.wait(), timeout=0.1)


@pytest.mark.asyncio
async def test_image_preview_v2_reference_final_projects_reference_url():
    """A schema-v2 reference-delivery FINAL image_preview projects to a
    ``data-image-preview`` carrying the protected relative URL verbatim (no
    base64), stable by ``item_id``."""

    async def source():
        yield make_event(
            "image_preview",
            sequence=1,
            data={
                "schema_version": 2,
                "item_id": "image-preview-0",
                "image_index": 0,
                "status": "final",
                "seq": 2,
                "media_type": "image/png",
                "delivery": {
                    "kind": "reference",
                    "image_id": "11111111-1111-4111-8111-111111111111",
                    "url": "/chat-images/11111111-1111-4111-8111-111111111111",
                },
            },
        )
        yield make_event("complete", sequence=2, data={"message": {"id": "m-1"}})

    payloads = await _collect_payloads(source)
    previews = [
        p
        for p in payloads
        if p != "[DONE]" and isinstance(p, dict) and p.get("type") == "data-image-preview"
    ]
    assert previews == [
        {
            "type": "data-image-preview",
            "id": "image-preview-0",
            "data": {
                "imageIndex": 0,
                "status": "final",
                "mediaType": "image/png",
                "url": "/chat-images/11111111-1111-4111-8111-111111111111",
                "seq": 2,
            },
            "transient": True,
        }
    ]


@pytest.mark.asyncio
async def test_image_preview_v2_inline_partial_projects_data_url():
    """A schema-v2 inline-delivery PARTIAL image_preview projects to a
    ``data-image-preview`` carrying the assembled ``data:`` URL."""

    async def source():
        yield make_event(
            "image_preview",
            sequence=1,
            data={
                "schema_version": 2,
                "item_id": "image-preview-0",
                "image_index": 0,
                "status": "partial",
                "seq": 1,
                "media_type": "image/png",
                "delivery": {"kind": "inline", "data_url": "data:image/png;base64,QUJD"},
            },
        )
        yield make_event("complete", sequence=2, data={"message": {"id": "m-1"}})

    payloads = await _collect_payloads(source)
    previews = [
        p
        for p in payloads
        if p != "[DONE]" and isinstance(p, dict) and p.get("type") == "data-image-preview"
    ]
    assert previews == [
        {
            "type": "data-image-preview",
            "id": "image-preview-0",
            "data": {
                "imageIndex": 0,
                "status": "partial",
                "mediaType": "image/png",
                "url": "data:image/png;base64,QUJD",
                "seq": 1,
            },
            "transient": True,
        }
    ]


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
    assert "render" not in tool_payloads[2]
    render_parts = [p for p in payloads if p != "[DONE]" and p["type"] == "data-tool-render"]
    assert render_parts == [
        {
            "type": "data-tool-render",
            "data": {"toolCallId": "call-1", "render": {"type": "text", "text": "result"}},
            "transient": True,
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("with_proxy", [True, False])
@pytest.mark.parametrize(
    "tool_name", ["brave_image_search", "brave_image_search__brave_image_search"]
)
async def test_direct_brave_tool_output_never_exposes_original_url_in_ai_sdk_stream(
    with_proxy,
    tool_name,
):
    original_url = "https://private-origin.example/full.jpg?token=secret"
    thumbnail_url = "https://imgs.search.brave.com/proxy.jpg"
    display_url = thumbnail_url if with_proxy else original_url

    class _BraveTool:
        name = tool_name
        metadata = (
            {
                "tool_origin": "server_mcp",
                "qualified_tool_id": "brave_image_search::brave_image_search",
                "source_tool_name": "brave_image_search",
                "server_name": "brave_image_search",
            }
            if tool_name != "brave_image_search"
            else {}
        )

        async def ainvoke(self, args):
            image = {
                "url": display_url,
                "original_image_url": original_url,
                "provider": "brave_image_search",
                "confidence": "high",
                "mime_type": "image/jpeg",
                "width": 1200,
                "height": 800,
            }
            if with_proxy:
                image["thumbnail_url"] = thumbnail_url
            return [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "provider": "brave_image_search",
                            "images": [image],
                        }
                    ),
                }
            ]

    outputs, _, _ = await execute_tool_calls(
        tool_calls=[
            {
                "id": "brave-call",
                "name": tool_name,
                "args": {"query": "T1 photo"},
            }
        ],
        tool_map={tool_name: _BraveTool()},
    )

    async def source():
        yield make_event(
            "tool_execution_end",
            sequence=1,
            tool_call_id="brave-call",
            tool_name=tool_name,
            data={
                "output": outputs[0]["content"],
                "render": outputs[0]["render"],
            },
        )
        yield make_event("complete", sequence=2, data={"message": {"id": "m-1"}})

    payloads = await _collect_payloads(source)
    serialized = json.dumps(payloads, ensure_ascii=False)

    assert "original_image_url" not in serialized
    assert original_url not in serialized
    if with_proxy:
        assert thumbnail_url in serialized


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
    wire = [payload for payload in payloads if payload != "[DONE]"]
    error_index = next(i for i, p in enumerate(wire) if p["type"] == "error")

    assert wire[error_index - 1 : error_index + 1] == [
        {
            "type": "data-stream-error",
            "data": {"statusCode": 409, "errorCode": "INTERRUPT_ALREADY_RESOLVED"},
            "transient": True,
        },
        {"type": "error", "errorText": "This interrupt has already been resolved."},
    ]
    assert payloads[-1] == "[DONE]"


@pytest.mark.asyncio
async def test_ai_sdk_stream_exception_retains_custom_http_metadata():
    async def source():
        raise CustomHTTPException(409, "Already resolved", "INTERRUPT_ALREADY_RESOLVED")
        yield  # pragma: no cover

    payloads = await _collect_payloads(source)
    wire = [payload for payload in payloads if payload != "[DONE]"]
    error_index = next(i for i, p in enumerate(wire) if p["type"] == "error")

    assert wire[error_index - 1 : error_index + 1] == [
        {
            "type": "data-stream-error",
            "data": {"statusCode": 409, "errorCode": "INTERRUPT_ALREADY_RESOLVED"},
            "transient": True,
        },
        {"type": "error", "errorText": "Already resolved"},
    ]
    assert payloads[-1] == "[DONE]"


@pytest.mark.asyncio
async def test_ai_sdk_stream_exception_text_is_not_sent_to_the_client():
    secret = "postgresql://app:hunter2@db/chat"

    async def source():
        raise RuntimeError(secret)
        yield  # pragma: no cover

    payloads = await _collect_payloads(source)

    assert next(
        payload for payload in payloads if payload != "[DONE]" and payload["type"] == "error"
    ) == {"type": "error", "errorText": "Response generation failed (RuntimeError)."}
    assert secret not in str(payloads)
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


# -- terminal outcomes ---------------------------------------------------------


def _finish(payloads):
    finishes = [p for p in payloads if p != "[DONE]" and p["type"] == "finish"]
    assert len(finishes) == 1, finishes
    return finishes[0]


def _stream_errors(payloads):
    return [p for p in payloads if p != "[DONE]" and p["type"] == "data-stream-error"]


def _errors(payloads):
    return [p for p in payloads if p != "[DONE]" and p["type"] == "error"]


async def _collect_with_state(source, **state_kwargs):
    state = AISDKV6StreamState(message_id="m-1", text_id="t-1", reasoning_id="r-1", **state_kwargs)
    adapter = AISDKV6StreamAdapter(lambda: source(), state)
    chunks = [chunk async for chunk in adapter.iter_sse()]
    payloads = []
    for line in "".join(chunks).splitlines():
        if line.startswith("data: "):
            raw = line[6:]
            payloads.append(raw if raw == "[DONE]" else json.loads(raw))
    return payloads, state


def _assert_incomplete(payloads):
    wire = [p for p in payloads if p != "[DONE]"]
    order = [p["type"] for p in wire]
    assert _stream_errors(payloads) == [
        {
            "type": "data-stream-error",
            "data": {"errorCode": "stream_incomplete"},
            "transient": True,
        }
    ]
    assert len(_errors(payloads)) == 1
    assert order.index("data-stream-error") < order.index("error") < order.index("finish")
    assert _finish(payloads) == {"type": "finish", "finishReason": "error"}
    assert order.count("text-end") == 1
    assert payloads.count("[DONE]") == 1
    assert payloads[-1] == "[DONE]"


@pytest.mark.asyncio
async def test_empty_source_is_an_incomplete_stream_not_a_success():
    async def source():
        return
        yield  # pragma: no cover

    payloads, state = await _collect_with_state(source)

    _assert_incomplete(payloads)
    assert state.outcome is None


@pytest.mark.asyncio
async def test_partial_only_eof_never_indicates_a_successful_finish():
    async def source():
        yield make_event("reasoning_delta", sequence=1, data={"text": "thinking"})
        yield make_event("message_delta", sequence=2, data={"text": "partial"})

    payloads, state = await _collect_with_state(source)

    _assert_incomplete(payloads)
    order = [p["type"] for p in payloads if p != "[DONE]"]
    assert order.count("reasoning-end") == 1
    assert [p["delta"] for p in payloads if p != "[DONE]" and p["type"] == "text-delta"] == [
        "partial"
    ]
    assert state.outcome is None


@pytest.mark.asyncio
@pytest.mark.parametrize("after_first_token", [False, True])
async def test_exception_finishes_with_an_error_reason(after_first_token):
    async def source():
        if after_first_token:
            yield make_event("message_delta", sequence=1, data={"text": "partial"})
        raise RuntimeError("provider failure")
        yield  # pragma: no cover

    payloads, state = await _collect_with_state(source)

    assert len(_errors(payloads)) == 1
    assert _finish(payloads) == {"type": "finish", "finishReason": "error"}
    order = [p["type"] for p in payloads if p != "[DONE]"]
    assert order.index("error") < order.index("text-end")
    assert order.count("text-end") == 1
    assert payloads.count("[DONE]") == 1
    assert state.outcome == "error"


@pytest.mark.asyncio
async def test_completion_finishes_normally():
    async def source():
        yield make_event("message_delta", sequence=1, data={"text": "done"})
        yield make_event("complete", sequence=2, data={"message": {"id": "m-1"}})

    payloads, state = await _collect_with_state(source)

    assert _finish(payloads) == {"type": "finish"}
    assert _errors(payloads) == []
    assert state.outcome == "complete"


@pytest.mark.asyncio
async def test_error_event_finishes_with_an_error_reason():
    async def source():
        yield make_event("error", sequence=1, data={"error": "Generation failed"})

    payloads, state = await _collect_with_state(source)

    assert _finish(payloads) == {"type": "finish", "finishReason": "error"}
    assert len(_errors(payloads)) == 1
    assert state.outcome == "error"


@pytest.mark.asyncio
async def test_hitl_interrupt_is_a_legitimate_terminal_outcome():
    async def source():
        yield make_event(
            "interrupt",
            sequence=1,
            data={"thread_id": "t", "interrupt": {"interrupt_id": "i"}, "message": "Approve?"},
        )

    payloads, state = await _collect_with_state(source)

    assert _errors(payloads) == []
    assert _finish(payloads) == {"type": "finish"}
    assert state.outcome == "interrupt"


@pytest.mark.asyncio
@pytest.mark.parametrize("offered", [True, False], ids=["continuable", "not-continuable"])
async def test_validated_budget_pause_is_a_legitimate_terminal_outcome(offered):
    async def source():
        yield make_event("message_delta", sequence=1, data={"text": "Validated partial"})
        yield make_event(
            "message_end",
            sequence=2,
            message_id="m-1",
            data={"message": {"id": "m-1", "content": "Validated partial"}},
        )
        if offered:
            yield make_event(
                "continuation_available",
                sequence=3,
                data={"generation_id": "g-1", "continuation_id": "c-1", "version": 3},
            )

    payloads, state = await _collect_with_state(source)

    assert _errors(payloads) == []
    assert _finish(payloads) == {"type": "finish"}
    assert state.outcome == "paused"
    phases = [
        p["data"]["phase"] for p in payloads if p != "[DONE]" and p["type"] == "data-generation"
    ]
    assert phases == (["continuation_available"] if offered else [])


@pytest.mark.asyncio
async def test_stop_without_a_terminal_event_is_reported_as_incomplete():
    """A Stop ends the service stream without a terminal event; the adapter does
    not invent `stopped` -- the client reads the generation status instead."""

    async def source():
        yield make_event("message_delta", sequence=1, data={"text": "partial before stop"})

    payloads, _ = await _collect_with_state(source)

    _assert_incomplete(payloads)
    assert not any(
        p != "[DONE]" and p["type"] == "data-generation" for p in payloads
    ), "a transport outcome must not publish a lifecycle transition"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["stopped", "completed_partial"])
async def test_a_service_published_stop_status_is_a_legitimate_end(status):
    async def source():
        yield make_event("message_delta", sequence=1, data={"text": "partial"})
        yield make_event(
            "generation_status",
            sequence=2,
            data={"generation_id": "g-1", "status": "stop_requested", "version": 3},
        )
        yield make_event(
            "generation_status",
            sequence=3,
            data={"generation_id": "g-1", "status": status, "version": 4},
        )

    payloads, state = await _collect_with_state(source)

    assert _errors(payloads) == []
    assert _finish(payloads) == {"type": "finish"}
    assert state.outcome == "stopped"


@pytest.mark.asyncio
async def test_a_pending_stop_status_alone_is_still_incomplete():
    """`stop_requested` is not an outcome; the client polls for the real one."""

    async def source():
        yield make_event(
            "generation_status",
            sequence=1,
            data={"generation_id": "g-1", "status": "stop_requested", "version": 3},
        )

    payloads, state = await _collect_with_state(source)

    _assert_incomplete(payloads)
    assert state.outcome is None


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["complete", "error"])
async def test_duplicate_terminal_events_close_the_stream_once(terminal):
    async def source():
        for sequence in (1, 2):
            data = {"message": {"id": "m-1"}} if terminal == "complete" else {"error": "x"}
            yield make_event(terminal, sequence=sequence, data=data)

    payloads, _ = await _collect_with_state(source)

    _finish(payloads)
    assert len(_errors(payloads)) == (1 if terminal == "error" else 0)
    assert payloads.count("[DONE]") == 1


@pytest.mark.asyncio
async def test_outer_aclose_during_a_tool_wait_closes_the_source_before_returning():
    cleaned = asyncio.Event()
    tool_wait = asyncio.Event()

    async def source():
        try:
            yield make_event("message_delta", sequence=1, data={"text": "before tool"})
            await tool_wait.wait()  # never released: a tool that is still running
            yield make_event("complete", sequence=2)  # pragma: no cover
        finally:
            cleaned.set()

    adapter = AISDKV6StreamAdapter(
        source,
        AISDKV6StreamState(message_id="m-1", text_id="t-1", reasoning_id="r-1"),
        heartbeat_interval_seconds=30,
    )
    stream = adapter.iter_sse()
    async for chunk in stream:
        if '"text-delta"' in chunk:
            break

    await stream.aclose()

    assert cleaned.is_set(), "upstream cleanup must finish before aclose() returns"


@pytest.mark.asyncio
async def test_cancellation_emits_no_frames_and_propagates():
    cleaned = asyncio.Event()
    reached_wait = asyncio.Event()
    frames: list[str] = []

    async def source():
        try:
            yield make_event("message_delta", sequence=1, data={"text": "partial"})
            reached_wait.set()
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    adapter = AISDKV6StreamAdapter(
        source,
        AISDKV6StreamState(message_id="m-1", text_id="t-1", reasoning_id="r-1"),
        heartbeat_interval_seconds=30,
    )

    async def consume():
        async for chunk in adapter.iter_sse():
            frames.append(chunk)

    task = asyncio.create_task(consume())
    await asyncio.wait_for(reached_wait.wait(), timeout=1)
    sent_before_cancel = len(frames)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(frames) == sent_before_cancel
    assert not any("[DONE]" in frame for frame in frames)
    assert cleaned.is_set()
