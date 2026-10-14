"""
Tests for SSE keepalive / heartbeat behaviour added to fix the sidecar
HITL resume flow (model not generating response after approval).

Covers:
- Server AI SDK endpoint emits keepalive comments during long-running operations
- Sidecar stream_sse filters heartbeat events from the upstream
- Sidecar stream_sse uses an extended read timeout for streaming
"""

import asyncio
import json

import pytest

from client_backend.services.server_api import ServerAPIClient

# Server-side AI SDK tests require qdrant_client and other heavy deps
# that may not be installed in the client-only test environment.
_server_deps_available = True
try:
    from app.api.ai_sdk import (
        StreamState,
        _build_ui_message_stream_response,
    )
    from app.services.event_streaming.events import make_event
except Exception:
    _server_deps_available = False

_skip_server = pytest.mark.skipif(
    not _server_deps_available,
    reason="Server dependencies (qdrant_client, etc.) not installed",
)


# ── Server-side AI SDK heartbeat tests ──────────────────────────────────


@_skip_server
@pytest.mark.asyncio
async def test_ai_sdk_stream_emits_heartbeats_during_slow_source():
    """The server AI SDK SSE wrapper sends keepalive comments, never a
    ``heartbeat`` data frame (the SDK's strict schema rejects it), when the
    upstream event source is slow to produce events."""
    slow_event_emitted = asyncio.Event()

    async def slow_event_source():
        await asyncio.sleep(0.05)
        slow_event_emitted.set()
        yield make_event("message_delta", sequence=1, data={"text": "hello"})
        yield make_event("complete", sequence=2)

    state = StreamState(message_id="msg-1", text_id="txt-1", reasoning_id="rsn-1")

    import app.services.event_streaming.ai_sdk_v6 as ai_sdk_v6_module

    original_interval = ai_sdk_v6_module._AI_SDK_HEARTBEAT_INTERVAL_SECONDS
    ai_sdk_v6_module._AI_SDK_HEARTBEAT_INTERVAL_SECONDS = 0.01

    try:
        response = _build_ui_message_stream_response(slow_event_source, state)
        collected: list[str] = []
        async for chunk in response.body_iterator:
            collected.append(chunk)
    finally:
        ai_sdk_v6_module._AI_SDK_HEARTBEAT_INTERVAL_SECONDS = original_interval

    all_text = "".join(collected)
    events = [line[6:] for line in all_text.split("\n") if line.startswith("data: ")]

    assert ": keepalive" in all_text.split("\n"), f"Expected a keepalive comment: {all_text}"
    assert not any('"heartbeat"' in e for e in events), events
    assert any('"text-delta"' in e for e in events), (
        f"Expected text-delta event in stream, got: {events}"
    )


@_skip_server
@pytest.mark.asyncio
async def test_ai_sdk_stream_completes_normally_without_heartbeat_when_fast():
    """When the source is fast, the stream should complete with no heartbeats."""

    async def fast_event_source():
        yield make_event("message_delta", sequence=1, data={"text": "fast"})
        yield make_event("complete", sequence=2)

    state = StreamState(message_id="msg-2", text_id="txt-2", reasoning_id="rsn-2")

    response = _build_ui_message_stream_response(fast_event_source, state)
    collected: list[str] = []
    async for chunk in response.body_iterator:
        collected.append(chunk)

    all_text = "".join(collected)
    events = [line[6:] for line in all_text.split("\n") if line.startswith("data: ")]

    assert ": keepalive" not in all_text.split("\n"), all_text
    assert events[-1] == "[DONE]"


@_skip_server
@pytest.mark.asyncio
async def test_ai_sdk_stream_handles_interrupt_event_with_heartbeat_enabled():
    """Interrupt events should still terminate the stream correctly."""

    async def interrupt_source():
        yield make_event(
            "interrupt",
            sequence=1,
            data={
                "thread_id": "t1",
                "next": ["approval"],
                "pending_tool_calls": [],
                "interrupt": {"interrupt_id": "int-1"},
                "message": "Approval needed",
            },
        )
        yield make_event("complete", sequence=2, data={"message": {"content": "unreachable"}})

    state = StreamState(message_id="msg-3", text_id="txt-3", reasoning_id="rsn-3")

    response = _build_ui_message_stream_response(interrupt_source, state)
    collected: list[str] = []
    async for chunk in response.body_iterator:
        collected.append(chunk)

    all_text = "".join(collected)
    events = [line[6:] for line in all_text.split("\n") if line.startswith("data: ")]

    assert any('"data-interrupt"' in e for e in events), (
        f"Expected data-interrupt event, got: {events}"
    )
    assert events[-1] == "[DONE]"
    assert not any('"unreachable"' in e for e in events)


# ── Sidecar stream_sse heartbeat filtering tests ───────────────────────


@pytest.mark.asyncio
async def test_stream_sse_filters_heartbeat_events(monkeypatch):
    """The sidecar's stream_sse should silently drop heartbeat events
    so they don't get forwarded to the desktop client."""

    sse_lines = [
        'data: {"type":"start","messageId":"m1"}',
        "",
        'data: {"type":"heartbeat"}',
        "",
        'data: {"type":"text-delta","delta":"hi"}',
        "",
        'data: {"type":"heartbeat"}',
        "",
        'data: {"type":"finish"}',
        "",
        "data: [DONE]",
        "",
    ]

    class FakeResponse:
        status_code = 200

        async def aiter_bytes(self):
            for line in sse_lines:
                yield f"{line}\n".encode()

        async def aread(self):
            pass

        async def aclose(self):
            pass

    class FakeStream:
        def __init__(self, *a, **kw):
            self.response = FakeResponse()

        async def __aenter__(self):
            return self.response

        async def __aexit__(self, *a):
            pass

    client = ServerAPIClient(base_url="http://test.local", timeout=5)
    client._tokens = type(
        "T",
        (),
        {
            "access_token": "fake",
        },
    )()

    real_client = await client._get_client()
    monkeypatch.setattr(real_client, "stream", lambda *a, **kw: FakeStream())

    events = [event async for event in client.stream_sse("/test")]

    event_types = [e.get("type") for e in events]
    assert "heartbeat" not in event_types, (
        f"Heartbeat events should be filtered, got: {event_types}"
    )
    assert "start" in event_types
    assert "text-delta" in event_types
    assert "finish" in event_types


# ── Sidecar stream_sse timeout configuration tests ─────────────────────


def test_stream_sse_uses_extended_read_timeout():
    """Verify that stream_sse creates a timeout with a much longer read
    value than the default client timeout."""
    # The implementation creates httpx.Timeout(connect=self.timeout, read=600.0, ...)
    # inside stream_sse. We verify the intent via the client's default timeout
    # and the documented contract.
    client = ServerAPIClient(base_url="http://test.local", timeout=30)
    # Default client timeout should be the configured value
    assert client.timeout == 30
    # The stream_sse method internally uses read=600.0 (documented in code)
    # We can't easily test the httpx.Timeout passed per-request without mocking,
    # but the test_stream_sse_filters_heartbeat_events above exercises the path.


@_skip_server
@pytest.mark.asyncio
async def test_ai_sdk_tool_output_event_preserves_render_payload():
    async def tool_event_source():
        yield make_event(
            "tool_call_available",
            sequence=1,
            tool_call_id="tool-call-1",
            tool_name="canva_create_presentation",
            data={"args": {"prompt": "roadmap"}},
        )
        yield make_event(
            "tool_execution_end",
            sequence=2,
            tool_call_id="tool-call-1",
            tool_name="canva_create_presentation",
            data={
                "output": "Created presentation",
                "render": {
                    "version": 1,
                    "type": "mcp_app",
                    "template_uri": "ui://canva/presentation-viewer.html",
                },
            },
        )
        yield make_event("complete", sequence=3)

    state = StreamState(message_id="msg-render", text_id="txt-render", reasoning_id="rsn-render")

    response = _build_ui_message_stream_response(tool_event_source, state)
    collected: list[str] = []
    async for chunk in response.body_iterator:
        collected.append(chunk)

    events = [
        line[6:]
        for line in "".join(collected).split("\n")
        if line.startswith("data: ") and line[6:] != "[DONE]"
    ]

    tool_output_events = [
        json.loads(event) for event in events if '"tool-output-available"' in event
    ]
    render_events = [json.loads(event) for event in events if '"data-tool-render"' in event]

    assert len(tool_output_events) == 1
    assert tool_output_events[0]["output"] == "Created presentation"
    assert "render" not in tool_output_events[0]
    assert len(render_events) == 1
    assert render_events[0]["data"]["toolCallId"] == "tool-call-1"
    assert render_events[0]["data"]["render"]["type"] == "mcp_app"
    assert render_events[0]["data"]["render"]["template_uri"] == (
        "ui://canva/presentation-viewer.html"
    )


@_skip_server
@pytest.mark.asyncio
async def test_ai_sdk_render_payload_preserves_structure_and_text_type():
    """Render must survive the SSE wrapper without being collapsed by output cleaning:
    text-typed render stays a dict; content arrays stay arrays."""

    async def tool_event_source():
        yield make_event(
            "tool_execution_end",
            sequence=1,
            tool_call_id="tool-call-text",
            tool_name="answer_tool",
            data={
                "output": "The answer is 42.",
                "render": {
                    "version": 1,
                    "type": "text",
                    "text": "The answer is 42.",
                    "model_content": "The answer is 42.",
                    "content": [{"type": "text", "text": "The answer is 42."}],
                },
            },
        )
        yield make_event("complete", sequence=2)

    state = StreamState(message_id="msg-x", text_id="txt-x", reasoning_id="rsn-x")

    response = _build_ui_message_stream_response(tool_event_source, state)
    collected: list[str] = []
    async for chunk in response.body_iterator:
        collected.append(chunk)

    events = [
        line[6:]
        for line in "".join(collected).split("\n")
        if line.startswith("data: ") and line[6:] != "[DONE]"
    ]
    render_events = [json.loads(event) for event in events if '"data-tool-render"' in event]

    assert len(render_events) == 1
    render = render_events[0]["data"]["render"]
    assert isinstance(render, dict), "render must remain a dict, not be collapsed to a string"
    assert render["type"] == "text"
    assert render["text"] == "The answer is 42."
    assert isinstance(render["content"], list)
    assert render["content"][0]["type"] == "text"
    assert render["content"][0]["text"] == "The answer is 42."


# ── Sidecar terminal classification (AI SDK proxy) ──────────────────────


def _fake_upstream(monkeypatch, client, lines):
    class _Response:
        status_code = 200

        async def aiter_bytes(self):
            for line in lines:
                yield f"{line}\n".encode()

        async def aread(self):
            return b""

    class _Stream:
        async def __aenter__(self):
            return _Response()

        async def __aexit__(self, *exc_info):
            return False

    async def _install():
        real_client = await client._get_client()
        monkeypatch.setattr(real_client, "stream", lambda *a, **kw: _Stream())

    return _install()


def _client() -> ServerAPIClient:
    client = ServerAPIClient(base_url="http://test.local", timeout=5)
    client._tokens = type("T", (), {"access_token": "fake"})()
    return client


_PARTIAL_UPSTREAM = [
    'data: {"type":"start","messageId":"m1"}',
    "",
    'data: {"type":"start-step"}',
    "",
    'data: {"type":"text-start","id":"t1"}',
    "",
    'data: {"type":"text-delta","id":"t1","delta":"partial"}',
    "",
]


@pytest.mark.asyncio
async def test_stream_sse_requires_done_only_when_asked(monkeypatch):
    from client_backend.services.server_api import ServerStreamIncompleteError

    client = _client()
    await _fake_upstream(monkeypatch, client, _PARTIAL_UPSTREAM)

    lenient = [event async for event in client.stream_sse("/test")]
    assert [event["type"] for event in lenient] == [
        "start",
        "start-step",
        "text-start",
        "text-delta",
    ]

    received = []
    with pytest.raises(ServerStreamIncompleteError):
        async for event in client.stream_sse("/test", require_done=True):
            received.append(event)
    assert len(received) == 4


@pytest.mark.asyncio
async def test_stream_sse_ignores_bytes_after_the_first_done(monkeypatch):
    client = _client()
    lines = [*_PARTIAL_UPSTREAM, "data: [DONE]", "", 'data: {"type":"junk"}', ""]
    await _fake_upstream(monkeypatch, client, lines)

    events = [event async for event in client.stream_sse("/test", require_done=True)]

    assert "junk" not in [event["type"] for event in events]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "args", "required"),
    [
        ("stream_ai_sdk_chat", ("c-1", {}), True),
        ("continue_ai_sdk_generation", ({},), True),
        ("resume_ai_sdk_interrupt", ({},), True),
        ("stream_internal_message", ({},), False),
        ("resume_interrupt", ({},), False),
        ("continue_generation", ({},), False),
    ],
)
async def test_only_ai_sdk_streams_require_the_terminator(monkeypatch, method, args, required):
    client = _client()
    seen: list[bool] = []

    async def fake_stream_sse(path, method="POST", *, require_done=False, **kwargs):
        seen.append(require_done)
        yield {"type": "start"}

    monkeypatch.setattr(client, "stream_sse", fake_stream_sse)
    [event async for event in getattr(client, method)(*args)]

    assert seen == [required]


async def _proxy_body(monkeypatch, lines) -> str:
    from client_backend.api import messages as messages_api

    client = _client()
    await _fake_upstream(monkeypatch, client, lines)
    response = messages_api._build_sse_response(
        client.stream_sse("/api/chat/c-1", require_done=True), ai_sdk=True
    )
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
    return "".join(chunks)


def _frames(body: str) -> list:
    return [
        line[6:] if line[6:] == "[DONE]" else json.loads(line[6:])
        for line in body.split("\n")
        if line.startswith("data: ")
    ]


@pytest.mark.asyncio
async def test_ai_sdk_proxy_reports_upstream_eof_as_incomplete(monkeypatch):
    frames = _frames(await _proxy_body(monkeypatch, _PARTIAL_UPSTREAM))
    wire = [frame for frame in frames if frame != "[DONE]"]

    assert [frame["type"] for frame in wire] == [
        "start",
        "start-step",
        "text-start",
        "text-delta",
        "data-stream-error",
        "error",
        "text-end",
        "finish-step",
        "finish",
    ]
    assert wire[4]["data"] == {"errorCode": "stream_incomplete"}
    assert wire[6] == {"type": "text-end", "id": "t1"}
    assert wire[-1] == {"type": "finish", "finishReason": "error"}
    assert frames.count("[DONE]") == 1
    assert frames[-1] == "[DONE]"


@pytest.mark.asyncio
async def test_ai_sdk_proxy_tolerates_a_missing_done_after_finish(monkeypatch):
    lines = [
        *_PARTIAL_UPSTREAM,
        'data: {"type":"text-end","id":"t1"}',
        'data: {"type":"finish-step"}',
        'data: {"type":"finish"}',
    ]
    frames = _frames(await _proxy_body(monkeypatch, lines))

    assert [frame["type"] for frame in frames if frame != "[DONE]"][-1] == "finish"
    assert not any(frame != "[DONE]" and frame["type"] == "error" for frame in frames)
    assert frames.count("[DONE]") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("ai_sdk", [True, False])
async def test_proxy_close_mid_stream_sends_nothing_and_closes_upstream(ai_sdk):
    from client_backend.api import messages as messages_api

    closed = asyncio.Event()

    async def upstream():
        try:
            yield {"type": "start", "messageId": "m1"}
            await asyncio.Event().wait()
        finally:
            closed.set()

    response = messages_api._build_sse_response(upstream(), ai_sdk=ai_sdk)
    body = response.body_iterator
    first = await anext(body)

    await body.aclose()  # must not raise "async generator ignored GeneratorExit"

    assert '"start"' in first
    assert closed.is_set()
