"""Generate AI SDK wire fixtures from the real adapter and the real sidecar proxy.

Python assertions here pin the frame shapes the pinned ``ai`` package accepts.
When ``AI_SDK_FIXTURE_DIR`` is set, every complete stream is also written there
with a ``manifest.json`` so ``tests/ai_sdk_wire/wire.test.mjs`` can parse each
one with the real SDK parser -- the compatibility oracle, since the SDK
validates every frame against strict schemas before application code sees it.

Run both halves together (see docs/superpowers/plans/
2026-10-14-production-answer-streaming.md, Task 1).
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest

from app.api.ai_sdk import _build_ui_message_stream_response
from app.core.exceptions import CustomHTTPException
from app.services.event_streaming.ai_sdk_v6 import AISDKV6StreamAdapter, AISDKV6StreamState
from app.services.event_streaming.events import make_event
from client_backend.api import messages as messages_api
from client_backend.services.server_api import ServerAPIClient, ServerAPIError

_FIXTURE_DIR_ENV = "AI_SDK_FIXTURE_DIR"
_MANIFEST = "manifest.json"
_V1_HEADER = "x-vercel-ai-ui-message-stream"


# -- fixture recording ------------------------------------------------------


def _fixture_dir() -> Path | None:
    raw = os.environ.get(_FIXTURE_DIR_ENV)
    if not raw:
        return None
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="module", autouse=True)
def _fresh_manifest():
    directory = _fixture_dir()
    if directory is not None:
        (directory / _MANIFEST).write_text("[]", encoding="utf-8")
    yield


def record_fixture(
    name: str,
    wire: str,
    *,
    expect: str = "accept",
    requires: list[str] | None = None,
    sdk_text: str | None = None,
) -> None:
    """Write one complete SSE stream and its expected SDK outcome.

    ``sdk_text`` is the text a vanilla ``readUIMessageStream`` consumer ends
    with -- the SDK appends text deltas and never applies the custom
    reconcile part, so for a web draft this is the draft, not the answer.
    """
    directory = _fixture_dir()
    if directory is None:
        return
    filename = f"{name}.sse"
    (directory / filename).write_text(wire, encoding="utf-8", newline="")
    manifest_path = directory / _MANIFEST
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [entry for entry in entries if entry["file"] != filename]
    entry: dict[str, Any] = {"file": filename, "expect": expect, "requires": requires or []}
    if sdk_text is not None:
        entry["sdkText"] = sdk_text
    entries.append(entry)
    manifest_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


# -- wire helpers -------------------------------------------------------------


def _state(**kwargs: Any) -> AISDKV6StreamState:
    return AISDKV6StreamState(message_id="m-1", text_id="t-1", reasoning_id="r-1", **kwargs)


async def direct_wire(
    source: Callable[[], AsyncIterator[Any]],
    *,
    state: AISDKV6StreamState | None = None,
    heartbeat_interval_seconds: float | None = None,
) -> str:
    adapter = AISDKV6StreamAdapter(
        source,
        state or _state(),
        heartbeat_interval_seconds=heartbeat_interval_seconds,
    )
    return "".join([chunk async for chunk in adapter.iter_sse()])


class _FakeUpstreamResponse:
    status_code = 200

    def __init__(self, wire: str) -> None:
        self._wire = wire

    async def aiter_lines(self):
        for line in self._wire.split("\n"):
            yield line

    async def aread(self):
        return b""


class _FakeUpstreamStream:
    def __init__(self, wire: str) -> None:
        self._response = _FakeUpstreamResponse(wire)

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc_info):
        return False


async def _server_client_for(wire: str, monkeypatch) -> ServerAPIClient:
    client = ServerAPIClient(base_url="http://upstream.test", timeout=5)
    client._tokens = type("Tokens", (), {"access_token": "fixture-token"})()
    http_client = await client._get_client()
    monkeypatch.setattr(http_client, "stream", lambda *a, **kw: _FakeUpstreamStream(wire))
    return client


async def _drain(response) -> str:
    chunks: list[str] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)
    return "".join(chunks)


async def proxy_wire(upstream_wire: str, monkeypatch) -> str:
    """Pass a direct server stream through the sidecar's real AI SDK proxy."""
    client = await _server_client_for(upstream_wire, monkeypatch)
    response = messages_api._build_sse_response(
        client.stream_ai_sdk_chat("c-1", {}), ai_sdk=True
    )
    assert response.headers[_V1_HEADER] == "v1"
    return await _drain(response)


def data_frames(wire: str) -> list[Any]:
    frames: list[Any] = []
    for line in wire.split("\n"):
        if not line.startswith("data: "):
            continue
        raw = line[len("data: ") :]
        frames.append(raw if raw == "[DONE]" else json.loads(raw))
    return frames


def payloads(wire: str) -> list[dict[str, Any]]:
    return [frame for frame in data_frames(wire) if frame != "[DONE]"]


def types(wire: str) -> list[str]:
    return [frame["type"] for frame in payloads(wire)]


def assert_standard_frames_are_strict(wire: str) -> None:
    """The SDK's strict schemas reject unknown keys on these standard frames."""
    allowed = {
        "error": {"type", "errorText"},
        "tool-output-available": {
            "type",
            "toolCallId",
            "output",
            "providerExecuted",
            "dynamic",
            "preliminary",
        },
    }
    for frame in payloads(wire):
        assert frame["type"] != "heartbeat", "raw heartbeat frames are not SDK chunks"
        keys = allowed.get(frame["type"])
        if keys is not None:
            assert set(frame) <= keys, frame


# -- scenarios ----------------------------------------------------------------


async def _text_source():
    yield make_event("message_delta", sequence=1, data={"text": "Hello "})
    yield make_event("message_delta", sequence=2, data={"text": "world"})
    yield make_event("complete", sequence=3, data={"message": {"id": "m-1", "content": "x"}})


async def _tool_render_source():
    yield make_event(
        "tool_call_available",
        sequence=1,
        tool_call_id="c-1",
        tool_name="lookup",
        data={"args": {"q": "x"}},
    )
    yield make_event(
        "tool_execution_end",
        sequence=2,
        tool_call_id="c-1",
        tool_name="lookup",
        data={"output": {"text": "Result"}, "render": {"type": "text", "text": "Result"}},
    )
    yield make_event("complete", sequence=3, data={"message": {"id": "m-1"}})


async def _structured_error_event_source():
    yield make_event("message_delta", sequence=1, data={"text": "partial"})
    yield make_event(
        "error",
        sequence=2,
        data={
            "error": "This interrupt has already been resolved.",
            "status_code": 409,
            "error_code": "INTERRUPT_ALREADY_RESOLVED",
            "message": {"id": "e-1", "content": "Failed", "sender": 2},
        },
    )


async def _structured_exception_source():
    raise CustomHTTPException(409, "Already resolved", "INTERRUPT_ALREADY_RESOLVED")
    yield  # pragma: no cover


async def _unexpected_exception_source():
    raise RuntimeError("postgresql://app:hunter2@db/chat")
    yield  # pragma: no cover


# -- tests --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_stream_fixture_direct_and_proxied(monkeypatch):
    direct = await direct_wire(_text_source)
    proxied = await proxy_wire(direct, monkeypatch)
    record_fixture("text-direct", direct)
    record_fixture("text-proxied", proxied)

    assert_standard_frames_are_strict(direct)
    assert payloads(proxied) == payloads(direct)
    assert data_frames(direct).count("[DONE]") == 1
    assert data_frames(proxied).count("[DONE]") == 1


@pytest.mark.asyncio
async def test_direct_response_declares_the_v1_ui_message_stream_header():
    response = _build_ui_message_stream_response(_text_source, _state())
    body = await _drain(response)

    assert response.headers[_V1_HEADER] == "v1"
    assert data_frames(body)[-1] == "[DONE]"


@pytest.mark.asyncio
async def test_quiet_tool_wait_sends_keepalive_comments_not_heartbeat_frames(monkeypatch):
    release = asyncio.Event()

    async def quiet_source():
        yield make_event(
            "tool_call_available",
            sequence=1,
            tool_call_id="c-1",
            tool_name="slow_tool",
            data={"args": {}},
        )
        await release.wait()
        yield make_event(
            "tool_execution_end",
            sequence=2,
            tool_call_id="c-1",
            tool_name="slow_tool",
            data={"output": "done"},
        )
        yield make_event("complete", sequence=3, data={"message": {"id": "m-1"}})

    adapter = AISDKV6StreamAdapter(quiet_source, _state(), heartbeat_interval_seconds=0.01)
    chunks: list[str] = []
    async for chunk in adapter.iter_sse():
        chunks.append(chunk)
        if chunk.startswith(":") or '"heartbeat"' in chunk:
            release.set()
    direct = "".join(chunks)
    proxied = await proxy_wire(direct, monkeypatch)
    record_fixture("quiet-tool-direct", direct, requires=["tool-output-available"])
    record_fixture("quiet-tool-proxied", proxied, requires=["tool-output-available"])

    assert ": keepalive\n\n" in direct
    assert_standard_frames_are_strict(direct)
    assert payloads(proxied) == payloads(direct)


@pytest.mark.asyncio
async def test_tool_render_moves_to_a_custom_data_part(monkeypatch):
    direct = await direct_wire(_tool_render_source)
    proxied = await proxy_wire(direct, monkeypatch)
    record_fixture("tool-render-direct", direct, requires=["data-tool-render"])
    record_fixture("tool-render-proxied", proxied, requires=["data-tool-render"])
    frames = payloads(direct)

    assert_standard_frames_are_strict(direct)
    render_parts = [frame for frame in frames if frame["type"] == "data-tool-render"]
    assert render_parts == [
        {
            "type": "data-tool-render",
            "data": {"toolCallId": "c-1", "render": {"type": "text", "text": "Result"}},
            "transient": True,
        }
    ]
    output = next(frame for frame in frames if frame["type"] == "tool-output-available")
    assert output == {
        "type": "tool-output-available",
        "toolCallId": "c-1",
        "output": {"text": "Result"},
    }
    order = types(direct)
    assert order.index("data-tool-render") < order.index("tool-output-available")

    assert payloads(proxied) == frames


@pytest.mark.asyncio
async def test_structured_error_event_moves_metadata_before_the_error(monkeypatch):
    direct = await direct_wire(_structured_error_event_source)
    proxied = await proxy_wire(direct, monkeypatch)
    requires = ["data-stream-error", "data-error-message"]
    record_fixture("structured-error-direct", direct, requires=requires)
    record_fixture("structured-error-proxied", proxied, requires=requires)
    frames = payloads(direct)
    order = types(direct)

    assert_standard_frames_are_strict(direct)
    assert {"type": "error", "errorText": "This interrupt has already been resolved."} in frames
    metadata = next(frame for frame in frames if frame["type"] == "data-stream-error")
    assert metadata == {
        "type": "data-stream-error",
        "data": {"statusCode": 409, "errorCode": "INTERRUPT_ALREADY_RESOLVED"},
        "transient": True,
    }
    assert order.index("data-stream-error") < order.index("error")
    assert order.index("data-error-message") < order.index("error")

    assert payloads(proxied) == frames


@pytest.mark.asyncio
async def test_structured_exception_keeps_status_and_code_in_custom_metadata():
    direct = await direct_wire(_structured_exception_source)
    record_fixture("structured-exception-direct", direct, requires=["data-stream-error"])
    frames = payloads(direct)

    assert_standard_frames_are_strict(direct)
    assert {"type": "error", "errorText": "Already resolved"} in frames
    assert {
        "type": "data-stream-error",
        "data": {"statusCode": 409, "errorCode": "INTERRUPT_ALREADY_RESOLVED"},
        "transient": True,
    } in frames
    assert types(direct).index("data-stream-error") < types(direct).index("error")


@pytest.mark.asyncio
async def test_unexpected_exception_sends_only_sanitized_error_text():
    direct = await direct_wire(_unexpected_exception_source)
    record_fixture("unexpected-exception-direct", direct)

    assert_standard_frames_are_strict(direct)
    assert "hunter2" not in direct
    assert {"type": "error", "errorText": "Response generation failed (RuntimeError)."} in (
        payloads(direct)
    )


@pytest.mark.asyncio
async def test_proxy_structured_upstream_failure_uses_custom_metadata():
    async def failing_upstream():
        raise ServerAPIError(
            "Server error: 409",
            status_code=409,
            detail={"code": "INTERRUPT_CONFLICT", "message": "Interrupt was claimed."},
        )
        yield  # pragma: no cover

    body = await _drain(messages_api._build_sse_response(failing_upstream(), ai_sdk=True))
    record_fixture("proxy-structured-failure", body, requires=["data-stream-error"])

    assert_standard_frames_are_strict(body)
    assert payloads(body) == [
        {
            "type": "data-stream-error",
            "data": {"statusCode": 409, "errorCode": "INTERRUPT_CONFLICT"},
            "transient": True,
        },
        {"type": "error", "errorText": "Interrupt was claimed."},
        {"type": "finish", "finishReason": "error"},
    ]
    assert data_frames(body).count("[DONE]") == 1


@pytest.mark.asyncio
async def test_proxy_unexpected_upstream_failure_is_sanitized():
    secret = "https://upstream.internal:8443/api?token=hunter2"

    async def failing_upstream():
        raise RuntimeError(f"connection reset by {secret}")
        yield  # pragma: no cover

    body = await _drain(messages_api._build_sse_response(failing_upstream(), ai_sdk=True))
    record_fixture("proxy-unexpected-failure", body)

    assert_standard_frames_are_strict(body)
    assert "hunter2" not in body
    assert payloads(body) == [
        {"type": "error", "errorText": "The request could not be completed."},
        {"type": "finish", "finishReason": "error"},
    ]


async def _web_repair_source():
    yield make_event("answer_preview", sequence=1, data={"operation": "replace", "text": "Draft"})
    yield make_event("answer_preview", sequence=2, data={"operation": "append", "text": " one"})
    yield make_event(
        "answer_preview", sequence=3, data={"operation": "replace", "text": "Repaired answer"}
    )
    yield make_event("message_delta", sequence=4, data={"text": "Final cited answer"})
    yield make_event(
        "complete",
        sequence=5,
        data={"message": {"id": "m-1", "content": "Final cited answer"}},
    )


@pytest.mark.asyncio
async def test_web_repair_fixture_keeps_the_draft_for_a_vanilla_sdk_consumer(monkeypatch):
    """Replacement is application behavior: the SDK alone keeps the draft."""
    direct = await direct_wire(_web_repair_source, state=_state(stream_web_answer_v1=True))
    proxied = await proxy_wire(direct, monkeypatch)
    requires = ["data-answer-preview", "data-answer-reconcile"]
    record_fixture("web-repair-direct", direct, requires=requires, sdk_text="Draft one")
    record_fixture("web-repair-proxied", proxied, requires=requires, sdk_text="Draft one")

    assert_standard_frames_are_strict(direct)
    assert payloads(proxied) == payloads(direct)
    reconcile = [p for p in payloads(direct) if p["type"] == "data-answer-reconcile"]
    assert [p["data"]["content"] for p in reconcile] == ["Final cited answer"]


@pytest.mark.asyncio
async def test_web_failure_fixture_discards_the_draft_before_the_error():
    async def source():
        yield make_event("answer_preview", sequence=1, data={"operation": "replace", "text": "D"})
        raise RuntimeError("provider failure")

    direct = await direct_wire(source, state=_state(stream_web_answer_v1=True))
    record_fixture("web-failure-direct", direct, requires=["data-answer-reconcile"])

    order = types(direct)
    assert order.index("data-answer-reconcile") < order.index("error")


@pytest.mark.asyncio
async def test_partial_eof_fixture_is_an_explicit_incomplete_error(monkeypatch):
    async def source():
        yield make_event("message_delta", sequence=1, data={"text": "partial"})

    direct = await direct_wire(source)
    # Drop the server's [DONE]: the upstream connection ended early.
    truncated = direct.replace("data: [DONE]\n\n", "").split('data: {"type":"data-stream')[0]
    proxied = await proxy_wire(truncated, monkeypatch)
    record_fixture("partial-eof-direct", direct, requires=["data-stream-error", "error"])
    record_fixture("partial-eof-proxied", proxied, requires=["data-stream-error", "error"])

    for wire in (direct, proxied):
        frames = payloads(wire)
        assert {"type": "finish", "finishReason": "error"} in frames
        incomplete = [f for f in frames if f["type"] == "data-stream-error"]
        assert [f["data"]["errorCode"] for f in incomplete] == ["stream_incomplete"]
        assert data_frames(wire).count("[DONE]") == 1


def test_legacy_invalid_frames_are_recorded_as_rejected_fixtures():
    """The SDK must keep rejecting the frames this backend used to send."""
    legacy = {
        "legacy-heartbeat": 'data: {"type":"heartbeat"}\n\ndata: [DONE]\n\n',
        "legacy-tool-render": (
            'data: {"type":"tool-output-available","toolCallId":"c-1",'
            '"output":"Result","render":{"type":"text","text":"Result"}}\n\n'
            "data: [DONE]\n\n"
        ),
        "legacy-structured-error": (
            'data: {"type":"error","errorText":"Conflict","statusCode":409,'
            '"errorCode":"conflict"}\n\ndata: [DONE]\n\n'
        ),
    }
    for name, wire in legacy.items():
        record_fixture(name, wire, expect="reject")
    assert all(wire.endswith("data: [DONE]\n\n") for wire in legacy.values())
