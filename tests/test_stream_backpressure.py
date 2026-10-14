"""Streaming buffers are bounded by stream count and event size, not event count.

Each producer/consumer boundary (the AI SDK adapter's source pump and the
sidecar's upstream reader) holds at most one queued event. A slow reader makes
the producer wait instead of buffering the whole answer, cancellation never
waits on a full queue, and a frame larger than the configured ceiling ends the
stream with a sanitized error instead of being written.

All waits are on explicit events or bounded loop turns -- no wall-clock sleeps.
"""

from __future__ import annotations

import asyncio
import contextlib
import json

import pytest

from app.services.event_streaming import ai_sdk_v6
from app.services.event_streaming.ai_sdk_v6 import AISDKV6StreamAdapter, AISDKV6StreamState
from app.services.event_streaming.events import make_event
from client_backend.api import messages as messages_api

_SOURCE_EVENTS = 5_000
# The event the consumer is processing, one queued, one blocked in put().
_MAX_PRODUCED_WHILE_HELD = 3


def _adapter(source, *, heartbeat: float = 30.0) -> AISDKV6StreamAdapter:
    return AISDKV6StreamAdapter(
        source,
        AISDKV6StreamState(message_id="m-1", text_id="t-1", reasoning_id="r-1"),
        heartbeat_interval_seconds=heartbeat,
    )


async def _let_the_loop_run(turns: int = 200) -> None:
    for _ in range(turns):
        await asyncio.sleep(0)


def _other_live_tasks() -> list[asyncio.Task]:
    current = asyncio.current_task()
    return [task for task in asyncio.all_tasks() if task is not current and not task.done()]


def _payloads(frames: list[str]) -> list[dict]:
    return [
        json.loads(line[6:])
        for line in "".join(frames).splitlines()
        if line.startswith("data: ") and line[6:] != "[DONE]"
    ]


# -- adapter source pump ------------------------------------------------------


@pytest.mark.asyncio
async def test_a_held_consumer_stops_the_adapter_producer():
    produced = 0

    async def source():
        nonlocal produced
        for index in range(_SOURCE_EVENTS):
            produced += 1
            yield make_event("message_delta", sequence=index, data={"text": "x"})

    stream = _adapter(source).iter_sse()
    async for frame in stream:
        if '"text-delta"' in frame:
            break  # the consumer now holds: it reads nothing more

    await _let_the_loop_run()

    assert produced <= _MAX_PRODUCED_WHILE_HELD, f"producer ran ahead by {produced} events"
    await stream.aclose()


@pytest.mark.asyncio
async def test_a_slow_consumer_still_receives_every_event_in_order():
    async def source():
        for index in range(50):
            yield make_event("message_delta", sequence=index, data={"text": f"{index},"})
        yield make_event("complete", sequence=99, data={"message": {"id": "m-1"}})

    frames = []
    async for frame in _adapter(source).iter_sse():
        frames.append(frame)
        await asyncio.sleep(0)

    text = "".join(p["delta"] for p in _payloads(frames) if p["type"] == "text-delta")
    assert text == "".join(f"{index}," for index in range(50))


@pytest.mark.asyncio
async def test_aclose_while_the_producer_is_blocked_in_put_closes_the_source():
    closed = asyncio.Event()

    async def source():
        try:
            for index in range(_SOURCE_EVENTS):
                yield make_event("message_delta", sequence=index, data={"text": "x"})
        finally:
            closed.set()

    stream = _adapter(source).iter_sse()
    async for frame in stream:
        if '"text-delta"' in frame:
            break
    await _let_the_loop_run()  # the producer is now blocked in put()

    await asyncio.wait_for(stream.aclose(), timeout=1)

    assert closed.is_set()
    assert _other_live_tasks() == []


@pytest.mark.asyncio
async def test_aclose_while_a_heartbeat_is_due_closes_the_source():
    closed = asyncio.Event()

    async def source():
        try:
            await asyncio.Event().wait()
            yield make_event("complete", sequence=1)  # pragma: no cover
        finally:
            closed.set()

    stream = _adapter(source, heartbeat=0.001).iter_sse()
    async for frame in stream:
        if frame.startswith(":"):
            break

    await asyncio.wait_for(stream.aclose(), timeout=1)

    assert closed.is_set()
    assert _other_live_tasks() == []


@pytest.mark.asyncio
async def test_an_upstream_error_with_a_full_queue_surfaces_once_after_queued_data():
    held = asyncio.Event()

    async def source():
        yield make_event("message_delta", sequence=1, data={"text": "one"})
        yield make_event("message_delta", sequence=2, data={"text": "two"})
        await held.wait()
        raise RuntimeError("provider failure")

    frames = []
    async for frame in _adapter(source).iter_sse():
        frames.append(frame)
        if '"two"' in frame:
            held.set()

    payloads = _payloads(frames)
    deltas = [p["delta"] for p in payloads if p["type"] == "text-delta"]
    errors = [p for p in payloads if p["type"] == "error"]
    assert deltas == ["one", "two"]
    assert len(errors) == 1
    assert [p["type"] for p in payloads].index("error") > [p["type"] for p in payloads].index(
        "text-delta"
    )


@pytest.mark.asyncio
async def test_cancellation_while_the_producer_waits_to_enqueue_does_not_deadlock():
    closed = asyncio.Event()
    first_frame = asyncio.Event()

    async def source():
        try:
            for index in range(_SOURCE_EVENTS):
                yield make_event("message_delta", sequence=index, data={"text": "x"})
        finally:
            closed.set()

    async def consume():
        async with contextlib.aclosing(_adapter(source).iter_sse()) as stream:
            async for frame in stream:
                if '"text-delta"' in frame:
                    first_frame.set()
                    await asyncio.Event().wait()  # a stalled socket write

    task = asyncio.create_task(consume())
    await asyncio.wait_for(first_frame.wait(), timeout=1)
    await _let_the_loop_run()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    assert closed.is_set()
    assert _other_live_tasks() == []


# -- adapter size ceiling -----------------------------------------------------


@pytest.mark.asyncio
async def test_an_oversized_frame_ends_the_stream_with_a_sanitized_error(monkeypatch):
    monkeypatch.setattr(ai_sdk_v6.settings, "stream_max_event_bytes", 1024)
    secret = "s3cr3t-" * 400

    async def source():
        yield make_event("message_delta", sequence=1, data={"text": "small"})
        yield make_event("message_delta", sequence=2, data={"text": secret})
        yield make_event("message_delta", sequence=3, data={"text": "never sent"})

    frames = [frame async for frame in _adapter(source).iter_sse()]
    payloads = _payloads(frames)

    assert all(len(frame.encode("utf-8")) <= 1024 for frame in frames)
    assert "s3cr3t" not in "".join(frames)
    assert "never sent" not in "".join(frames)
    assert {
        "type": "data-stream-error",
        "data": {"statusCode": 413, "errorCode": "stream_event_too_large"},
        "transient": True,
    } in payloads
    assert payloads[-1] == {"type": "finish", "finishReason": "error"}


@pytest.mark.asyncio
async def test_the_ceiling_counts_utf8_bytes_not_characters(monkeypatch):
    monkeypatch.setattr(ai_sdk_v6.settings, "stream_max_event_bytes", 1024)
    wide = "中" * 400  # ~450 characters framed, but ~1,250 UTF-8 bytes

    async def source():
        yield make_event("message_delta", sequence=1, data={"text": wide})

    payloads = _payloads([frame async for frame in _adapter(source).iter_sse()])

    assert not any(p["type"] == "text-delta" for p in payloads)
    assert any(
        p["type"] == "data-stream-error" and p["data"]["errorCode"] == "stream_event_too_large"
        for p in payloads
    )


@pytest.mark.asyncio
async def test_an_inline_preview_at_its_own_budget_fits_the_default_ceiling():
    from app.core.config import settings

    budget = settings.image_stream_preview_max_b64_chars
    data_url = "data:image/png;base64," + "A" * (budget - len("data:image/png;base64,"))

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
                "delivery": {"kind": "inline", "data_url": data_url},
            },
        )
        yield make_event("complete", sequence=2, data={"message": {"id": "m-1"}})

    payloads = _payloads([frame async for frame in _adapter(source).iter_sse()])
    previews = [p for p in payloads if p["type"] == "data-image-preview"]

    assert budget < settings.stream_max_event_bytes
    assert not any(p["type"] == "data-stream-error" for p in payloads)
    assert len(previews) == 1
    assert previews[0]["data"].get("url", "").startswith("data:image/png;base64,")


@pytest.mark.parametrize("value", [0, -1])
def test_stream_max_event_bytes_must_be_positive(value):
    from pydantic import ValidationError

    from app.core.config import Settings

    with pytest.raises(ValidationError):
        Settings(stream_max_event_bytes=value)


def test_stream_max_event_bytes_defaults_to_eight_mebibytes():
    from app.core.config import settings
    from client_backend.core.config import client_settings

    assert settings.stream_max_event_bytes == 8 * 1024 * 1024
    assert client_settings.stream_max_event_bytes == 8 * 1024 * 1024


# -- sidecar proxy boundary ---------------------------------------------------


async def _proxy_frames(upstream, *, stop_after: str | None = None):
    response = messages_api._build_sse_response(upstream, ai_sdk=True)
    body = response.body_iterator
    frames = []
    async for chunk in body:
        frames.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        if stop_after is not None and stop_after in frames[-1]:
            return frames, body
    return frames, body


@pytest.mark.asyncio
async def test_a_held_consumer_stops_the_sidecar_reader():
    produced = 0
    closed = asyncio.Event()

    async def upstream():
        nonlocal produced
        try:
            for index in range(_SOURCE_EVENTS):
                produced += 1
                yield {"type": "text-delta", "id": "t1", "delta": str(index)}
        finally:
            closed.set()

    _, body = await _proxy_frames(upstream(), stop_after='"text-delta"')
    await _let_the_loop_run()

    assert produced <= _MAX_PRODUCED_WHILE_HELD, f"reader ran ahead by {produced} events"
    await asyncio.wait_for(body.aclose(), timeout=1)
    assert closed.is_set()
    assert _other_live_tasks() == []


@pytest.mark.asyncio
async def test_the_sidecar_drains_queued_events_before_reporting_a_failure():
    async def upstream():
        yield {"type": "text-start", "id": "t1"}
        yield {"type": "text-delta", "id": "t1", "delta": "kept"}
        raise RuntimeError("socket reset")

    frames, _ = await _proxy_frames(upstream())
    payloads = _payloads(frames)

    types = [p["type"] for p in payloads]
    assert types.index("text-delta") < types.index("error")
    assert types.count("error") == 1


@pytest.mark.asyncio
async def test_the_sidecar_refuses_an_oversized_outbound_frame(monkeypatch):
    from client_backend.core.config import client_settings

    monkeypatch.setattr(client_settings, "stream_max_event_bytes", 1024)

    async def upstream():
        yield {"type": "text-start", "id": "t1"}
        yield {"type": "text-delta", "id": "t1", "delta": "y" * 4096}

    frames, _ = await _proxy_frames(upstream())
    payloads = _payloads(frames)

    assert all(len(frame.encode("utf-8")) <= 1024 for frame in frames)
    assert not any(p["type"] == "text-delta" for p in payloads)
    assert any(
        p["type"] == "data-stream-error" and p["data"]["errorCode"] == "stream_event_too_large"
        for p in payloads
    )
    assert payloads[-1] == {"type": "finish", "finishReason": "error"}


# -- sidecar inbound line buffer ---------------------------------------------


class _ByteUpstream:
    status_code = 200

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.delivered = 0

    async def aiter_bytes(self):
        for chunk in self._chunks:
            self.delivered += 1
            yield chunk

    async def aread(self):
        return b""


async def _client_over(monkeypatch, upstream: _ByteUpstream):
    from client_backend.services.server_api import ServerAPIClient

    class _Stream:
        async def __aenter__(self):
            return upstream

        async def __aexit__(self, *exc_info):
            return False

    client = ServerAPIClient(base_url="http://test.local", timeout=5)
    client._tokens = type("T", (), {"access_token": "fake"})()
    http_client = await client._get_client()
    monkeypatch.setattr(http_client, "stream", lambda *a, **kw: _Stream())
    return client


@pytest.mark.asyncio
async def test_inbound_lines_split_across_chunks_are_reassembled(monkeypatch):
    upstream = _ByteUpstream(
        [b'data: {"type":"text-de', b'lta","delta":"h\xc3', b'\xa9"}\r\n\r\ndata: [DONE]\n\n']
    )
    client = await _client_over(monkeypatch, upstream)

    events = [event async for event in client.stream_sse("/x", require_done=True)]

    assert events == [{"type": "text-delta", "delta": "hé"}]


@pytest.mark.asyncio
async def test_an_oversized_inbound_line_is_refused_before_it_is_buffered(monkeypatch):
    from client_backend.core.config import client_settings
    from client_backend.services.server_api import ServerStreamEventTooLargeError

    monkeypatch.setattr(client_settings, "stream_max_event_bytes", 1024)
    endless_line = [b'data: {"type":"text-delta","delta":"'] + [b"z" * 512] * 1_000
    upstream = _ByteUpstream(endless_line)
    client = await _client_over(monkeypatch, upstream)

    with pytest.raises(ServerStreamEventTooLargeError):
        [event async for event in client.stream_sse("/x", require_done=True)]

    assert upstream.delivered <= 4, "the reader kept buffering past the ceiling"
