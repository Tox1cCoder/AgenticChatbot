import pytest

from app.services.event_streaming.events import (
    build_image_preview_reference_data,
    make_event,
)
from app.services.event_streaming.subagents import SubagentEventSink


def _preview(seq):
    return make_event("image_preview", sequence=0, data={"status": "partial", "seq": seq})


def _partial(item_id, seq):
    return make_event(
        "image_preview",
        sequence=0,
        data={"item_id": item_id, "status": "partial", "seq": seq},
    )


def _final_reference(item_id, seq):
    return make_event(
        "image_preview",
        sequence=0,
        data=build_image_preview_reference_data(
            image_index=0,
            item_id=item_id,
            image_id="img-1",
            url="/chat-images/img-1",
            media_type="image/png",
            seq=seq,
        ),
    )


@pytest.mark.asyncio
async def test_transient_frames_bounded_by_maxsize():
    sink = SubagentEventSink(maxsize=3)
    for i in range(10):
        sink.emit_event(_preview(i))
    assert sink._queue.qsize() <= 3
    assert sink.dropped_transient_count >= 7


@pytest.mark.asyncio
async def test_lifecycle_events_never_dropped():
    sink = SubagentEventSink(maxsize=2)
    for i in range(5):
        sink.emit_event(_preview(i))
    await sink.emit(  # lifecycle 'end' must survive even when saturated
        "subagent_end", task_id="t1", agent_name="worker", status="completed"
    )
    drained = await sink.drain()
    assert any(e.type == "subagent_end" for e in drained)


@pytest.mark.asyncio
async def test_unbounded_when_maxsize_zero():
    sink = SubagentEventSink(maxsize=0)
    for i in range(20):
        sink.emit_event(_preview(i))
    assert sink._queue.qsize() == 20
    assert sink.dropped_transient_count == 0


@pytest.mark.asyncio
async def test_final_reference_never_dropped_under_backpressure():
    """A FINAL image delivered by protected reference is lossless: it survives
    even when the transient queue is already saturated with partials."""
    sink = SubagentEventSink(maxsize=1)
    sink.emit_event(_partial("image-preview-0", seq=1))  # fill the queue
    sink.emit_event(_final_reference("image-preview-0", seq=2))

    drained = await sink.drain()
    finals = [
        e
        for e in drained
        if e.type == "image_preview" and (e.data or {}).get("status") == "final"
    ]
    assert len(finals) == 1, [e.data for e in drained]
    assert finals[0].data["delivery"]["url"] == "/chat-images/img-1"


@pytest.mark.asyncio
async def test_newer_partial_replaces_stale_same_item_under_backpressure():
    """Under backpressure a newer partial for the SAME item evicts the stale
    one (client replaces in place), so the freshest preview wins rather than
    the oldest surviving."""
    sink = SubagentEventSink(maxsize=1)
    sink.emit_event(_partial("image-preview-0", seq=1))
    sink.emit_event(_partial("image-preview-0", seq=2))

    drained = await sink.drain()
    assert [e.data["seq"] for e in drained] == [2]


@pytest.mark.asyncio
async def test_partial_for_distinct_item_coalesced_when_saturated():
    """A partial for a DIFFERENT item cannot evict an unrelated item's frame,
    so it is coalesced (dropped) rather than displacing unrelated work."""
    sink = SubagentEventSink(maxsize=1)
    sink.emit_event(_partial("image-preview-0", seq=1))
    sink.emit_event(_partial("image-preview-1", seq=1))

    drained = await sink.drain()
    assert [e.data["item_id"] for e in drained] == ["image-preview-0"]
    assert sink.dropped_transient_count == 1
