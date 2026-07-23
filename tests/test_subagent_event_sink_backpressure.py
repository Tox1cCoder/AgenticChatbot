import pytest

from app.services.event_streaming.events import make_event
from app.services.event_streaming.subagents import SubagentEventSink


def _preview(seq):
    return make_event("image_preview", sequence=0, data={"status": "partial", "seq": seq})


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
