"""In-progress image previews stay bounded without a side queue.

Previews used to go through a ``SubagentEventSink`` whose soft cap dropped
bulky transient frames once it saturated. That queue is gone -- previews ride
the graph's own custom channel now -- but the reason for the cap is not.

Probed against LangGraph 1.2.9: the custom channel does **not** backpressure
its writer. A node emitting 2000 frames finished in 17ms while a deliberately
slow consumer still held the first one, so every frame was buffered. Without a
cap, a slow client makes a long image run buffer every base64 partial.

Only partials are droppable. A final delivery is lossless by definition -- it
carries a reference, not bytes -- and dropping one would lose the image.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.ai.graph import MultiAgentWorkflow
from app.core.config import settings
from app.services.event_streaming.events import build_image_preview_reference_data


def _emitter(monkeypatch, cap: int, written: list[dict]):
    monkeypatch.setattr(settings, "enable_image_streaming", True)
    monkeypatch.setattr(settings, "image_preview_max_partials_per_image", cap)
    monkeypatch.setattr("app.ai.graph._graph_stream_writer", lambda: written.append)

    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    return workflow._build_image_preview_emitter({})


def _partial(item_id: str, seq: int) -> dict:
    return {"item_id": item_id, "status": "partial", "seq": seq, "data_b64": "x" * 64}


def _final(item_id: str) -> dict:
    return build_image_preview_reference_data(
        image_index=0,
        item_id=item_id,
        image_id="img-1",
        url="/chat-images/img-1",
        media_type="image/png",
        seq=99,
    )


def test_partials_are_capped_per_image(monkeypatch):
    written: list[dict] = []
    emit = _emitter(monkeypatch, 3, written)

    for seq in range(10):
        emit(_partial("item-1", seq))

    assert len(written) == 3
    assert [event["seq"] for event in written] == [0, 1, 2]


def test_the_final_delivery_is_never_dropped(monkeypatch):
    """It carries a reference, not bytes, and losing it loses the image."""
    written: list[dict] = []
    emit = _emitter(monkeypatch, 2, written)

    for seq in range(10):
        emit(_partial("item-1", seq))
    emit(_final("item-1"))

    statuses = [event.get("status") for event in written]
    assert statuses.count("partial") == 2
    assert "final" in statuses
    assert written[-1]["delivery"]["kind"] == "reference"


def test_each_image_gets_its_own_budget(monkeypatch):
    """A long first image must not silence the second one entirely."""
    written: list[dict] = []
    emit = _emitter(monkeypatch, 2, written)

    for seq in range(5):
        emit(_partial("item-1", seq))
    for seq in range(5):
        emit(_partial("item-2", seq))

    by_item: dict[str, int] = {}
    for event in written:
        by_item[event["item_id"]] = by_item.get(event["item_id"], 0) + 1
    assert by_item == {"item-1": 2, "item-2": 2}


def test_a_non_partial_status_is_always_written(monkeypatch):
    """``preview_skipped`` and friends carry no bytes and are observability."""
    written: list[dict] = []
    emit = _emitter(monkeypatch, 1, written)

    emit(_partial("item-1", 0))
    emit(_partial("item-1", 1))
    emit({"item_id": "item-1", "status": "preview_skipped", "reason": "too_large"})

    assert [event.get("status") for event in written] == ["partial", "preview_skipped"]


def test_every_frame_is_tagged_for_the_custom_channel(monkeypatch):
    written: list[dict] = []
    emit = _emitter(monkeypatch, 5, written)

    emit(_partial("item-1", 0))

    assert written[0]["type"] == "image_preview"


def test_no_emitter_without_a_run_to_write_into(monkeypatch):
    monkeypatch.setattr(settings, "enable_image_streaming", True)
    monkeypatch.setattr("app.ai.graph._graph_stream_writer", lambda: None)
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)

    assert workflow._build_image_preview_emitter({}) is None


def test_no_emitter_when_streaming_is_disabled(monkeypatch):
    monkeypatch.setattr(settings, "enable_image_streaming", False)
    monkeypatch.setattr("app.ai.graph._graph_stream_writer", lambda: lambda _e: None)
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)

    assert workflow._build_image_preview_emitter({}) is None


def test_the_cap_is_a_validated_positive_setting():
    with pytest.raises(ValueError):
        type(settings)(image_preview_max_partials_per_image=0)


def test_the_probe_that_motivates_this_cap_is_recorded():
    """Guards the reason, not the mechanism.

    If someone concludes the graph channel backpressures after all and removes
    the cap, the docstring naming the probe should be what they read first.
    """
    import inspect

    source = inspect.getsource(MultiAgentWorkflow._build_image_preview_emitter)
    assert "does **not** backpressure" in source


def _unused() -> SimpleNamespace:  # pragma: no cover - import shape only
    return SimpleNamespace()
