from __future__ import annotations

import asyncio

import pytest

from app.ai.selected_image_sink import offer_selected_images, selected_image_sink
from app.ai.tool_execution import _attach_rich_candidates_to_artifact


def test_selected_images_are_collected_in_order():
    with selected_image_sink() as sink:
        offer_selected_images([{"id": "image:selected:a"}])
        offer_selected_images([{"id": "image:selected:b"}])

    assert [item["id"] for item in sink] == [
        "image:selected:a",
        "image:selected:b",
    ]


def test_selected_images_outside_a_sink_are_dropped_silently():
    offer_selected_images([{"id": "image:selected:orphan"}])  # must not raise


def test_nested_selected_image_sinks_do_not_leak_into_each_other():
    with selected_image_sink() as outer:
        with selected_image_sink() as inner:
            offer_selected_images([{"id": "inner"}])
        offer_selected_images([{"id": "outer"}])

    assert [item["id"] for item in inner] == ["inner"]
    assert [item["id"] for item in outer] == ["outer"]


@pytest.mark.asyncio
async def test_selected_images_from_an_awaited_coroutine_reach_the_sink():
    async def _tool():
        offer_selected_images([{"id": "from-await"}])

    with selected_image_sink() as sink:
        await _tool()

    assert [item["id"] for item in sink] == ["from-await"]


@pytest.mark.asyncio
async def test_selected_images_from_a_child_task_reach_the_sink():
    async def _tool():
        offer_selected_images([{"id": "from-task"}])

    with selected_image_sink() as sink:
        await asyncio.create_task(_tool())

    assert [item["id"] for item in sink] == ["from-task"]


@pytest.mark.asyncio
async def test_concurrent_selected_image_sinks_do_not_leak_into_each_other():
    async def _run(label: str) -> list[dict[str, object]]:
        with selected_image_sink() as sink:
            offer_selected_images([{"id": f"{label}-1"}])
            await asyncio.sleep(0)
            offer_selected_images([{"id": f"{label}-2"}])
        return sink

    sink_a, sink_b = await asyncio.gather(_run("a"), _run("b"))

    assert [item["id"] for item in sink_a] == ["a-1", "a-2"]
    assert [item["id"] for item in sink_b] == ["b-1", "b-2"]


def test_selected_images_are_attached_to_the_artifact():
    artifact: dict = {}

    _attach_rich_candidates_to_artifact(
        artifact,
        raw_result=None,
        result_text="{}",
        render=None,
        tool_call_id="call-1",
        tool_name="web_research",
        selected_images=[{"id": "image:selected:a", "type": "image"}],
    )

    assert artifact["_rich_item_candidates"] == [
        {"id": "image:selected:a", "type": "image"}
    ]
