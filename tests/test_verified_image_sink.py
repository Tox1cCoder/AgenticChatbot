from __future__ import annotations

import asyncio

import pytest

from app.ai.tool_execution import _attach_rich_candidates_to_artifact
from app.ai.verified_image_sink import offer_verified_images, verified_image_sink


def test_offers_are_collected_inside_the_sink():
    with verified_image_sink() as sink:
        offer_verified_images([{"id": "image:verified:a"}])
        offer_verified_images([{"id": "image:verified:b"}])

    assert [item["id"] for item in sink] == ["image:verified:a", "image:verified:b"]


def test_offers_outside_a_sink_are_dropped_silently():
    offer_verified_images([{"id": "image:verified:orphan"}])  # must not raise


def test_nested_sinks_do_not_leak_into_each_other():
    with verified_image_sink() as outer:
        with verified_image_sink() as inner:
            offer_verified_images([{"id": "inner"}])
        offer_verified_images([{"id": "outer"}])

    assert [item["id"] for item in inner] == ["inner"]
    assert [item["id"] for item in outer] == ["outer"]


@pytest.mark.asyncio
async def test_offers_from_an_awaited_coroutine_reach_the_sink():
    async def _tool():
        offer_verified_images([{"id": "from-await"}])

    with verified_image_sink() as sink:
        await _tool()

    assert [item["id"] for item in sink] == ["from-await"]


@pytest.mark.asyncio
async def test_offers_from_a_child_task_reach_the_sink():
    async def _tool():
        offer_verified_images([{"id": "from-task"}])

    with verified_image_sink() as sink:
        await asyncio.create_task(_tool())

    assert [item["id"] for item in sink] == ["from-task"]


def test_verified_images_are_attached_to_the_artifact():
    artifact: dict = {}

    _attach_rich_candidates_to_artifact(
        artifact,
        raw_result=None,
        result_text="{}",
        render=None,
        tool_call_id="call-1",
        tool_name="web_research",
        verified_images=[{"id": "image:verified:a", "type": "image"}],
    )

    assert artifact["_rich_item_candidates"] == [{"id": "image:verified:a", "type": "image"}]
