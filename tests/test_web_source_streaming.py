from __future__ import annotations

import json

import pytest

from app.services.event_streaming.ai_sdk_projection import ensure_leading_text_part
from app.services.event_streaming.ai_sdk_v6 import AISDKV6StreamAdapter, AISDKV6StreamState
from app.services.event_streaming.events import make_event
from app.services.event_streaming.internal_sse import legacy_event_from_v3

SOURCE = {
    "source_id": "S1",
    "title": "Release notes",
    "url": "https://example.test/release",
    "snippet": "Released today.",
    "status": "search_result",
}


def test_internal_stream_projects_canonical_sources() -> None:
    event = make_event(
        "sources", sequence=1, data={"operation": "upsert", "sources": [SOURCE]}
    )

    assert legacy_event_from_v3(event) == {
        "type": "sources",
        "operation": "upsert",
        "sources": [SOURCE],
    }


@pytest.mark.asyncio
async def test_ai_sdk_stream_projects_native_source_url() -> None:
    async def no_events():
        if False:
            yield None

    adapter = AISDKV6StreamAdapter(
        no_events,
        AISDKV6StreamState("message", "text", "reasoning"),
    )
    event = make_event("sources", sequence=1, data={"sources": [SOURCE]})

    chunks = [chunk async for chunk in adapter._map_event(event)]
    payload = json.loads(chunks[0].removeprefix("data: ").strip())

    assert payload == {
        "type": "source-url",
        "sourceId": "S1",
        "url": "https://example.test/release",
        "title": "Release notes",
    }


def test_history_uses_the_same_source_identity() -> None:
    message = {
        "content": "Answer [1](https://example.test/release).",
        "metadata": {"web_sources_version": 1, "web_sources": [SOURCE]},
    }

    ensure_leading_text_part(message)

    source_part = next(part for part in message["parts"] if part["type"] == "source-url")
    assert source_part["sourceId"] == "S1"
    assert source_part["url"] == SOURCE["url"]
