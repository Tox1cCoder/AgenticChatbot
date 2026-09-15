from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.ai.web_query_contract import WebSearchRequest, normalize_web_search
from app.ai.web_research.contracts import ResearchRequest
from app.ai.web_research.providers import (
    BraveImageSearchProvider,
    ProviderFailure,
    TavilyTextSearchProvider,
)

FIXTURES = Path(__file__).parent / "fixtures" / "web_research"


class _Tool:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    async def ainvoke(self, _args: dict) -> str:
        return json.dumps(self.payload)


def _json(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _normalized():
    return normalize_web_search(
        WebSearchRequest(query="release notes", objective="find current release"),
        now=datetime(2026, 9, 15, tzinfo=timezone.utc),
        configured_max_results=5,
    )


@pytest.mark.asyncio
async def test_tavily_adapter_drops_raw_content_and_assigns_query_index() -> None:
    records = await TavilyTextSearchProvider(_Tool(_json("tavily_search_success.json"))).search(
        _normalized(), query_index=2
    )

    assert records[0].provider == "tavily"
    assert records[0].query_index == 2
    assert records[0].snippet == "Version 9.2 was released."
    assert "raw_content" not in records[0].model_dump_json()


@pytest.mark.asyncio
async def test_brave_adapter_returns_provider_neutral_candidates() -> None:
    request = ResearchRequest(
        query="release notes",
        objective="find current release",
        visual_intent="figure",
        image_query="version 9.2 interface",
    )

    images = await BraveImageSearchProvider(
        _Tool(_json("brave_images_success.json"))
    ).search(request)

    assert images[0].provider == "brave"
    assert images[0].source_url == "https://docs.example.test/release"


@pytest.mark.asyncio
async def test_provider_cancellation_is_not_normalized_as_a_failure() -> None:
    class Blocking:
        async def ainvoke(self, _args: dict) -> str:
            await asyncio.Event().wait()
            return "{}"

    task = asyncio.create_task(
        TavilyTextSearchProvider(Blocking()).search(_normalized(), query_index=1)
    )
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_malformed_payload_has_a_bounded_failure() -> None:
    with pytest.raises(ProviderFailure) as raised:
        await TavilyTextSearchProvider(_Tool(["not", "an", "object"])).search(
            _normalized(), query_index=1
        )

    assert raised.value.code == "invalid_response"
    assert raised.value.retryable is False
