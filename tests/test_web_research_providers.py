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

T1_IMAGE_QUERY = "T1 League of Legends current roster full team official team photo"


class _Tool:
    """Records every call and replays payloads in order.

    The last payload repeats, so a single-payload construction behaves exactly
    like the one-shot double this replaced. An ``Exception`` payload is raised
    rather than returned, which is how a failing supplemental probe is staged.
    """

    def __init__(self, *payloads: object) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    async def ainvoke(self, args: dict) -> str:
        self.calls.append(args)
        payload = self.payloads[min(len(self.calls) - 1, len(self.payloads) - 1)]
        if isinstance(payload, Exception):
            raise payload
        return json.dumps(payload)


def _json(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _normalized():
    return normalize_web_search(
        WebSearchRequest(query="release notes", objective="find current release"),
        now=datetime(2026, 9, 15, tzinfo=timezone.utc),
        configured_max_results=5,
    )


def _t1_request(**overrides) -> ResearchRequest:
    payload = {
        "query": "T1 roster",
        "objective": "find the official team photo",
        "visual_intent": "gallery",
        "image_query": T1_IMAGE_QUERY,
    }
    payload.update(overrides)
    return ResearchRequest(**payload)


def _image(**overrides) -> dict:
    payload = {
        "url": "https://imgs.search.brave.com/thumb/ELIDED",
        "provider": "brave_image_search",
        "result_rank": 1,
        "confidence": "high",
        "source_url": "https://t1.gg/en/news/one",
        "thumbnail_url": "https://imgs.search.brave.com/thumb/ELIDED",
        "original_image_url": "https://cdn.t1.gg/one.jpg",
        "title": "One",
        "source_domain": "t1.gg",
        "width": 1200,
        "height": 675,
        "thumbnail_width": 500,
        "thumbnail_height": 281,
    }
    payload.update(overrides)
    return payload


def _payload(*images: dict) -> dict:
    return {"provider": "brave_image_search", "images": list(images), "total_results": len(images)}


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

    payload = _json("brave_images_success.json")

    images = await BraveImageSearchProvider(_Tool(payload)).search(request)

    assert [image.provider for image in images] == ["brave"] * len(payload["images"])
    assert [image.source_url for image in images] == [
        raw["source_url"] for raw in payload["images"]
    ]
    assert [image.rank for image in images] == [1, 2, 3]


@pytest.mark.asyncio
async def test_brave_forwards_locale_and_enforces_domain_restrictions() -> None:
    """The model supplied both; neither used to reach the provider."""

    tool = _Tool(_json("brave_images_mixed_domains.json"), _json("brave_images_t1_official.json"))

    images = await BraveImageSearchProvider(tool).search(
        # Deliberately written as a URL, because that is what a model sends.
        _t1_request(locale="vi-VN", include_domains=("https://T1.gg/",))
    )

    assert tool.calls[0] == {
        "query": T1_IMAGE_QUERY,
        "count": 10,
        "safesearch": "strict",
        "country": "VN",
        "search_lang": "vi",
    }
    assert tool.calls[1]["query"].endswith(" site:t1.gg")
    assert {image.source_domain for image in images} == {"t1.gg", "www.t1.gg"}
    assert all("reddit.com" not in image.source_url for image in images)
    assert all("dotesports.com" not in image.source_url for image in images)


@pytest.mark.asyncio
async def test_brave_preserves_provider_quality_metadata() -> None:
    tool = _Tool(_json("brave_images_mixed_domains.json"), _json("brave_images_t1_official.json"))

    images = await BraveImageSearchProvider(tool).search(
        _t1_request(include_domains=("t1.gg",))
    )
    official = next(image for image in images if image.source_domain == "www.t1.gg")

    assert official.confidence == "high"
    assert (official.width, official.height) == (1920, 1080)
    assert official.rank == 1


@pytest.mark.asyncio
async def test_brave_makes_at_most_one_probe_against_the_first_allowed_domain() -> None:
    tool = _Tool(_json("brave_images_mixed_domains.json"), _json("brave_images_t1_official.json"))

    await BraveImageSearchProvider(tool).search(
        _t1_request(
            include_domains=("t1.gg", "lolesports.com", "x.com", "fourth.test"),
        )
    )

    assert len(tool.calls) == 2
    assert tool.calls[1]["query"] == f"{T1_IMAGE_QUERY} site:t1.gg"


@pytest.mark.asyncio
async def test_brave_accepts_a_subdomain_of_an_allowed_host() -> None:
    tool = _Tool(
        _payload(
            _image(source_url="https://shop.t1.gg/merch", source_domain="shop.t1.gg"),
            _image(source_url="https://nott1.gg/page", source_domain="nott1.gg", result_rank=2),
        )
    )

    images = await BraveImageSearchProvider(tool).search(_t1_request(include_domains=("t1.gg",)))

    assert [image.source_domain for image in images] == ["shop.t1.gg"]


@pytest.mark.asyncio
async def test_brave_deduplicates_repeats_preserving_first_provider_order() -> None:
    """The broad call and the probe overlap; the overlap must not double up."""

    tool = _Tool(_json("brave_images_mixed_domains.json"), _json("brave_images_t1_official.json"))

    images = await BraveImageSearchProvider(tool).search(_t1_request(include_domains=("t1.gg",)))
    keys = [(image.source_url, image.image_url) for image in images]

    assert len(keys) == len(set(keys))
    assert images[0].source_url == "https://t1.gg/en/news/2026-roster-announcement"


@pytest.mark.asyncio
async def test_a_failed_probe_does_not_erase_already_allowed_results() -> None:
    tool = _Tool(
        _json("brave_images_mixed_domains.json"),
        ProviderFailure("rate_limited", provider="brave", retryable=True),
    )

    images = await BraveImageSearchProvider(tool).search(_t1_request(include_domains=("t1.gg",)))

    assert [image.source_domain for image in images] == ["t1.gg"]


@pytest.mark.asyncio
async def test_a_first_call_failure_still_raises() -> None:
    tool = _Tool(ProviderFailure("rate_limited", provider="brave", retryable=True))

    with pytest.raises(ProviderFailure):
        await BraveImageSearchProvider(tool).search(_t1_request(include_domains=("t1.gg",)))


@pytest.mark.asyncio
async def test_without_include_domains_the_adapter_makes_exactly_one_call() -> None:
    tool = _Tool(_json("brave_images_mixed_domains.json"))

    images = await BraveImageSearchProvider(tool).search(_t1_request())

    assert len(tool.calls) == 1
    assert "site:" not in tool.calls[0]["query"]
    assert len(images) == 4


@pytest.mark.asyncio
async def test_a_full_allowed_result_set_skips_the_probe() -> None:
    tool = _Tool(
        _payload(
            *(
                _image(
                    source_url=f"https://t1.gg/en/news/{index}",
                    original_image_url=f"https://cdn.t1.gg/{index}.jpg",
                    result_rank=index,
                )
                for index in range(1, 11)
            )
        )
    )

    images = await BraveImageSearchProvider(tool).search(_t1_request(include_domains=("t1.gg",)))

    assert len(tool.calls) == 1
    assert len(images) == 10


@pytest.mark.asyncio
async def test_an_unparseable_locale_forwards_no_locale_arguments() -> None:
    tool = _Tool(_json("brave_images_mixed_domains.json"))

    await BraveImageSearchProvider(tool).search(_t1_request(locale="klingon"))

    assert "country" not in tool.calls[0]
    assert "search_lang" not in tool.calls[0]


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
