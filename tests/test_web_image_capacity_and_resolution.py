"""Image candidates must survive a full text search, and publish at full size.

These three properties were all silently false: a text search that filled the
source budget starved every image page out of the registry, the adapter chose
the provider's 500px proxy over the original, and neither loss was reported.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.ai.research_budget import ResearchBudget
from app.ai.web_research.contracts import (
    ProviderImageCandidate,
    ProviderSource,
    ResearchRequest,
    ResearchScope,
)
from app.ai.web_research.providers import BraveImageSearchProvider, ProviderResolver
from app.ai.web_research.service import WebResearchService
from app.services.web_image_service import FetchedWebImage, WebImageRejected

FIXTURES = Path(__file__).parent / "fixtures" / "web_research"


def _brave(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class _Tool:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    async def ainvoke(self, _args: dict) -> str:
        return json.dumps(self.payload)


class SaturatingText:
    """Returns exactly the source budget, as Tavily does for max_results."""

    name = "tavily"
    health_key = "tavily:test"

    def __init__(self, count: int) -> None:
        self.count = count

    async def search(self, _request, *, query_index: int):
        return tuple(
            ProviderSource(
                provider="tavily",
                url=f"https://news{index}.test/article",
                title=f"Article {index}",
                snippet="text evidence",
                rank=index,
                query_index=query_index,
            )
            for index in range(1, self.count + 1)
        )


class UnrelatedImages:
    """Image pages that share no host with the text results, as in production."""

    name = "brave"
    health_key = "brave:test"

    def __init__(self, count: int = 6) -> None:
        self.count = count

    async def search(self, _request):
        return tuple(
            ProviderImageCandidate(
                provider="brave",
                image_url=f"https://cdn{index}.test/original.png",
                preview_url=f"https://proxy.test/thumb{index}.png",
                source_url=f"https://gallery{index}.test/page",
                title=f"Image {index}",
                rank=index,
            )
            for index in range(1, self.count + 1)
        )


class RecordingImageService:
    max_bytes = 5_000_000

    def __init__(self, *, failing: set[str] | None = None) -> None:
        self.failing = failing or set()
        self.fetched: list[str] = []
        self.registered: list[str] = []

    async def fetch_url(self, url: str, *, provider: str, max_bytes: int | None = None):
        self.fetched.append(url)
        if url in self.failing:
            raise WebImageRejected("upstream_status")
        return _png(url, width=1600, height=900)

    async def register(self, **kwargs):
        self.registered.append(str(kwargs.get("upstream_url")))
        return SimpleNamespace(id=uuid4(), **kwargs)

    async def release_references(self, ids, **_scope):
        return None


def _png(seed: str, *, width: int, height: int) -> FetchedWebImage:
    """Distinct bytes per URL, so the digest de-duplication does not collapse them."""

    from hashlib import sha256
    from io import BytesIO

    from PIL import Image

    tint = sha256(seed.encode()).digest()
    image = Image.new("RGB", (width, height), color=(tint[0], tint[1], tint[2]))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return FetchedWebImage(
        content=buffer.getvalue(), media_type="image/png", width=width, height=height
    )


def _scope() -> ResearchScope:
    return ResearchScope(
        conversation_id=str(uuid4()), user_id=str(uuid4()), logical_turn_id=str(uuid4())
    )


async def _run(
    *,
    text_count: int,
    mode: str = "quick",
    visual_intent: str = "figure",
    image_query: str | None = "the subject",
    image_service: RecordingImageService | None = None,
    image_count: int = 6,
):
    service = WebResearchService(
        resolver=ProviderResolver(
            text=(SaturatingText(text_count),), images=(UnrelatedImages(image_count),)
        ),
        image_service=image_service or RecordingImageService(),
        now=lambda: datetime(2026, 9, 16, tzinfo=timezone.utc),
    )
    session = service.new_session(_scope(), ResearchBudget(), mode=mode)
    bundle = await session.search(
        ResearchRequest(
            query="a concrete subject",
            objective="show the subject",
            mode=mode,
            visual_intent=visual_intent,
            image_query=image_query,
        )
    )
    return session, bundle


@pytest.mark.asyncio
async def test_a_full_text_search_does_not_starve_image_candidates() -> None:
    """The regression: five Tavily results used to consume all five slots."""

    _session, bundle = await _run(text_count=5)

    assert len(bundle.sources) == 5 + 4, "image pages need their own capacity"
    assert [image.candidate_id for image in bundle.images] == ["I1", "I2", "I3", "I4"]


@pytest.mark.asyncio
async def test_agentic_mode_also_reserves_image_capacity() -> None:
    _session, bundle = await _run(text_count=8, mode="agentic")

    assert len(bundle.images) == 4


@pytest.mark.asyncio
async def test_gallery_intent_admits_six_image_pages() -> None:
    _session, bundle = await _run(text_count=5, visual_intent="gallery", image_count=8)

    assert len(bundle.images) == 6


@pytest.mark.asyncio
async def test_over_supply_is_counted_but_is_not_a_failure() -> None:
    """Providers always return more than the intent's slots; that is not an alarm."""

    _session, bundle = await _run(text_count=5, image_count=8)

    assert bundle.omitted_image_count == 4
    assert bundle.failures == ()
    assert bundle.status == "success"


@pytest.mark.asyncio
async def test_losing_every_candidate_is_reported() -> None:
    class UnusablePages(UnrelatedImages):
        async def search(self, _request):
            return tuple(
                candidate.model_copy(update={"source_url": "https://localhost/page"})
                for candidate in await super().search(_request)
            )

    service = WebResearchService(
        resolver=ProviderResolver(text=(SaturatingText(5),), images=(UnusablePages(6),)),
        image_service=RecordingImageService(),
        now=lambda: datetime(2026, 9, 16, tzinfo=timezone.utc),
    )
    session = service.new_session(_scope(), ResearchBudget(), mode="quick")

    bundle = await session.search(
        ResearchRequest(
            query="a concrete subject",
            objective="show the subject",
            mode="quick",
            visual_intent="figure",
            image_query="the subject",
        )
    )

    assert bundle.images == ()
    assert bundle.omitted_image_count == 6
    assert [failure.code for failure in bundle.failures] == ["image_source_capacity"]


@pytest.mark.asyncio
async def test_visual_intent_without_an_image_query_is_reported() -> None:
    _session, bundle = await _run(text_count=5, image_query=None)

    assert [failure.code for failure in bundle.failures] == ["image_query_missing"]
    assert bundle.images == ()


@pytest.mark.asyncio
async def test_publication_fetches_the_original_not_the_proxy_thumbnail() -> None:
    image_service = RecordingImageService()

    _session, _bundle = await _run(text_count=5, image_service=image_service)

    assert all(url.startswith("https://cdn") for url in image_service.registered)
    assert not any(url.startswith("https://proxy.test") for url in image_service.registered)


@pytest.mark.asyncio
async def test_an_unfetchable_original_falls_back_to_the_provider_preview() -> None:
    image_service = RecordingImageService(failing={"https://cdn1.test/original.png"})

    _session, bundle = await _run(text_count=5, image_service=image_service, image_count=1)

    assert "https://proxy.test/thumb1.png" in image_service.registered
    assert len(bundle.images) == 1


@pytest.mark.asyncio
async def test_model_sees_a_downscaled_preview_not_the_full_publication_bytes() -> None:
    session, _bundle = await _run(text_count=5, image_count=1)

    blocks = session.model_evidence_blocks(supports_vision=True)
    prepared = session.prepared_images["I1"]

    assert len(prepared.preview) < len(prepared.content)
    assert prepared.record.width == 1600, "the published record keeps full resolution"
    assert blocks[1]["image_url"]["url"].startswith(f"data:{prepared.preview_mime};base64,")


@pytest.mark.asyncio
async def test_brave_adapter_prefers_the_original_over_the_proxy() -> None:
    request = ResearchRequest(
        query="graphics card",
        objective="show the card",
        visual_intent="figure",
        image_query="graphics card photo",
    )

    images = await BraveImageSearchProvider(_Tool(_brave("brave_images_success.json"))).search(
        request
    )

    assert images[0].image_url.startswith("https://")
    assert "imgs.search.brave.com" not in images[0].image_url
    assert images[0].preview_url is not None
    assert "imgs.search.brave.com" in images[0].preview_url


@pytest.mark.asyncio
async def test_brave_adapter_keeps_the_proxy_when_the_original_is_smaller() -> None:
    """Wikipedia hands Brave a 250px thumb; the 500px proxy is the better image."""

    request = ResearchRequest(
        query="release interface",
        objective="show the interface",
        visual_intent="figure",
        image_query="release interface screenshot",
    )

    images = await BraveImageSearchProvider(
        _Tool(_brave("brave_images_small_original.json"))
    ).search(request)

    assert "imgs.search.brave.com" in images[0].image_url
