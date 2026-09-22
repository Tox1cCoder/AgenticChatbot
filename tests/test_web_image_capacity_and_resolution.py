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
from app.ai.web_research.grounding import GroundingParser
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


class ScriptedImages:
    """Serves one cohort per search, so a session can be searched twice."""

    name = "brave"
    health_key = "brave:test"

    def __init__(self, *cohorts: tuple[ProviderImageCandidate, ...]) -> None:
        self.cohorts = list(cohorts)
        self.calls = 0

    async def search(self, _request):
        cohort = self.cohorts[min(self.calls, len(self.cohorts) - 1)]
        self.calls += 1
        return cohort


def _candidate(
    name: str,
    *,
    width: int,
    height: int,
    rank: int = 1,
    confidence: str | None = None,
    host: str = "gallery.test",
) -> ProviderImageCandidate:
    """A candidate whose declared size is encoded in its image URL.

    The published record's dimensions come from the fetched bytes, not from
    these declared ones, so the fake image service reads them back out of the
    URL to keep the two halves honest.
    """

    return ProviderImageCandidate(
        provider="brave",
        image_url=f"https://cdn.test/{name}-{width}x{height}.png",
        source_url=f"https://{host}/{name}",
        title=name,
        width=width,
        height=height,
        rank=rank,
        confidence=confidence,
        source_domain=host,
    )


def _declared_size(url: str) -> tuple[int, int]:
    width, _, height = url.rsplit("-", 1)[-1].removesuffix(".png").partition("x")
    return int(width), int(height)


class RecordingImageService:
    max_bytes = 5_000_000

    def __init__(self, *, failing: set[str] | None = None) -> None:
        self.failing = failing or set()
        self.fetched: list[str] = []
        self.registered: list[str] = []
        self.released: list[object] = []

    async def fetch_url(self, url: str, *, provider: str, max_bytes: int | None = None):
        self.fetched.append(url)
        if url in self.failing:
            raise WebImageRejected("upstream_status")
        if "-" in url and url.endswith(".png"):
            try:
                width, height = _declared_size(url)
            except ValueError:
                width, height = 1600, 900
        else:
            width, height = 1600, 900
        return _png(url, width=width, height=height)

    async def register(self, **kwargs):
        self.registered.append(str(kwargs.get("upstream_url")))
        return SimpleNamespace(id=uuid4(), **kwargs)

    async def release_references(self, ids, **_scope):
        self.released.extend(ids)


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


def _scripted_session(
    *cohorts: tuple[ProviderImageCandidate, ...],
    text_count: int = 1,
    visual_intent: str = "figure",
    image_service: RecordingImageService | None = None,
):
    service = WebResearchService(
        resolver=ProviderResolver(
            text=(SaturatingText(text_count),), images=(ScriptedImages(*cohorts),)
        ),
        image_service=image_service or RecordingImageService(),
        now=lambda: datetime(2026, 9, 16, tzinfo=timezone.utc),
    )
    session = service.new_session(_scope(), ResearchBudget(), mode="quick")
    return session, visual_intent


async def _search(session, visual_intent: str, *, query: str, domains: tuple[str, ...] = ()):
    """One search. A distinct scope keeps the budget from refusing a rerun.

    ``ResearchBudget.reserve_search`` refuses a near-duplicate query unless the
    scope tuple differs, so a second search that only rewords the first is
    refused with ``duplicate_query`` for reasons that have nothing to do with
    the catalog.
    """

    return await session.search(
        ResearchRequest(
            query=query,
            objective="show the subject",
            mode="quick",
            visual_intent=visual_intent,
            image_query=query,
            include_domains=domains,
        )
    )


def _cards_and_photo() -> tuple[ProviderImageCandidate, ...]:
    """The reported T1 shape: small roster cards ahead of the real photo.

    Every candidate carries a confidence because Brave grades every result.
    A cohort that mixes graded and ungraded candidates is not a shape the
    provider produces, and ordering one against the other proves nothing.
    """

    return (
        _candidate("card-one", width=140, height=140, rank=1, confidence="medium"),
        _candidate("card-two", width=140, height=140, rank=2, confidence="medium"),
        _candidate("card-three", width=140, height=140, rank=3, confidence="medium"),
        _candidate("team-photo", width=1200, height=675, rank=4, confidence="high"),
    )


@pytest.mark.asyncio
async def test_a_full_text_search_does_not_starve_image_candidates() -> None:
    """The regression: five Tavily results used to consume all five slots."""

    session, bundle = await _run(text_count=5)
    urls = {str(source.url) for source in bundle.sources}

    assert all(f"https://news{index}.test/article" in urls for index in range(1, 6))
    assert all(f"https://gallery{index}.test/page" in urls for index in range(1, 7))
    assert len(session.source_registry.records) == 5 + 6
    assert [image.candidate_id for image in bundle.images] == ["I1", "I2", "I3", "I4"]


@pytest.mark.asyncio
async def test_a_larger_photo_outranks_small_cards_that_arrived_first() -> None:
    """Provider rank is the last tie-breaker, not the first signal."""

    session, intent = _scripted_session(_cards_and_photo())

    bundle = await _search(session, intent, query="the subject")

    assert (bundle.images[0].width, bundle.images[0].height) == (1200, 675)
    assert list(session.prepared_images) == ["I1", "I2", "I3", "I4"]


@pytest.mark.asyncio
async def test_a_worse_later_cohort_does_not_displace_the_stronger_photo() -> None:
    session, intent = _scripted_session(
        _cards_and_photo(),
        (
            _candidate("thumb-one", width=100, height=100, rank=1, confidence="low"),
            _candidate("thumb-two", width=100, height=100, rank=2, confidence="low"),
        ),
    )

    await _search(session, intent, query="the subject")
    photo_id = next(
        candidate_id
        for candidate_id, prepared in session.prepared_images.items()
        if prepared.record.width == 1200
    )
    bundle = await _search(session, intent, query="a different subject", domains=("gallery.test",))

    assert photo_id in session.prepared_images
    assert (bundle.images[0].width, bundle.images[0].height) == (1200, 675)
    assert all(image.width != 100 for image in bundle.images)


@pytest.mark.asyncio
async def test_a_better_later_cohort_enters_a_full_window_and_evicts_the_weakest() -> None:
    image_service = RecordingImageService()
    session, intent = _scripted_session(
        _cards_and_photo(),
        (_candidate("official", width=1920, height=1080, rank=1, confidence="high", host="t1.gg"),),
        image_service=image_service,
    )

    await _search(session, intent, query="the subject")
    before = dict(session.prepared_images)
    evicted_id = "I4"
    evicted_reference = before[evicted_id].reference_id

    bundle = await _search(session, intent, query="the official subject", domains=("t1.gg",))

    assert (bundle.images[0].width, bundle.images[0].height) == (1920, 1080)
    # Retained candidates keep the IDs the model was already shown.
    assert list(session.prepared_images) == ["I5", "I1", "I2", "I3"]
    assert evicted_id not in session.prepared_images
    assert image_service.released == [evicted_reference]


@pytest.mark.asyncio
async def test_an_evicted_candidate_id_is_never_rebound_to_different_bytes() -> None:
    session, intent = _scripted_session(
        _cards_and_photo(),
        (_candidate("official", width=1920, height=1080, rank=1, confidence="high", host="t1.gg"),),
    )

    await _search(session, intent, query="the subject")
    retired = session.prepared_images["I4"].record.digest
    await _search(session, intent, query="the official subject", domains=("t1.gg",))

    assert "I4" not in session.prepared_images
    assert all(prepared.record.digest != retired for prepared in session.prepared_images.values())


@pytest.mark.asyncio
async def test_an_evicted_candidate_cannot_be_selected_through_grounding() -> None:
    session, intent = _scripted_session(
        _cards_and_photo(),
        (_candidate("official", width=1920, height=1080, rank=1, confidence="high", host="t1.gg"),),
    )

    await _search(session, intent, query="the subject")
    await _search(session, intent, query="the official subject", domains=("t1.gg",))

    resolution = GroundingParser(session).resolve("Look [[image:I4]] [[source:S1]]")

    assert resolution.selected_image_ids == ()
    assert resolution.rich_items == ()
    assert [warning["code"] for warning in resolution.warnings] == ["unknown_image_id"]


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
    """Providers always return more than the intent's slots; that is not an alarm.

    Twelve exceeds ``max_candidate_pool`` (8), so four are omitted at the
    catalog boundary. The four that sit in the catalog outside the active
    window are *retained for a later window*, not omitted.
    """

    _session, bundle = await _run(text_count=5, image_count=12)

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


@pytest.mark.asyncio
async def test_a_candidate_label_names_its_dimensions_and_domain() -> None:
    """Only server-normalized facts. Retrieved titles stay untrusted evidence."""

    session, intent = _scripted_session(
        (_candidate("official", width=1920, height=1080, rank=1, confidence="high", host="t1.gg"),)
    )

    await _search(session, intent, query="the subject")
    blocks = session.model_evidence_blocks(supports_vision=True)

    assert blocks[0]["text"].startswith(
        "Image candidate I1; source S1; 1920x1080; domain t1.gg."
    )
