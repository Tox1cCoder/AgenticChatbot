"""The three reported visual failures, locked end to end.

What these prove: the good candidate reaches the vision window at all, and the
literal ``IMAGE TARGET`` line reaches the model with the pixels.

What they deliberately do NOT prove: that the ranking picks the right *kind* of
picture. It cannot -- ordering is quality-only and cannot see subject matter.
``test_ordering_is_quality_only_and_cannot_see_subject_matter`` is the negative
control that keeps that boundary honest and failing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage
from PIL import Image

from app.ai.research_budget import ResearchBudget
from app.ai.web_research.contracts import (
    ProviderImageCandidate,
    ProviderSource,
    ResearchRequest,
    ResearchScope,
)
from app.ai.web_research.grounding import GroundingParser
from app.ai.web_research.model_context import inject_latest_web_evidence
from app.ai.web_research.providers import ProviderResolver
from app.ai.web_research.service import WebResearchService
from app.services.web_image_service import FetchedWebImage


class Text:
    name = "tavily"
    health_key = "tavily:test"

    async def search(self, _request, *, query_index: int):
        return (
            ProviderSource(
                provider="tavily",
                url="https://coverage.test/article",
                title="Background",
                snippet="text evidence",
                rank=1,
                query_index=query_index,
            ),
        )


class Images:
    """One cohort per search, in order."""

    name = "brave"
    health_key = "brave:test"

    def __init__(self, *cohorts: tuple[ProviderImageCandidate, ...]) -> None:
        self.cohorts = list(cohorts)
        self.calls = 0

    async def search(self, _request):
        cohort = self.cohorts[min(self.calls, len(self.cohorts) - 1)]
        self.calls += 1
        return cohort


class ImageService:
    """Bytes whose real dimensions match what the candidate declared.

    The published record's width and height come from the fetched bytes while
    the ranking reads the provider's declared ones. A fake that returns one
    fixed size would let an assertion pass for the wrong reason.
    """

    max_bytes = 5_000_000

    def __init__(self) -> None:
        self.sizes: dict[str, tuple[int, int]] = {}
        self.released: list[object] = []

    def declare(self, candidate: ProviderImageCandidate) -> ProviderImageCandidate:
        self.sizes[candidate.image_url] = (candidate.width or 1, candidate.height or 1)
        return candidate

    async def fetch_url(self, url: str, *, provider: str, max_bytes: int | None = None):
        width, height = self.sizes.get(url, (800, 600))
        tint = sha256(url.encode()).digest()
        image = Image.new("RGB", (width, height), color=(tint[0], tint[1], tint[2]))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return FetchedWebImage(
            content=buffer.getvalue(), media_type="image/png", width=width, height=height
        )

    async def register(self, **kwargs):
        return SimpleNamespace(id=uuid4(), **kwargs)

    async def release_references(self, ids, **_scope):
        self.released.extend(ids)


def _candidate(
    name: str,
    *,
    width: int,
    height: int,
    host: str,
    confidence: str,
    rank: int = 1,
    title: str | None = None,
) -> ProviderImageCandidate:
    return ProviderImageCandidate(
        provider="brave",
        image_url=f"https://cdn.test/{name}.png",
        source_url=f"https://{host}/{name}",
        title=title or name,
        width=width,
        height=height,
        rank=rank,
        confidence=confidence,
        source_domain=host,
    )


def _build(*cohorts: tuple[ProviderImageCandidate, ...]):
    image_service = ImageService()
    for cohort in cohorts:
        for candidate in cohort:
            image_service.declare(candidate)
    service = WebResearchService(
        resolver=ProviderResolver(text=(Text(),), images=(Images(*cohorts),)),
        image_service=image_service,
        now=lambda: datetime(2026, 9, 17, tzinfo=timezone.utc),
    )
    scope = ResearchScope(
        conversation_id=str(uuid4()), user_id=str(uuid4()), logical_turn_id=str(uuid4())
    )
    return service.new_session(scope, ResearchBudget(), mode="quick"), image_service


async def _search(session, *, query: str, domains: tuple[str, ...] = ()):
    return await session.search(
        ResearchRequest(
            query=query,
            objective="show the requested visual",
            mode="quick",
            visual_intent="figure",
            image_query=query,
            include_domains=domains,
        )
    )


def _injected(session) -> str:
    messages = inject_latest_web_evidence(
        [HumanMessage(content="question")], session, supports_vision=True
    )
    parts: list[str] = []
    for message in messages:
        content = message.content
        if isinstance(content, str):
            parts.append(content)
        else:
            parts.extend(str(part.get("text") or "") for part in content if isinstance(part, dict))
    return "\n".join(parts)


T1_QUERY = "full T1 League of Legends team photo"


def _t1_first_cohort() -> tuple[ProviderImageCandidate, ...]:
    return (
        _candidate(
            "card-one", width=140, height=140, host="reddit.com", confidence="medium", rank=1
        ),
        _candidate(
            "card-two", width=140, height=140, host="reddit.com", confidence="medium", rank=2
        ),
        _candidate(
            "card-three", width=140, height=140, host="reddit.com", confidence="low", rank=3
        ),
        _candidate(
            "team-photo",
            width=1200,
            height=675,
            host="dotesports.com",
            confidence="high",
            rank=4,
            title="T1 team photo",
        ),
    )


@pytest.mark.asyncio
async def test_t1_small_cards_then_a_later_official_full_team_photo() -> None:
    """The reported turn: the second, better query used to contribute nothing."""

    official = _candidate(
        "official",
        width=1920,
        height=1080,
        host="www.t1.gg",
        confidence="high",
        rank=1,
        title="T1 official team photo",
    )
    session, _service = _build(_t1_first_cohort(), (official,))

    await _search(session, query="T1 roster cards")
    await _search(session, query=T1_QUERY, domains=("https://T1.gg/",))
    active = [prepared.record for prepared in session.prepared_images.values()]

    assert active[0].source_domain == "www.t1.gg"
    assert (active[0].width, active[0].height) == (1920, 1080)
    assert (
        f"IMAGE TARGET: {T1_QUERY}. Match the requested visual form literally"
        in _injected(session)
    )


@pytest.mark.asyncio
async def test_a_high_confidence_close_up_stays_selectable() -> None:
    """Grace-hair case: a 1024x576 close-up led the provider results."""

    close_up = _candidate(
        "hair-close-up",
        width=1024,
        height=576,
        host="fandom.com",
        confidence="high",
        # Deliberately not the provider's leading result: a test whose good
        # candidate is already rank 1 passes under rank-only ordering too, and
        # so proves nothing about the ranking.
        rank=4,
        title="Grace Ashcroft hair close-up",
    )
    noise = tuple(
        _candidate(
            f"noise-{index}",
            width=300,
            height=300,
            host="shop.test",
            confidence="low",
            rank=index,
        )
        for index in (1, 2, 3, 5)
    )
    session, _service = _build((close_up, *noise))

    await _search(session, query="Grace Ashcroft hair close-up")
    active = [prepared.record for prepared in session.prepared_images.values()]

    assert (active[0].width, active[0].height) == (1024, 576)
    assert (
        "IMAGE TARGET: Grace Ashcroft hair close-up. Match the requested visual form"
        " literally" in _injected(session)
    )


@pytest.mark.asyncio
async def test_a_settings_screenshot_beats_unrelated_product_imagery() -> None:
    """Windows case: the screenshot led, the marketing renders did not."""

    screenshot = _candidate(
        "bluetooth-settings",
        width=1920,
        height=1080,
        host="support.microsoft.com",
        confidence="high",
        rank=5,
        title="Windows 11 Bluetooth settings",
    )
    product = tuple(
        _candidate(
            f"laptop-{index}",
            width=400,
            height=300,
            host="store.test",
            confidence="low",
            rank=index,
        )
        for index in range(1, 5)
    )
    session, _service = _build((screenshot, *product))

    await _search(session, query="Windows 11 Bluetooth settings screenshot")
    active = [prepared.record for prepared in session.prepared_images.values()]

    assert active[0].title == "Windows 11 Bluetooth settings"
    assert (active[0].width, active[0].height) == (1920, 1080)
    assert (
        "IMAGE TARGET: Windows 11 Bluetooth settings screenshot. Match the requested"
        " visual form literally" in _injected(session)
    )


@pytest.mark.asyncio
async def test_an_active_id_publishes_full_resolution_and_an_evicted_one_does_not() -> None:
    official = _candidate(
        "official", width=1920, height=1080, host="www.t1.gg", confidence="high", rank=1
    )
    session, _service = _build(_t1_first_cohort(), (official,))

    await _search(session, query="T1 roster cards")
    before = set(session.prepared_images)
    await _search(session, query=T1_QUERY, domains=("t1.gg",))
    evicted_id = next(iter(before - set(session.prepared_images)))

    leader = next(iter(session.prepared_images.values())).record
    selected = GroundingParser(session).resolve(
        f"Here [[image:{leader.candidate_id}]] [[source:{leader.source_id}]]"
    )
    dropped = GroundingParser(session).resolve(
        f"Here [[image:{evicted_id}]] [[source:{leader.source_id}]]"
    )

    assert [item["payload"]["width"] for item in selected.rich_items] == [1920]
    assert selected.rich_items[0]["payload"]["url"].startswith("/web-images/")
    assert dropped.rich_items == ()
    assert [warning["code"] for warning in dropped.warnings] == ["unknown_image_id"]


@pytest.mark.asyncio
async def test_ordering_is_quality_only_and_cannot_see_subject_matter() -> None:
    """The negative control. Read this before editing it.

    A larger roster *graphic* outranks a smaller team *photo*, and that is
    correct behavior for a function that ranks pixels rather than meaning. If a
    future change makes the photo lead by subject matter, this test should fail
    and be read -- it means a visual taxonomy was introduced, which the design
    rules out.
    """

    graphic = _candidate(
        "roster-graphic",
        width=1920,
        height=1080,
        host="dotesports.com",
        confidence="high",
        rank=2,
        title="T1 2026 roster graphic",
    )
    photo = _candidate(
        "team-photo",
        width=1200,
        height=675,
        host="dotesports.com",
        confidence="high",
        rank=1,
        title="T1 team photo",
    )
    session, _service = _build((graphic, photo))

    await _search(session, query=T1_QUERY)
    active = [prepared.record for prepared in session.prepared_images.values()]
    injected = _injected(session)

    assert active[0].title == "T1 2026 roster graphic"
    # Both reach the model, and the model is told what to look for.
    assert {record.title for record in active} >= {"T1 2026 roster graphic", "T1 team photo"}
    assert "A roster graphic or list of names is not a full team photo" in injected
