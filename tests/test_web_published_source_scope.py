"""What a reader is shown is not what grounding may resolve.

The registry keeps every image-cohort page so ``[[source:S#]]`` resolves for
anything the model was offered. Most of those pages are gallery hosts whose
candidate never entered the vision window; publishing them buries the handful
of sources the answer actually rests on.
"""

from __future__ import annotations

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
from app.ai.web_research.providers import ProviderResolver
from app.ai.web_research.service import WebResearchService
from app.services.web_image_service import FetchedWebImage


class Text:
    name = "tavily"
    health_key = "tavily:test"

    def __init__(self, *urls: str) -> None:
        self.urls = urls

    async def search(self, _request, *, query_index: int):
        return tuple(
            ProviderSource(
                provider="tavily",
                url=url,
                title=f"Article {rank}",
                snippet="text evidence",
                rank=rank,
                query_index=query_index,
            )
            for rank, url in enumerate(self.urls, start=1)
        )


class Images:
    name = "brave"
    health_key = "brave:test"

    def __init__(self, *cohorts: tuple[ProviderImageCandidate, ...]) -> None:
        self.cohorts = list(cohorts)
        self.calls = 0

    async def search(self, _request):
        cohort = self.cohorts[min(self.calls, len(self.cohorts) - 1)]
        self.calls += 1
        return cohort


class Opener:
    name = "tavily"
    health_key = "tavily:test"

    async def open(self, urls, question, *, query_index: int):
        return tuple(
            ProviderSource(
                provider="tavily",
                url=url,
                title="Opened page",
                snippet="the deliberate read",
                rank=rank,
                query_index=query_index,
            )
            for rank, url in enumerate(urls, start=1)
        )


class ImageService:
    max_bytes = 5_000_000

    def __init__(self) -> None:
        self.released: list[object] = []

    async def fetch_url(self, url: str, *, provider: str, max_bytes: int | None = None):
        from hashlib import sha256
        from io import BytesIO

        from PIL import Image

        tint = sha256(url.encode()).digest()
        image = Image.new("RGB", (1600, 900), color=(tint[0], tint[1], tint[2]))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return FetchedWebImage(
            content=buffer.getvalue(), media_type="image/png", width=1600, height=900
        )

    async def register(self, **kwargs):
        return SimpleNamespace(id=uuid4(), **kwargs)

    async def release_references(self, ids, **_scope):
        self.released.extend(ids)


def _candidate(name: str, *, width: int, confidence: str, host: str) -> ProviderImageCandidate:
    return ProviderImageCandidate(
        provider="brave",
        image_url=f"https://cdn.test/{name}.png",
        source_url=f"https://{host}/{name}",
        title=name,
        width=width,
        height=int(width * 9 / 16),
        rank=1,
        confidence=confidence,
        source_domain=host,
    )


def _session(*cohorts: tuple[ProviderImageCandidate, ...], text: tuple[str, ...]):
    service = WebResearchService(
        resolver=ProviderResolver(
            text=(Text(*text),), images=(Images(*cohorts),), openers=(Opener(),)
        ),
        image_service=ImageService(),
        now=lambda: datetime(2026, 9, 16, tzinfo=timezone.utc),
    )
    scope = ResearchScope(
        conversation_id=str(uuid4()), user_id=str(uuid4()), logical_turn_id=str(uuid4())
    )
    return service.new_session(scope, ResearchBudget(), mode="quick")


async def _search(session, query: str, *, domains: tuple[str, ...] = ()):
    return await session.search(
        ResearchRequest(
            query=query,
            objective="show the subject",
            mode="quick",
            visual_intent="figure",
            image_query=query,
            include_domains=domains,
        )
    )


def _urls(records) -> set[str]:
    return {str(record.url) for record in records}


@pytest.mark.asyncio
async def test_gallery_pages_stay_in_the_registry_but_out_of_the_published_list() -> None:
    session = _session(
        tuple(
            _candidate(
                f"shot-{index}", width=1600, confidence="medium", host=f"gallery{index}.test"
            )
            for index in range(1, 9)
        ),
        text=("https://news1.test/a", "https://news2.test/b"),
    )

    await _search(session, "a concrete subject")
    published = _urls(session.answer_sources)

    assert len(session.source_registry.records) > len(session.answer_sources)
    assert {"https://news1.test/a", "https://news2.test/b"} <= published
    backing = {
        str(session.source_registry.resolve(prepared.record.source_id).url)
        for prepared in session.prepared_images.values()
    }
    assert {url for url in published if "gallery" in url} == backing


@pytest.mark.asyncio
async def test_an_evicted_image_page_leaves_the_published_list_but_still_grounds() -> None:
    session = _session(
        tuple(
            _candidate(
                f"shot-{index}", width=1600, confidence="medium", host=f"gallery{index}.test"
            )
            for index in range(1, 5)
        ),
        (_candidate("official", width=1920, confidence="high", host="t1.gg"),),
        text=("https://news1.test/a",),
    )

    await _search(session, "a concrete subject")
    before = {p.record.source_id for p in session.prepared_images.values()}
    await _search(session, "the official subject", domains=("t1.gg",))
    after = {p.record.source_id for p in session.prepared_images.values()}
    evicted_id = next(iter(before - after))

    assert evicted_id not in {record.source_id for record in session.answer_sources}
    # Still resolvable, because the model was offered this S# and may cite it.
    assert session.source_registry.resolve(evicted_id) is not None
    resolution = GroundingParser(session).resolve(f"Cited [[source:{evicted_id}]]")
    assert resolution.source_ids == (evicted_id,)
    assert resolution.warnings == ()


@pytest.mark.asyncio
async def test_a_page_both_cohorts_returned_stays_published() -> None:
    """The image cohort merely got there first; it is still a text source."""

    shared = "https://news1.test/a"
    # The image cohort is admitted first, so it reaches this page first and
    # marks it image-only; the text pass must then take it back.
    both = _candidate("shot", width=1600, confidence="medium", host="gallery.test").model_copy(
        update={"source_url": shared}
    )
    session = _session((both,), text=(shared,))

    await _search(session, "a concrete subject")

    assert shared in _urls(session.answer_sources)
    assert len(session.source_registry.records) == 1


@pytest.mark.asyncio
async def test_an_opened_image_page_stays_published_even_with_no_active_image() -> None:
    session = _session(
        tuple(
            _candidate(
                f"shot-{index}", width=1600, confidence="medium", host=f"gallery{index}.test"
            )
            for index in range(1, 9)
        ),
        text=("https://news1.test/a",),
    )

    await _search(session, "a concrete subject")
    inactive = next(
        record
        for record in session.source_registry.records
        if "gallery" in str(record.url)
        and record.source_id
        not in {p.record.source_id for p in session.prepared_images.values()}
    )

    assert inactive.source_id not in {record.source_id for record in session.answer_sources}

    await session.open([inactive.source_id], "What does this page say?")

    assert inactive.source_id in {record.source_id for record in session.answer_sources}


def test_specialists_publish_the_narrowed_view_not_the_whole_registry() -> None:
    specialists = (
        Path(__file__).parents[1] / "app" / "ai" / "workflow" / "specialists.py"
    ).read_text(encoding="utf-8")

    assert "source_registry.records" not in specialists
    assert specialists.count("web_research_session.answer_sources") == 1
    assert specialists.count("web_research_session.published_sources(") == 2
    assert specialists.count("web_research_session.source_registry.import_records") == 1


@pytest.mark.asyncio
async def test_a_cited_image_page_is_published_even_with_no_active_image() -> None:
    """A citation with no entry in the published list is a dangling [1].

    ``model_context`` offers the model every admitted ``S#``, so it can cite a
    gallery page whose candidate never entered the vision window. Grounding
    resolves that citation and renders it as a numbered link, so the reader
    must be given the source it points at.
    """

    session = _session(
        tuple(
            _candidate(
                f"shot-{index}", width=1600, confidence="medium", host=f"gallery{index}.test"
            )
            for index in range(1, 9)
        ),
        text=("https://news1.test/a",),
    )

    await _search(session, "a concrete subject")
    inactive = next(
        record
        for record in session.source_registry.records
        if "gallery" in str(record.url)
        and record.source_id
        not in {p.record.source_id for p in session.prepared_images.values()}
    )

    resolution = GroundingParser(session).resolve(f"Claim [[source:{inactive.source_id}]]")

    assert resolution.source_ids == (inactive.source_id,)
    assert inactive.source_id not in {record.source_id for record in session.answer_sources}
    published = session.published_sources(resolution.source_ids)
    assert inactive.source_id in {record.source_id for record in published}
    # Still narrower than the registry: only the cited page was added back.
    assert len(published) < len(session.source_registry.records)
