from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from app.ai.research_budget import ResearchBudget
from app.ai.web_research.contracts import (
    ProviderImageCandidate,
    ProviderSource,
    ResearchRequest,
    ResearchScope,
)
from app.ai.web_research.providers import ProviderFailure, ProviderResolver
from app.ai.web_research.service import ProviderHealthRegistry, WebResearchService

SCOPE = ResearchScope(conversation_id="c", user_id="u", logical_turn_id="t")
REQUEST = ResearchRequest(query="release notes", objective="find current release", mode="quick")


class SequenceTextProvider:
    name = "primary"
    health_key = "primary:key-a"

    def __init__(self, values: list[object]) -> None:
        self.values = list(values)
        self.calls = 0

    async def search(self, _request, *, query_index: int):
        self.calls += 1
        value = self.values.pop(0)
        if isinstance(value, BaseException):
            raise value
        return tuple(value)


class ImageFailure:
    name = "brave"
    health_key = "brave:key-a"

    async def search(self, _request):
        raise ProviderFailure("timeout", provider=self.name, retryable=True)


@pytest.mark.asyncio
async def test_retry_then_fallback_preserves_one_source_registry() -> None:
    failures = [
        ProviderFailure("rate_limited", provider="primary", retryable=True),
        ProviderFailure("rate_limited", provider="primary", retryable=True),
    ]
    primary = SequenceTextProvider(failures)
    fallback = SequenceTextProvider(
        [[ProviderSource(provider="fallback", url="https://docs.test/a", rank=1, query_index=1)]]
    )
    fallback.name = "fallback"
    fallback.health_key = "fallback:key-b"
    service = WebResearchService(
        resolver=ProviderResolver(text=(primary, fallback)),
        now=lambda: datetime(2026, 9, 15, tzinfo=timezone.utc),
        retry_backoff=lambda _attempt: None,
    )

    bundle = await service.new_session(SCOPE, ResearchBudget(), mode="quick").search(REQUEST)

    assert [source.source_id for source in bundle.sources] == ["S1"]
    assert [failure.code for failure in bundle.failures] == ["rate_limited", "rate_limited"]
    assert bundle.providers_used == ("primary", "fallback")


@pytest.mark.asyncio
async def test_image_failure_does_not_discard_text_success() -> None:
    text = SequenceTextProvider(
        [[ProviderSource(provider="primary", url="https://docs.test/a", rank=1, query_index=1)]]
    )
    service = WebResearchService(
        resolver=ProviderResolver(text=(text,), images=(ImageFailure(),)),
        now=lambda: datetime(2026, 9, 15, tzinfo=timezone.utc),
        retry_backoff=lambda _attempt: None,
    )
    request = REQUEST.model_copy(
        update={"visual_intent": "figure", "image_query": "release interface"}
    )

    bundle = await service.new_session(SCOPE, ResearchBudget(), mode="quick").search(request)

    assert bundle.sources
    assert not bundle.images
    assert any(failure.operation == "image_search" for failure in bundle.failures)


def test_health_is_partitioned_by_provider_configuration() -> None:
    now = datetime(2026, 9, 15, tzinfo=timezone.utc)
    health = ProviderHealthRegistry(
        failure_threshold=1,
        cooldown=timedelta(minutes=1),
        now=lambda: now,
    )

    health.record_failure("tavily:key-a")

    assert health.is_open("tavily:key-a")
    assert not health.is_open("tavily:key-b")


@pytest.mark.asyncio
async def test_valid_image_sources_are_admitted_before_text_sources() -> None:
    class ImageProvider:
        name = "brave"
        health_key = "brave:key"

        async def search(self, _request):
            return (
                ProviderImageCandidate(
                    provider="brave",
                    image_url="https://images.test/a.jpg",
                    source_url="https://image-source.test/a",
                    rank=1,
                ),
            )

    text = SequenceTextProvider(
        [
            [
                ProviderSource(
                    provider="primary",
                    url="https://text-source.test/a",
                    rank=1,
                    query_index=1,
                )
            ]
        ]
    )
    service = WebResearchService(
        resolver=ProviderResolver(text=(text,), images=(ImageProvider(),)),
        now=lambda: datetime(2026, 9, 15, tzinfo=timezone.utc),
    )
    request = REQUEST.model_copy(update={"visual_intent": "figure", "image_query": "a"})

    bundle = await service.new_session(SCOPE, ResearchBudget(), mode="quick").search(request)

    # Image pages first, each class against its own quota. Text used to be
    # admitted first and return its full budget, which left the image cohort
    # nothing and discarded every candidate before the model saw one.
    assert [str(source.url) for source in bundle.sources] == [
        "https://image-source.test/a",
        "https://text-source.test/a",
    ]


@pytest.mark.asyncio
async def test_open_resolves_source_ids_in_the_same_registry() -> None:
    class Opener:
        name = "extract"
        health_key = "extract:key"

        def __init__(self) -> None:
            self.urls: list[str] = []

        async def open(self, urls, _question, *, query_index: int):
            self.urls = list(urls)
            return (
                ProviderSource(
                    provider="extract",
                    url=urls[0],
                    snippet="Focused release evidence.",
                    rank=1,
                    query_index=query_index,
                ),
            )

    text = SequenceTextProvider(
        [[ProviderSource(provider="primary", url="https://docs.test/a", rank=1, query_index=1)]]
    )
    opener = Opener()
    service = WebResearchService(
        resolver=ProviderResolver(text=(text,), openers=(opener,)),
        now=lambda: datetime(2026, 9, 15, tzinfo=timezone.utc),
    )
    session = service.new_session(SCOPE, ResearchBudget(), mode="quick")
    await session.search(REQUEST)

    bundle = await session.open(["S1"], "What changed?")

    assert opener.urls == ["https://docs.test/a"]
    assert bundle.sources[0].source_id == "S1"
    assert bundle.sources[0].status == "opened"
    assert bundle.sources[0].snippet == "Focused release evidence."


@pytest.mark.asyncio
async def test_parallel_search_calls_are_serialized_per_session() -> None:
    class ConcurrentProvider:
        name = "primary"
        health_key = "primary:key"

        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0

        async def search(self, request, *, query_index: int):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0)
            self.active -= 1
            return (
                ProviderSource(
                    provider=self.name,
                    url=f"https://docs.test/{query_index}",
                    rank=1,
                    query_index=query_index,
                ),
            )

    provider = ConcurrentProvider()
    session = WebResearchService(resolver=ProviderResolver(text=(provider,))).new_session(
        SCOPE, ResearchBudget(), mode="quick"
    )

    first, second = await asyncio.gather(
        session.search(REQUEST),
        session.search(REQUEST.model_copy(update={"query": "security notes"})),
    )

    assert provider.max_active == 1
    assert (first.operation_index, second.operation_index) == (1, 2)


@pytest.mark.asyncio
async def test_later_text_search_does_not_invalidate_visual_evidence() -> None:
    text = SequenceTextProvider(
        [
            [ProviderSource(provider="primary", url="https://docs.test/a", rank=1, query_index=1)],
            [ProviderSource(provider="primary", url="https://docs.test/b", rank=1, query_index=2)],
        ]
    )
    service = WebResearchService(resolver=ProviderResolver(text=(text,)))
    session = service.new_session(SCOPE, ResearchBudget(), mode="quick")

    await session.search(
        REQUEST.model_copy(update={"visual_intent": "figure", "image_query": "release"})
    )
    bundle = await session.search(
        REQUEST.model_copy(update={"query": "security notes", "visual_intent": "none"})
    )

    assert bundle.visual_intent == "figure"


@pytest.mark.asyncio
async def test_open_rejects_private_urls_before_provider_call() -> None:
    class Opener:
        name = "extract"
        health_key = "extract:key"

        def __init__(self) -> None:
            self.calls = 0

        async def open(self, urls, question, *, query_index: int):
            self.calls += 1
            return ()

    opener = Opener()
    session = WebResearchService(resolver=ProviderResolver(openers=(opener,))).new_session(
        SCOPE, ResearchBudget(), mode="quick"
    )

    bundle = await session.open(["https://127.0.0.1/private"], "Inspect")

    assert opener.calls == 0
    assert bundle.failures[0].code == "invalid_source"


class FreshTextProvider:
    """Five brand-new, on-topic results every search, as Tavily really behaves."""

    name = "tavily"
    health_key = "tavily:key"

    def __init__(self) -> None:
        self.calls = 0

    async def search(self, _request, *, query_index: int):
        self.calls += 1
        return tuple(
            ProviderSource(
                provider="tavily",
                url=f"https://search{self.calls}-{rank}.test/article",
                title=f"Result {self.calls}.{rank}",
                snippet="relevant evidence",
                rank=rank,
                query_index=query_index,
            )
            for rank in range(1, 6)
        )


def _text_request(query: str) -> ResearchRequest:
    return ResearchRequest(query=query, objective="identify the current roster", mode="quick")


@pytest.mark.asyncio
async def test_an_exhausted_text_quota_is_named_not_reported_as_an_empty_result() -> None:
    """An empty delta must not read as "the provider found nothing".

    The text quota is per turn. Once it is full every later search admits
    nothing, and reporting that as ``status=success, sources=[], failures=[]``
    is indistinguishable from a search that genuinely found nothing -- so the
    model rewords and searches again, and again, and finally answers with no
    citation because it believes it has no evidence.
    """

    text = FreshTextProvider()
    service = WebResearchService(
        resolver=ProviderResolver(text=(text,)),
        now=lambda: datetime(2026, 9, 22, tzinfo=timezone.utc),
    )
    session = service.new_session(SCOPE, ResearchBudget(), mode="quick")

    first = await session.search(_text_request("current T1 League of Legends roster"))
    second = await session.search(_text_request("T1 2026 starting lineup announcement"))

    assert len(first.operation_source_ids) == 5
    assert first.failures == ()
    assert [failure.code for failure in second.failures] == ["source_quota_exhausted"]
    assert second.operation_source_ids == ()
    # The wasted provider round trip is the other half of the cost: a search
    # that cannot admit anything must not spend one.
    assert text.calls == 1


@pytest.mark.asyncio
async def test_a_saturated_text_quota_still_lets_an_image_search_run() -> None:
    """Only the text half is exhausted; a visual search can still contribute."""

    text = FreshTextProvider()

    class OneImage:
        name = "brave"
        health_key = "brave:key"

        async def search(self, _request):
            return (
                ProviderImageCandidate(
                    provider="brave",
                    image_url="https://cdn.test/photo.png",
                    source_url="https://gallery.test/page",
                    rank=1,
                    width=1920,
                    height=1080,
                    confidence="high",
                    source_domain="gallery.test",
                ),
            )

    service = WebResearchService(
        resolver=ProviderResolver(text=(text,), images=(OneImage(),)),
        now=lambda: datetime(2026, 9, 22, tzinfo=timezone.utc),
    )
    session = service.new_session(SCOPE, ResearchBudget(), mode="quick")

    await session.search(_text_request("current T1 League of Legends roster"))
    visual = await session.search(
        ResearchRequest(
            query="T1 official team photograph",
            objective="show the team",
            mode="quick",
            visual_intent="figure",
            image_query="T1 official team photo",
        )
    )

    assert "source_quota_exhausted" in [failure.code for failure in visual.failures]
    assert visual.operation_source_ids == ("S6",), "the image page still gets admitted"
    assert text.calls == 2, "the text provider still runs; images share the operation"
