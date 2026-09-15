from __future__ import annotations

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
async def test_valid_image_sources_are_admitted_after_text_sources() -> None:
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

    assert [str(source.url) for source in bundle.sources] == [
        "https://text-source.test/a",
        "https://image-source.test/a",
    ]
