"""Turn-scoped orchestration for provider-neutral web research."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any, TypeVar

from app.ai.research_budget import ResearchBudget
from app.ai.web_query_contract import WebSearchRequest, normalize_web_search

from .contracts import (
    ProviderImageCandidate,
    ProviderSource,
    ResearchFailure,
    ResearchMode,
    ResearchRequest,
    ResearchScope,
    WebEvidenceBundle,
)
from .policy import ResearchLimits
from .providers import ProviderFailure, ProviderResolver
from .source_registry import SourceRegistry

T = TypeVar("T")


class ProviderHealthRegistry:
    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        cooldown: timedelta = timedelta(seconds=30),
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.failure_threshold = max(1, int(failure_threshold))
        self.cooldown = cooldown
        self.now = now or (lambda: datetime.now(timezone.utc))
        self._failures: dict[str, int] = {}
        self._opened_at: dict[str, datetime] = {}

    def is_open(self, health_key: str) -> bool:
        opened = self._opened_at.get(health_key)
        if opened is None:
            return False
        if self.now() - opened >= self.cooldown:
            self._opened_at.pop(health_key, None)
            self._failures[health_key] = 0
            return False
        return True

    def record_failure(self, health_key: str) -> None:
        count = self._failures.get(health_key, 0) + 1
        self._failures[health_key] = count
        if count >= self.failure_threshold:
            self._opened_at[health_key] = self.now()

    def record_success(self, health_key: str) -> None:
        self._failures.pop(health_key, None)
        self._opened_at.pop(health_key, None)


class ResearchResultCache:
    def __init__(self, *, now: Callable[[], datetime] | None = None) -> None:
        self.now = now or (lambda: datetime.now(timezone.utc))
        self._values: dict[str, tuple[datetime, Any]] = {}

    def get(self, key: str) -> Any | None:
        value = self._values.get(key)
        if value is None:
            return None
        expires_at, result = value
        if expires_at <= self.now():
            self._values.pop(key, None)
            return None
        return result

    def put(self, key: str, value: Any, *, ttl: timedelta) -> None:
        self._values[key] = (self.now() + ttl, value)


class WebResearchService:
    def __init__(
        self,
        *,
        resolver: ProviderResolver,
        now: Callable[[], datetime] | None = None,
        health: ProviderHealthRegistry | None = None,
        cache: ResearchResultCache | None = None,
        retry_backoff: Callable[[int], Awaitable[None] | None] | None = None,
    ) -> None:
        self.resolver = resolver
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.health = health or ProviderHealthRegistry(now=self.now)
        self.cache = cache or ResearchResultCache(now=self.now)
        self.retry_backoff = retry_backoff or (lambda _attempt: None)

    def new_session(
        self,
        scope: ResearchScope,
        budget: ResearchBudget,
        *,
        mode: ResearchMode,
    ) -> WebResearchSession:
        return WebResearchSession(self, scope, budget, mode=mode)


class WebResearchSession:
    def __init__(
        self,
        service: WebResearchService,
        scope: ResearchScope,
        budget: ResearchBudget,
        *,
        mode: ResearchMode,
    ) -> None:
        self.service = service
        self.scope = scope
        self.budget = budget
        self.mode = mode
        self.limits = ResearchLimits.for_mode(mode)
        self.source_registry = SourceRegistry(max_sources=self.limits.max_sources)
        self._operation_index = 0
        self.pending_provider_images: tuple[ProviderImageCandidate, ...] = ()

    async def search(self, request: ResearchRequest) -> WebEvidenceBundle:
        self._operation_index += 1
        operation_index = self._operation_index
        normalized = normalize_web_search(
            WebSearchRequest(
                query=request.query,
                objective=request.objective,
                freshness=request.freshness,
                start_date=request.start_date,
                end_date=request.end_date,
                locale=request.locale,
                include_domains=list(request.include_domains),
                max_results=self.limits.max_sources,
            ),
            now=self.service.now(),
            configured_max_results=self.limits.max_sources,
        )
        search_scope = (
            normalized.freshness,
            normalized.start_date,
            normalized.end_date,
            normalized.include_domains,
        )
        refusal = self.budget.reserve_search(normalized.query, scope=search_scope)
        if refusal is not None:
            return self._bundle(
                request,
                operation_index,
                failures=(
                    ResearchFailure(
                        operation="search",
                        provider="server",
                        code=refusal,
                        retryable=False,
                    ),
                ),
            )

        text_task = self._call_chain(
            self.service.resolver.text,
            operation="search",
            invoke=lambda provider: provider.search(normalized, query_index=operation_index),
        )
        image_task = (
            self._call_chain(
                self.service.resolver.images,
                operation="image_search",
                invoke=lambda provider: provider.search(request),
            )
            if request.visual_intent != "none" and request.image_query
            else None
        )
        try:
            text_result = await text_task
            image_result = await image_task if image_task is not None else ((), (), ())
        except BaseException:
            self.budget.release_search(normalized.query, scope=search_scope)
            raise

        text_records, text_failures, text_providers = text_result
        image_records, image_failures, image_providers = image_result
        self.budget.record_search(normalized.query, "canonical_web_evidence", scope=search_scope)

        self.source_registry.admit(tuple(text_records))
        image_sources = tuple(
            ProviderSource(
                provider=image.provider,
                url=image.source_url,
                title=image.title,
                snippet=image.description,
                rank=image.rank,
                query_index=operation_index,
                published_at=image.published_at,
            )
            for image in image_records
        )
        self.source_registry.admit(image_sources)
        self.pending_provider_images = tuple(
            image for image in image_records if self.source_registry.resolve(image.source_url)
        )
        return self._bundle(
            request,
            operation_index,
            failures=(*text_failures, *image_failures),
            providers=tuple(dict.fromkeys((*text_providers, *image_providers))),
        )

    async def _call_chain(
        self,
        providers: Sequence[Any],
        *,
        operation: str,
        invoke: Callable[[Any], Awaitable[T]],
    ) -> tuple[T | tuple[Any, ...], tuple[ResearchFailure, ...], tuple[str, ...]]:
        failures: list[ResearchFailure] = []
        used: list[str] = []
        for provider in providers:
            name = str(provider.name)
            health_key = str(provider.health_key)
            used.append(name)
            if self.service.health.is_open(health_key):
                failures.append(
                    ResearchFailure(
                        operation=operation,
                        provider=name,
                        code="circuit_open",
                        retryable=True,
                    )
                )
                continue
            for attempt in range(2):
                try:
                    value = await invoke(provider)
                    self.service.health.record_success(health_key)
                    if value:
                        return value, tuple(failures), tuple(used)
                    break
                except ProviderFailure as exc:
                    failures.append(
                        ResearchFailure(
                            operation=operation,
                            provider=exc.provider,
                            code=exc.code,
                            retryable=exc.retryable,
                        )
                    )
                    if exc.retryable:
                        self.service.health.record_failure(health_key)
                    if not exc.retryable or attempt == 1:
                        break
                    delayed = self.service.retry_backoff(attempt)
                    if inspect.isawaitable(delayed):
                        await delayed
        return (), tuple(failures), tuple(used)

    def _bundle(
        self,
        request: ResearchRequest,
        operation_index: int,
        *,
        failures: tuple[ResearchFailure, ...] = (),
        providers: tuple[str, ...] = (),
    ) -> WebEvidenceBundle:
        sources = self.source_registry.records
        status = "success" if sources and not failures else "partial" if sources else "failed"
        return WebEvidenceBundle(
            status=status,
            mode=self.mode,
            visual_intent=request.visual_intent,
            operation_index=operation_index,
            sources=sources,
            failures=failures,
            providers_used=providers,
        )


__all__ = [
    "ProviderHealthRegistry",
    "ResearchResultCache",
    "WebResearchService",
    "WebResearchSession",
]
