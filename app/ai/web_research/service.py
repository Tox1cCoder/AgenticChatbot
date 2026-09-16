"""Turn-scoped orchestration for provider-neutral web research."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, TypeVar
from uuid import UUID

from app.ai.research_budget import ResearchBudget
from app.ai.web_query_contract import WebSearchRequest, normalize_web_search
from app.core.rich_response import (
    GENERIC_IMAGE_ALT_TEXT,
    ImageRichItem,
    RichItemType,
)

from .contracts import (
    ImageCandidateRecord,
    ProviderImageCandidate,
    ProviderSource,
    ResearchFailure,
    ResearchMode,
    ResearchRequest,
    ResearchScope,
    VisualIntent,
    WebEvidenceBundle,
)
from .policy import ResearchLimits
from .providers import ProviderFailure, ProviderResolver
from .source_registry import SourceRegistry

T = TypeVar("T")


@dataclass(frozen=True)
class PreparedImage:
    record: ImageCandidateRecord
    reference_id: UUID
    content: bytes
    rich_item: dict[str, Any]


@dataclass(frozen=True)
class ResearchCloseout:
    selected_candidate_ids: tuple[str, ...]


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
        resolver: ProviderResolver | None = None,
        now: Callable[[], datetime] | None = None,
        health: ProviderHealthRegistry | None = None,
        cache: ResearchResultCache | None = None,
        retry_backoff: Callable[[int], Awaitable[None] | None] | None = None,
        image_service: Any | None = None,
        max_candidate_pool: int = 8,
        max_download_bytes: int = 20 * 1024 * 1024,
        max_model_bytes: int = 8 * 1024 * 1024,
        max_image_concurrency: int = 3,
        pending_image_ttl: timedelta = timedelta(minutes=15),
        metrics: Any | None = None,
    ) -> None:
        self.resolver = resolver or ProviderResolver()
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.health = health or ProviderHealthRegistry(now=self.now)
        self.cache = cache or ResearchResultCache(now=self.now)
        self.retry_backoff = retry_backoff or (lambda _attempt: None)
        self.image_service = image_service
        self.max_candidate_pool = max(1, int(max_candidate_pool))
        self.max_download_bytes = max(1, int(max_download_bytes))
        self.max_model_bytes = max(1, int(max_model_bytes))
        self.max_image_concurrency = max(1, int(max_image_concurrency))
        self.pending_image_ttl = pending_image_ttl
        self.metrics = metrics

    def new_session(
        self,
        scope: ResearchScope,
        budget: ResearchBudget,
        *,
        mode: ResearchMode,
        resolver: ProviderResolver | None = None,
    ) -> WebResearchSession:
        return WebResearchSession(
            self, scope, budget, mode=mode, resolver=resolver or self.resolver
        )


class WebResearchSession:
    def __init__(
        self,
        service: WebResearchService,
        scope: ResearchScope,
        budget: ResearchBudget,
        *,
        mode: ResearchMode,
        resolver: ProviderResolver,
    ) -> None:
        self.service = service
        self.scope = scope
        self.budget = budget
        self.mode = mode
        self.resolver = resolver
        self.limits = ResearchLimits.for_mode(mode)
        self.source_registry = SourceRegistry(max_sources=self.limits.max_sources)
        self._operation_index = 0
        self.pending_provider_images: tuple[ProviderImageCandidate, ...] = ()
        self.prepared_images: dict[str, PreparedImage] = {}
        self.reason_codes: set[str] = set()
        self._image_digests: set[str] = set()
        self._model_image_bytes = 0
        self._downloaded_image_bytes = 0
        self._omitted_image_count = 0
        self._closed = False
        self._opened_urls: set[str] = set()
        self._visual_intent = "none"
        self._operation_lock = asyncio.Lock()

    async def search(self, request: ResearchRequest) -> WebEvidenceBundle:
        async with self._operation_lock:
            bundle = await self._search(request)
        if self.service.metrics is not None:
            with suppress(Exception):
                self.service.metrics.record(
                    operation="search",
                    mode=self.mode,
                    outcome=(bundle.status if bundle.status in {"success", "partial"} else "error"),
                    visual_intent=bundle.visual_intent,
                )
        return bundle

    async def _search(self, request: ResearchRequest) -> WebEvidenceBundle:
        if self._visual_intent == "none" or request.visual_intent == "gallery":
            self._visual_intent = request.visual_intent
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
                operation_index,
                visual_intent=request.visual_intent,
                failures=(
                    ResearchFailure(
                        operation="search",
                        provider="server",
                        code=refusal,
                        retryable=False,
                    ),
                ),
            )

        text_task = asyncio.create_task(
            self._call_chain(
                self.resolver.text,
                operation="search",
                invoke=lambda provider: provider.search(normalized, query_index=operation_index),
            )
        )
        image_task = (
            asyncio.create_task(
                self._call_chain(
                    self.resolver.images,
                    operation="image_search",
                    invoke=lambda provider: provider.search(request),
                )
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
        image_fetch_failures = await self._prepare_images(request)
        return self._bundle(
            operation_index,
            visual_intent=request.visual_intent,
            failures=(*text_failures, *image_failures, *image_fetch_failures),
            providers=tuple(dict.fromkeys((*text_providers, *image_providers))),
        )

    async def open(self, source_ids_or_urls: Sequence[str], question: str) -> WebEvidenceBundle:
        async with self._operation_lock:
            bundle = await self._open(source_ids_or_urls, question)
        if self.service.metrics is not None:
            with suppress(Exception):
                self.service.metrics.record(
                    operation="open",
                    mode=self.mode,
                    outcome=(bundle.status if bundle.status in {"success", "partial"} else "error"),
                    visual_intent=bundle.visual_intent,
                )
        return bundle

    async def _open(self, source_ids_or_urls: Sequence[str], question: str) -> WebEvidenceBundle:
        self._operation_index += 1
        operation_index = self._operation_index
        requested: list[str] = []
        invalid_source = False
        for value in source_ids_or_urls:
            record = self.source_registry.resolve(value)
            if record is not None:
                url = str(record.url)
            else:
                from .source_registry import canonicalize_public_url

                url = canonicalize_public_url(str(value or "").strip()) or ""
                invalid_source = invalid_source or not url
            if url and url not in self._opened_urls and url not in requested:
                requested.append(url)

        remaining = max(0, self.limits.max_page_opens - len(self._opened_urls))
        requested = requested[:remaining]
        if not requested:
            return self._bundle(
                operation_index,
                failures=(
                    ResearchFailure(
                        operation="open",
                        provider="server",
                        code="invalid_source" if invalid_source else "page_open_limit",
                        retryable=False,
                    ),
                ),
            )

        if not self.resolver.openers:
            return self._bundle(
                operation_index,
                failures=(
                    ResearchFailure(
                        operation="open",
                        provider="server",
                        code="provider_unavailable",
                        retryable=False,
                    ),
                ),
            )

        self._opened_urls.update(requested)
        opened, failures, providers = await self._call_chain(
            self.resolver.openers,
            operation="open",
            invoke=lambda provider: provider.open(requested, question, query_index=operation_index),
        )
        self.source_registry.admit(tuple(opened))
        for candidate in opened:
            record = self.source_registry.resolve(candidate.url)
            if record is not None:
                self.source_registry.mark_opened(record.source_id, snippet=candidate.snippet)
        return self._bundle(operation_index, failures=failures, providers=providers)

    async def _prepare_images(self, request: ResearchRequest) -> tuple[ResearchFailure, ...]:
        if self.service.image_service is None or not self.pending_provider_images:
            return ()

        limit = ResearchLimits.for_mode(
            self.mode, visual_intent=self._visual_intent
        ).max_model_images
        remaining = max(0, limit - len(self.prepared_images))
        candidates = self.pending_provider_images[: self.service.max_candidate_pool]
        if not remaining:
            self._omitted_image_count += len(candidates)
            return ()

        download_lock = asyncio.Lock()
        next_candidate = 0
        reserved_bytes = 0
        outcomes: list[tuple[int, ProviderImageCandidate, Any, str | None]] = []
        per_image_limit = max(
            1,
            int(
                getattr(
                    self.service.image_service,
                    "max_bytes",
                    self.service.max_download_bytes,
                )
            ),
        )

        async def fetch_worker() -> None:
            nonlocal next_candidate, reserved_bytes
            while True:
                async with download_lock:
                    remaining_bytes = (
                        self.service.max_download_bytes
                        - self._downloaded_image_bytes
                        - reserved_bytes
                    )
                    if next_candidate >= len(candidates) or remaining_bytes <= 0:
                        return
                    index = next_candidate
                    candidate = candidates[index]
                    next_candidate += 1
                    allowance = min(per_image_limit, remaining_bytes)
                    reserved_bytes += allowance
                image = None
                failure_code = None
                try:
                    image = await self.service.image_service.fetch_url(
                        candidate.image_url,
                        provider=candidate.provider,
                        max_bytes=allowance,
                    )
                except Exception as exc:
                    failure_code = str(getattr(exc, "reason", "fetch_failed"))
                async with download_lock:
                    reserved_bytes -= allowance
                    if image is not None:
                        byte_size = len(image.content)
                        if byte_size > allowance:
                            image = None
                            failure_code = "size"
                        else:
                            self._downloaded_image_bytes += byte_size
                    outcomes.append((index, candidate, image, failure_code))

        await asyncio.gather(
            *(
                fetch_worker()
                for _ in range(min(self.service.max_image_concurrency, len(candidates)))
            )
        )
        self._omitted_image_count += len(candidates) - len(outcomes)
        failures: list[ResearchFailure] = []
        for _index, candidate, fetched, failure_code in sorted(outcomes):
            source = self.source_registry.resolve(candidate.source_url)
            if failure_code is not None or fetched is None or source is None:
                self._omitted_image_count += 1
                failures.append(
                    ResearchFailure(
                        operation="image_fetch",
                        provider=candidate.provider,
                        code=(failure_code or "source_missing")[:64],
                        retryable=False,
                        source_id=source.source_id if source is not None else None,
                    )
                )
                continue

            byte_size = len(fetched.content)
            digest = hashlib.sha256(fetched.content).hexdigest()
            if digest in self._image_digests:
                self._omitted_image_count += 1
                continue
            if (
                self._downloaded_image_bytes > self.service.max_download_bytes
                or self._model_image_bytes + byte_size > self.service.max_model_bytes
                or len(self.prepared_images) >= limit
            ):
                self._omitted_image_count += 1
                continue

            persisted = await self.service.image_service.register(
                conversation_id=UUID(self.scope.conversation_id),
                user_id=UUID(self.scope.user_id),
                upstream_url=candidate.image_url,
                expected_mime=fetched.media_type,
                provider=candidate.provider,
                cached=fetched,
                expires_at=self.service.now() + self.service.pending_image_ttl,
            )
            candidate_id = f"I{len(self.prepared_images) + 1}"
            delivery_url = f"/web-images/{persisted.id}"
            record = ImageCandidateRecord(
                candidate_id=candidate_id,
                source_id=source.source_id,
                delivery_url=delivery_url,
                mime_type=fetched.media_type,
                width=fetched.width,
                height=fetched.height,
                byte_size=byte_size,
                digest=digest,
                title=candidate.title,
                description=candidate.description,
                provider=candidate.provider,
            )
            rich_item = ImageRichItem(
                id=f"image:web:{persisted.id}",
                type=RichItemType.image,
                source="image_search",
                title=candidate.title,
                alt_text=str(
                    candidate.description or candidate.title or GENERIC_IMAGE_ALT_TEXT
                ),
                payload={
                    "url": delivery_url,
                    "mime_type": fetched.media_type,
                    "source_url": str(source.url),
                    "width": fetched.width,
                    "height": fetched.height,
                    "description": candidate.description,
                },
                provenance={
                    "provider": candidate.provider,
                    "source_id": source.source_id,
                },
            ).model_dump(mode="json", exclude_none=True)
            self.prepared_images[candidate_id] = PreparedImage(
                record=record,
                reference_id=persisted.id,
                content=fetched.content,
                rich_item=rich_item,
            )
            self._image_digests.add(digest)
            self._model_image_bytes += byte_size
        return tuple(failures)

    def model_evidence_blocks(self, *, supports_vision: bool) -> list[dict[str, Any]]:
        if not supports_vision:
            if self.prepared_images:
                self.reason_codes.add("answer_model_not_vision_capable")
            return []

        blocks: list[dict[str, Any]] = []
        for candidate_id, prepared in self.prepared_images.items():
            record = prepared.record
            blocks.extend(
                (
                    {
                        "type": "text",
                        "text": (
                            f"Image candidate {candidate_id}; source {record.source_id}. "
                            "Select it only if its visible content supports the answer."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:{record.mime_type};base64,"
                                f"{base64.b64encode(prepared.content).decode('ascii')}"
                            ),
                            "detail": "low",
                        },
                    },
                )
            )
        return blocks

    async def finish(self, selected_candidate_ids: Sequence[str]) -> ResearchCloseout:
        selected = tuple(
            dict.fromkeys(
                candidate_id
                for candidate_id in selected_candidate_ids
                if candidate_id in self.prepared_images
            )
        )
        selected_set = set(selected)
        released_refs = [
            prepared.reference_id
            for candidate_id, prepared in self.prepared_images.items()
            if candidate_id not in selected_set
        ]
        scope = {
            "user_id": UUID(self.scope.user_id),
            "conversation_id": UUID(self.scope.conversation_id),
        }
        if released_refs:
            await self.service.image_service.release_references(released_refs, **scope)
        self._closed = True
        if self.service.metrics is not None:
            with suppress(Exception):
                self.service.metrics.record(
                    operation="finish",
                    mode=self.mode,
                    outcome="selected" if selected else "released",
                    visual_intent=self._visual_intent,
                )
        return ResearchCloseout(selected_candidate_ids=selected)

    async def abort(self) -> None:
        if self._closed or self.service.image_service is None:
            return
        await self.service.image_service.release_references(
            [prepared.reference_id for prepared in self.prepared_images.values()],
            user_id=UUID(self.scope.user_id),
            conversation_id=UUID(self.scope.conversation_id),
        )
        self._closed = True
        if self.service.metrics is not None:
            with suppress(Exception):
                self.service.metrics.record(
                    operation="finish",
                    mode=self.mode,
                    outcome="released",
                    visual_intent=self._visual_intent,
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
        operation_index: int,
        *,
        visual_intent: VisualIntent | None = None,
        failures: tuple[ResearchFailure, ...] = (),
        providers: tuple[str, ...] = (),
    ) -> WebEvidenceBundle:
        if visual_intent is not None and (
            self._visual_intent == "none" or visual_intent == "gallery"
        ):
            self._visual_intent = visual_intent
        sources = self.source_registry.records
        status = "success" if sources and not failures else "partial" if sources else "failed"
        return WebEvidenceBundle(
            status=status,
            mode=self.mode,
            visual_intent=self._visual_intent,
            operation_index=operation_index,
            sources=sources,
            images=tuple(prepared.record for prepared in self.prepared_images.values()),
            failures=failures,
            providers_used=providers,
            omitted_image_count=self._omitted_image_count,
        )


__all__ = [
    "ProviderHealthRegistry",
    "ResearchResultCache",
    "PreparedImage",
    "ResearchCloseout",
    "WebResearchService",
    "WebResearchSession",
]
