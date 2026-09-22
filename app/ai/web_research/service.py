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
from urllib.parse import urlsplit
from uuid import UUID

from app.ai.research_budget import ResearchBudget
from app.ai.web_query_contract import (
    WebSearchRequest,
    bare_host,
    host_matches,
    normalize_web_search,
)
from app.core.rich_response import (
    GENERIC_IMAGE_ALT_TEXT,
    ImageRichItem,
    RichItemType,
)
from app.services.web_image_service import downscale_for_model

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
    #: A downscaled rendition of ``content``, shown to the answer model instead
    #: of the publication bytes.
    preview: bytes
    preview_mime: str
    #: Identity of the catalog entry this was prepared from, so a recomputed
    #: window can retain it without refetching.
    catalog_key: tuple[str, str]


@dataclass(frozen=True)
class ResearchCloseout:
    selected_candidate_ids: tuple[str, ...]


_CONFIDENCE_PRIORITY = {"high": 3, "medium": 2, "low": 1}

#: Longest edge at which a candidate is worth showing at all. Below it a
#: candidate keeps its place in the catalog but loses the leading bucket.
_ADEQUATE_EDGE = 640


def _image_candidate_priority(
    candidate: ProviderImageCandidate,
    *,
    preferred: bool,
) -> tuple[int, int, int, int, int]:
    """Order candidates by usable quality. Resolution adequacy dominates.

    ``preferred`` sits *below* adequacy on purpose. The provider filters a
    domain-restricted cohort strictly, so every candidate it produced is
    preferred; if the flag led, any candidate from a restricted search would
    outrank every candidate from an unrestricted one, and a 200x150 logo from
    the named site would displace a 4000x3000 photo. Provenance breaks ties
    between comparable images; it does not buy a small one a window slot.

    This is a *quality* ordering and nothing more. It cannot tell a team photo
    from a roster infographic, and a large graphic will outrank a smaller
    photo. Matching the requested visual form is the answer model's judgement,
    made from the pixels plus the explicit IMAGE TARGET line.
    """

    confidence = _CONFIDENCE_PRIORITY.get(str(candidate.confidence or "").lower(), 0)
    width = candidate.width or 0
    height = candidate.height or 0
    adequate = int(max(width, height) >= _ADEQUATE_EDGE)
    area = width * height
    return (-adequate, -confidence, -int(preferred), -area, candidate.rank)


def _fetch_order(candidate: ProviderImageCandidate) -> tuple[str, ...]:
    """The renditions to try, best first, without repeating one URL."""

    urls = [candidate.image_url]
    if candidate.preview_url and candidate.preview_url != candidate.image_url:
        urls.append(candidate.preview_url)
    return tuple(urls)


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
        max_candidate_catalog: int = 24,
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
        self.max_candidate_catalog = max(1, int(max_candidate_catalog))
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
        # Two explicit quotas, never one shared pool: text results used to be
        # admitted first and return their full budget, which discarded every
        # image page before the model saw one.
        self._text_source_capacity = self.limits.max_sources
        self._image_catalog_capacity = self.service.max_candidate_catalog
        self.source_registry = SourceRegistry(
            max_sources=self._text_source_capacity + self._image_catalog_capacity
        )
        self._admitted_text_sources = 0
        self._admitted_image_sources = 0
        self._image_only_source_ids: set[str] = set()
        self._omitted_source_count = 0
        self._operation_index = 0
        #: Every candidate this session has seen, best-effort ranked on demand.
        #: Insertion-ordered on purpose: equal priority tuples fall back to the
        #: order the providers returned them in.
        self._candidate_catalog: dict[tuple[str, str], ProviderImageCandidate] = {}
        self._preferred_candidate_keys: set[tuple[str, str]] = set()
        self._prepared_by_key: dict[tuple[str, str], PreparedImage] = {}
        self._next_candidate_index = 1
        self._latest_image_query: str | None = None
        self._latest_image_objective: str | None = None
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
        wants_images = request.visual_intent != "none"
        gate_failures: tuple[ResearchFailure, ...] = ()
        if wants_images and not request.image_query:
            # Silence here read as "the provider found nothing". Name it, so the
            # model can retry with the subject it forgot to state.
            wants_images = False
            gate_failures = (
                ResearchFailure(
                    operation="image_search",
                    provider="server",
                    code="image_query_missing",
                    retryable=False,
                ),
            )
        image_task = (
            asyncio.create_task(
                self._call_chain(
                    self.resolver.images,
                    operation="image_search",
                    invoke=lambda provider: provider.search(request),
                )
            )
            if wants_images
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

        # Image pages first, each class against its own quota. A text search
        # that returns its full budget can no longer spend the capacity the
        # image cohort needs, and vice versa.
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
        image_room = max(0, self._image_catalog_capacity - self._admitted_image_sources)
        image_admitted = self.source_registry.admit(image_sources[:image_room])
        self._admitted_image_sources += len(image_admitted)
        self._image_only_source_ids.update(record.source_id for record in image_admitted)

        text_room = max(0, self._text_source_capacity - self._admitted_text_sources)
        text_admitted = self.source_registry.admit(tuple(text_records)[:text_room])
        self._admitted_text_sources += len(text_admitted)
        for candidate in text_records:
            record = self.source_registry.resolve(candidate.url)
            if record is None:
                self._omitted_source_count += 1
            else:
                # A page both a text search and an image cohort returned is a
                # text source: the image cohort merely got there first.
                self._image_only_source_ids.discard(record.source_id)

        capacity_failures = self._merge_candidates(request, image_records)
        image_fetch_failures = await self._prepare_images()
        return self._bundle(
            operation_index,
            operation_source_ids=tuple(
                record.source_id for record in (*image_admitted, *text_admitted)
            ),
            visual_intent=request.visual_intent,
            failures=(
                *text_failures,
                *image_failures,
                *gate_failures,
                *capacity_failures,
                *image_fetch_failures,
            ),
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
        opened_ids: list[str] = []
        for candidate in opened:
            record = self.source_registry.resolve(candidate.url)
            if record is None:
                continue
            self.source_registry.mark_opened(record.source_id, snippet=candidate.snippet)
            # Not admit()'s delta, which by definition excludes a page an
            # earlier search already admitted -- and that is the usual case
            # here, since the model opens an S# it was shown.
            if record.source_id not in opened_ids:
                opened_ids.append(record.source_id)
        return self._bundle(
            operation_index,
            operation_source_ids=tuple(opened_ids),
            failures=failures,
            providers=providers,
        )

    def _merge_candidates(
        self,
        request: ResearchRequest,
        image_records: Sequence[ProviderImageCandidate],
    ) -> tuple[ResearchFailure, ...]:
        """Fold one cohort into the session catalog, bounded on both ends.

        Over-supply stays a counter -- providers always return more than the
        intent's slots -- while a cohort that lost *every* candidate stays a
        failure the model is told about.
        """

        if not image_records:
            return ()
        self._latest_image_query = request.image_query
        self._latest_image_objective = request.objective
        allowed = tuple(
            dict.fromkeys(host for host in map(bare_host, request.include_domains) if host)
        )

        pooled = tuple(image_records)[: self.service.max_candidate_pool]
        self._omitted_image_count += len(image_records) - len(pooled)
        entered = 0
        for candidate in pooled:
            source = self.source_registry.resolve(candidate.source_url)
            if source is None:
                self._omitted_image_count += 1
                continue
            key = (str(source.url), candidate.image_url)
            if key not in self._candidate_catalog:
                if len(self._candidate_catalog) >= self._image_catalog_capacity:
                    self._omitted_image_count += 1
                    continue
                self._candidate_catalog[key] = candidate
            entered += 1
            host = candidate.source_domain or urlsplit(candidate.source_url).hostname or ""
            if allowed and any(host_matches(host, entry) for entry in allowed):
                # The preference belongs to the request that produced this
                # candidate. Applying it globally would let a later search
                # naming another domain promote an unrelated older candidate.
                self._preferred_candidate_keys.add(key)

        if entered:
            return ()
        return (
            ResearchFailure(
                operation="image_search",
                provider=image_records[0].provider,
                code="image_source_capacity",
                retryable=False,
            ),
        )

    def _ranked_catalog(self) -> list[tuple[tuple[str, str], ProviderImageCandidate]]:
        """The whole catalog, best first. Equal tuples keep insertion order."""

        return sorted(
            self._candidate_catalog.items(),
            key=lambda item: _image_candidate_priority(
                item[1], preferred=item[0] in self._preferred_candidate_keys
            ),
        )

    async def _prepare_images(self) -> tuple[ResearchFailure, ...]:
        """Recompute the active vision window over the whole catalog.

        Already-prepared candidates are retained without refetching and newly
        active ones are fetched; only once the replacements exist are evicted
        references released, so a failed fetch costs the improvement rather
        than the image the model already had.
        """

        if self.service.image_service is None:
            return ()
        limit = ResearchLimits.for_mode(
            self.mode, visual_intent=self._visual_intent
        ).max_model_images
        if limit <= 0:
            await self._retain_active(())
            return ()

        failures: list[ResearchFailure] = []
        unusable: set[tuple[str, str]] = set()
        fetch_budget = self.service.max_candidate_pool
        while fetch_budget > 0:
            window = [key for key, _candidate in self._ranked_catalog() if key not in unusable][
                :limit
            ]
            missing = [key for key in window if key not in self._prepared_by_key]
            if not missing:
                break
            batch = missing[:fetch_budget]
            fetch_budget -= len(batch)
            batch_failures, rejected = await self._fetch_candidates(batch)
            failures.extend(batch_failures)
            if not rejected:
                break
            # A candidate that was attempted and did not become usable never
            # will be, so step past it to the next ranked one.
            unusable |= rejected

        active = [key for key, _candidate in self._ranked_catalog() if key in self._prepared_by_key]
        await self._retain_active(tuple(active[:limit]))
        return tuple(failures)

    async def _fetch_candidates(
        self, keys: Sequence[tuple[str, str]]
    ) -> tuple[list[ResearchFailure], set[tuple[str, str]]]:
        """Fetch, validate, and prepare one batch. Returns failures and losers.

        The loser set is every key that did not become a prepared image, for
        any reason -- unreachable, oversized, duplicate, or never attempted
        because the download budget ran out.
        """

        candidates = [(key, self._candidate_catalog[key]) for key in keys]
        download_lock = asyncio.Lock()
        next_candidate = 0
        reserved_bytes = 0
        outcomes: list[
            tuple[int, tuple[str, str], ProviderImageCandidate, Any, str | None, str]
        ] = []
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
                    key, candidate = candidates[index]
                    next_candidate += 1
                    allowance = min(per_image_limit, remaining_bytes)
                    reserved_bytes += allowance
                image = None
                failure_code = None
                fetched_url = candidate.image_url
                # The publication rendition first; the provider's smaller copy
                # only if that one is unreachable, so hotlink protection costs
                # resolution rather than the whole candidate.
                for url in _fetch_order(candidate):
                    try:
                        image = await self.service.image_service.fetch_url(
                            url,
                            provider=candidate.provider,
                            max_bytes=allowance,
                        )
                    except Exception as exc:
                        failure_code = str(getattr(exc, "reason", "fetch_failed"))
                        continue
                    fetched_url = url
                    failure_code = None
                    break
                async with download_lock:
                    reserved_bytes -= allowance
                    if image is not None:
                        byte_size = len(image.content)
                        if byte_size > allowance:
                            image = None
                            failure_code = "size"
                        else:
                            self._downloaded_image_bytes += byte_size
                    outcomes.append((index, key, candidate, image, failure_code, fetched_url))

        await asyncio.gather(
            *(
                fetch_worker()
                for _ in range(min(self.service.max_image_concurrency, len(candidates)))
            )
        )
        self._omitted_image_count += len(candidates) - len(outcomes)
        failures: list[ResearchFailure] = []
        rejected = {key for key, _candidate in candidates}
        for _index, key, candidate, fetched, failure_code, fetched_url in sorted(
            outcomes, key=lambda outcome: outcome[0]
        ):
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
            preview = downscale_for_model(fetched)
            # Held previews, not active ones: during a replacement pass the
            # outgoing image is still held, so this is conservative by up to
            # one window. It fails safe -- the model is never sent more bytes
            # than the budget allows.
            if (
                self._downloaded_image_bytes > self.service.max_download_bytes
                or self._model_image_bytes + len(preview.content) > self.service.max_model_bytes
            ):
                self._omitted_image_count += 1
                continue

            persisted = await self.service.image_service.register(
                conversation_id=UUID(self.scope.conversation_id),
                user_id=UUID(self.scope.user_id),
                upstream_url=fetched_url,
                expected_mime=fetched.media_type,
                provider=candidate.provider,
                cached=fetched,
                expires_at=self.service.now() + self.service.pending_image_ttl,
            )
            # Never derived from len(prepared_images): with eviction that would
            # rebind a retired ID to different bytes.
            candidate_id = f"I{self._next_candidate_index}"
            self._next_candidate_index += 1
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
                source_domain=candidate.source_domain,
            )
            rich_item = ImageRichItem(
                id=f"image:web:{persisted.id}",
                type=RichItemType.image,
                source="image_search",
                title=candidate.title,
                alt_text=str(candidate.description or candidate.title or GENERIC_IMAGE_ALT_TEXT),
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
            self._prepared_by_key[key] = PreparedImage(
                record=record,
                reference_id=persisted.id,
                content=fetched.content,
                rich_item=rich_item,
                preview=preview.content,
                preview_mime=preview.media_type,
                catalog_key=key,
            )
            self._image_digests.add(digest)
            self._model_image_bytes += len(preview.content)
            rejected.discard(key)
        return failures, rejected

    async def _retain_active(self, keys: Sequence[tuple[str, str]]) -> None:
        """Make ``keys`` the active window and release everything else."""

        keep = set(keys)
        evicted = [
            (key, prepared)
            for key, prepared in self._prepared_by_key.items()
            if key not in keep
        ]
        for key, prepared in evicted:
            del self._prepared_by_key[key]
            self._image_digests.discard(prepared.record.digest)
            self._model_image_bytes -= len(prepared.preview)
        # Insertion order is priority order, so the evidence blocks and the
        # bundle both present the best candidates first.
        self.prepared_images = {
            self._prepared_by_key[key].record.candidate_id: self._prepared_by_key[key]
            for key in keys
        }
        if evicted and self.service.image_service is not None:
            await self.service.image_service.release_references(
                [prepared.reference_id for _key, prepared in evicted],
                user_id=UUID(self.scope.user_id),
                conversation_id=UUID(self.scope.conversation_id),
            )


    @property
    def latest_image_query(self) -> str | None:
        """The literal visual subject the most recent image search asked for."""

        return self._latest_image_query

    @property
    def latest_image_objective(self) -> str | None:
        return self._latest_image_objective

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
                            f"Image candidate {candidate_id}; source {record.source_id}; "
                            f"{record.width}x{record.height}; "
                            f"domain {record.source_domain or 'unknown'}. "
                            "Select it only if its visible content supports the answer."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:{prepared.preview_mime};base64,"
                                f"{base64.b64encode(prepared.preview).decode('ascii')}"
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
        operation_source_ids: tuple[str, ...] = (),
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
            operation_source_ids=operation_source_ids,
            failures=failures,
            providers_used=providers,
            omitted_source_count=self._omitted_source_count,
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
