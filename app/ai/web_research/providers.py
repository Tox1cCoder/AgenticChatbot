"""Thin provider adapters for canonical web research."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol
from urllib.parse import urlsplit

from app.ai.tool_result_rendering import provider_result_text
from app.ai.web_query_contract import (
    NormalizedWebSearch,
    bare_host,
    host_matches,
    tavily_search_args,
)

from .contracts import ProviderImageCandidate, ProviderSource, ResearchRequest
from .source_registry import canonicalize_public_url

logger = logging.getLogger(__name__)


class ProviderFailure(RuntimeError):
    def __init__(self, code: str, *, provider: str, retryable: bool) -> None:
        self.code = code
        self.provider = provider
        self.retryable = retryable
        super().__init__(code)


class TextSearchProvider(Protocol):
    name: str
    health_key: str

    async def search(
        self, request: NormalizedWebSearch, *, query_index: int
    ) -> tuple[ProviderSource, ...]: ...


class ImageSearchProvider(Protocol):
    name: str
    health_key: str

    async def search(self, request: ResearchRequest) -> tuple[ProviderImageCandidate, ...]: ...


class PageOpenProvider(Protocol):
    name: str
    health_key: str

    async def open(
        self, urls: Sequence[str], question: str, *, query_index: int
    ) -> tuple[ProviderSource, ...]: ...


class ProviderResolver:
    """Configured provider order, with no runtime service lookup."""

    def __init__(
        self,
        *,
        text: Sequence[TextSearchProvider] = (),
        images: Sequence[ImageSearchProvider] = (),
        openers: Sequence[PageOpenProvider] = (),
    ) -> None:
        self.text = tuple(text)
        self.images = tuple(images)
        self.openers = tuple(openers)


def _datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


async def _payload(tool: Any, args: dict[str, Any], *, provider: str) -> dict[str, Any]:
    try:
        result = await tool.ainvoke(args)
    except Exception as exc:
        raise ProviderFailure("transport_error", provider=provider, retryable=True) from exc
    text = provider_result_text(result, tool_name=provider)
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ProviderFailure("invalid_response", provider=provider, retryable=False) from exc
    if not isinstance(payload, dict):
        raise ProviderFailure("invalid_response", provider=provider, retryable=False)
    error = payload.get("error")
    if error:
        code = str(payload.get("error_type") or "provider_error").strip().lower()
        retryable = code in {"rate_limited", "timeout", "transport_error", "server_error"}
        raise ProviderFailure(code[:64], provider=provider, retryable=retryable)
    return payload


class TavilyTextSearchProvider:
    name = "tavily"

    def __init__(self, tool: Any, *, health_key: str = "tavily:default") -> None:
        self.tool = tool
        self.health_key = health_key

    async def search(
        self, request: NormalizedWebSearch, *, query_index: int
    ) -> tuple[ProviderSource, ...]:
        payload = await _payload(self.tool, tavily_search_args(request), provider=self.name)
        records: list[ProviderSource] = []
        for rank, raw in enumerate(payload.get("results") or (), start=1):
            if not isinstance(raw, dict) or not raw.get("url"):
                continue
            records.append(
                ProviderSource(
                    provider=self.name,
                    url=str(raw["url"]),
                    title=str(raw["title"]) if raw.get("title") else None,
                    snippet=str(raw.get("content") or raw.get("snippet") or "")[:4000] or None,
                    rank=rank,
                    query_index=query_index,
                    published_at=_datetime(raw.get("published_date") or raw.get("published_at")),
                )
            )
        return tuple(records)


def _brave_renditions(raw: dict[str, Any]) -> tuple[str | None, str | None]:
    """Pick what to publish and what to fall back to, from one Brave result.

    Brave returns the page's own image (``original_image_url``) alongside its
    proxied thumbnail, which is capped near 500px on the long edge. The original
    is usually far larger and is the one worth publishing -- but not always: a
    Wikipedia page hands Brave a 250px rendition, and preferring it blindly
    would publish a smaller image than the proxy already offers. Compare the
    declared dimensions and take the larger, treating unknown dimensions as
    "no reason to switch".
    """

    thumbnail = raw.get("thumbnail_url") or None
    original = raw.get("original_image_url") or None
    if not original:
        return (thumbnail or raw.get("url") or raw.get("image_url"), None)
    if not thumbnail:
        return (original, None)

    original_edge = _longest_edge(raw.get("width"), raw.get("height"))
    thumbnail_edge = _longest_edge(raw.get("thumbnail_width"), raw.get("thumbnail_height"))
    if original_edge is None or (thumbnail_edge is not None and original_edge <= thumbnail_edge):
        return (thumbnail, original)
    return (original, thumbnail)


def _longest_edge(width: Any, height: Any) -> int | None:
    values = [value for value in (width, height) if isinstance(value, int) and value > 0]
    return max(values) if values else None


#: Results asked of Brave per call. Also the threshold below which a
#: domain-restricted search earns its one supplemental probe.
_BRAVE_IMAGE_COUNT = 10


def _brave_locale(locale: str | None) -> dict[str, str]:
    """Split a BCP-47 hint into Brave's own two arguments.

    Anything that is not a two-letter subtag is dropped rather than guessed:
    forwarding a bad ``country`` narrows the search to nothing.
    """

    parts = str(locale or "").replace("_", "-").split("-")
    args: dict[str, str] = {}
    if parts and len(parts[0]) == 2:
        args["search_lang"] = parts[0].lower()
    if len(parts) > 1 and len(parts[1]) == 2:
        args["country"] = parts[1].upper()
    return args


class BraveImageSearchProvider:
    name = "brave"

    def __init__(self, tool: Any, *, health_key: str = "brave:default") -> None:
        self.tool = tool
        self.health_key = health_key

    async def search(self, request: ResearchRequest) -> tuple[ProviderImageCandidate, ...]:
        query = str(request.image_query or request.query).strip()
        args: dict[str, Any] = {
            "query": query,
            "count": _BRAVE_IMAGE_COUNT,
            "safesearch": "strict",
        }
        args.update(_brave_locale(request.locale))
        payload = await _payload(self.tool, args, provider=self.name)

        # ``include_domains`` carries whatever the model typed -- often a full
        # URL. Normalize once here; every later comparison and the probe query
        # use the bare host, never the raw entry.
        allowed = tuple(
            dict.fromkeys(host for host in map(bare_host, request.include_domains) if host)
        )
        records = self._candidates(payload, allowed=allowed)

        if allowed and len(records) < _BRAVE_IMAGE_COUNT:
            probe = {**args, "query": f"{query} site:{allowed[0]}"}
            try:
                extra = await _payload(self.tool, probe, provider=self.name)
            except ProviderFailure as exc:
                # The broad call already produced usable allowed-domain results;
                # losing the supplement must not lose them too.
                logger.warning(
                    "brave image domain probe failed provider=%s domain=%s code=%s",
                    self.name,
                    allowed[0],
                    exc.code,
                )
            else:
                records.extend(self._candidates(extra, allowed=allowed))

        return _deduplicate_candidates(records)

    def _candidates(
        self, payload: dict[str, Any], *, allowed: tuple[str, ...]
    ) -> list[ProviderImageCandidate]:
        records: list[ProviderImageCandidate] = []
        for fallback_rank, raw in enumerate(payload.get("images") or (), start=1):
            if not isinstance(raw, dict):
                continue
            image_url, preview_url = _brave_renditions(raw)
            source_url = raw.get("source_url") or raw.get("page_url")
            if not image_url or not source_url:
                continue
            host = str(
                raw.get("source_domain") or urlsplit(str(source_url)).hostname or ""
            )
            if allowed and not any(host_matches(host, entry) for entry in allowed):
                # include_domains is a restriction, not a preference: a
                # disallowed result is dropped, never returned as a fallback.
                continue
            records.append(
                ProviderImageCandidate(
                    provider=self.name,
                    image_url=str(image_url),
                    preview_url=str(preview_url) if preview_url else None,
                    source_url=str(source_url),
                    title=str(raw["title"]) if raw.get("title") else None,
                    description=(
                        str(raw.get("description") or raw.get("alt_text") or "")[:1000] or None
                    ),
                    width=raw.get("width"),
                    height=raw.get("height"),
                    rank=max(1, int(raw.get("result_rank") or fallback_rank)),
                    published_at=_datetime(raw.get("published_at")),
                    confidence=str(raw["confidence"]).lower() if raw.get("confidence") else None,
                    source_domain=host or None,
                )
            )
        return records


def _deduplicate_candidates(
    records: Sequence[ProviderImageCandidate],
) -> tuple[ProviderImageCandidate, ...]:
    """Collapse the broad call and the probe on their overlap, first wins."""

    seen: set[tuple[str, str]] = set()
    unique: list[ProviderImageCandidate] = []
    for record in records:
        key = (
            canonicalize_public_url(record.source_url) or record.source_url,
            record.image_url,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return tuple(unique)


class TavilyPageOpenProvider:
    name = "tavily"

    def __init__(self, tool: Any, *, health_key: str = "tavily:default") -> None:
        self.tool = tool
        self.health_key = health_key

    async def open(
        self, urls: Sequence[str], question: str, *, query_index: int
    ) -> tuple[ProviderSource, ...]:
        payload = await _payload(
            self.tool,
            {
                "urls": list(urls),
                "query": question,
                "chunks_per_source": 3,
                "include_images": False,
            },
            provider=self.name,
        )
        records: list[ProviderSource] = []
        for rank, raw in enumerate(payload.get("results") or (), start=1):
            if not isinstance(raw, dict) or not raw.get("url"):
                continue
            records.append(
                ProviderSource(
                    provider=self.name,
                    url=str(raw["url"]),
                    title=str(raw["title"]) if raw.get("title") else None,
                    snippet=str(raw.get("raw_content") or raw.get("content") or "")[:4000]
                    or None,
                    rank=rank,
                    query_index=query_index,
                )
            )
        return tuple(records)


__all__ = [
    "BraveImageSearchProvider",
    "ImageSearchProvider",
    "PageOpenProvider",
    "ProviderFailure",
    "ProviderResolver",
    "TavilyTextSearchProvider",
    "TavilyPageOpenProvider",
    "TextSearchProvider",
]
