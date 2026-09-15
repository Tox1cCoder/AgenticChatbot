"""Thin provider adapters for canonical web research."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol

from app.ai.tool_result_rendering import provider_result_text
from app.ai.web_query_contract import NormalizedWebSearch, tavily_search_args

from .contracts import ProviderImageCandidate, ProviderSource, ResearchRequest


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


class BraveImageSearchProvider:
    name = "brave"

    def __init__(self, tool: Any, *, health_key: str = "brave:default") -> None:
        self.tool = tool
        self.health_key = health_key

    async def search(self, request: ResearchRequest) -> tuple[ProviderImageCandidate, ...]:
        query = str(request.image_query or request.query).strip()
        payload = await _payload(
            self.tool,
            {"query": query, "count": 10, "safesearch": "strict"},
            provider=self.name,
        )
        records: list[ProviderImageCandidate] = []
        for fallback_rank, raw in enumerate(payload.get("images") or (), start=1):
            if not isinstance(raw, dict):
                continue
            image_url = raw.get("thumbnail_url") or raw.get("url") or raw.get("image_url")
            source_url = raw.get("source_url") or raw.get("page_url")
            if not image_url or not source_url:
                continue
            records.append(
                ProviderImageCandidate(
                    provider=self.name,
                    image_url=str(image_url),
                    source_url=str(source_url),
                    title=str(raw["title"]) if raw.get("title") else None,
                    description=(
                        str(raw.get("description") or raw.get("alt_text") or "")[:1000] or None
                    ),
                    width=raw.get("width"),
                    height=raw.get("height"),
                    rank=max(1, int(raw.get("result_rank") or fallback_rank)),
                    published_at=_datetime(raw.get("published_at")),
                )
            )
        return tuple(records)


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
