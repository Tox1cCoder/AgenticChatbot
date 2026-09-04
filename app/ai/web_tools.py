"""The web operations ordinary agents are given, in place of raw providers.

Three tools, three decisions:

- ``web_search`` discovers sources. It states an objective and a freshness
  intent; the server anchors the dates and never asks for raw page bodies.
- ``web_open`` extracts an answer to one stated question from a few chosen
  pages. It cannot be called without that question.
- ``image_search`` finds a picture for one visual subject, independently of any
  text search, so a figure never waits on a web round trip it does not need.

The combined tool these replace did all three at once: every research call
spent an image request, every search dragged whole pages into context, and the
model's only recency lever was a year it had typed from memory.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo

from langchain_core.tools import StructuredTool
from pydantic import AnyHttpUrl, BaseModel, Field, ValidationError

from ..core.config import settings
from .focused_tool_result import select_focused_excerpts
from .image_discovery_flow import discover_images, record_discovery_outcome
from .research_budget import get_research_budget
from .selected_image_sink import offer_selected_images
from .tool_context import get_tool_context
from .tool_result_rendering import provider_result_text
from .tool_scope import is_client_only_scope
from .web_query_contract import (
    WebQueryError,
    WebSearchRequest,
    normalize_web_search,
    tavily_search_args,
)

logger = logging.getLogger(__name__)

#: Longest snippet kept per search result before the whole payload is bounded.
_SNIPPET_MAX_CHARS = 1_200
#: Failure records reported by one web_open call.
_MAX_FAILURE_RECORDS = 10
#: Longest provider failure string echoed back, per URL.
_FAILURE_REASON_MAX_CHARS = 200
#: Floor for the excerpt budget once the web_open envelope is subtracted.
_MIN_EXCERPT_BUDGET = 1_000

WEB_SEARCH_DESCRIPTION = (
    "Search the web for sources. State the query as you would type it into a "
    "search engine and state the objective — the concrete fact the search has "
    "to produce.\n\n"
    "Set freshness='recent' for current, latest, or ongoing subjects: the "
    "server rewrites stale years in your query to today's date, so never guess "
    "the current year yourself. Set freshness='as_of' with an explicit "
    "end_date for a historical cutoff, which preserves every year as written. "
    "Leave freshness='timeless' for facts that do not move.\n\n"
    "Returns ranked titles, URLs, publication dates, and snippets. It does not "
    "fetch page bodies: read a snippet first, and call web_open only for the "
    "few URLs whose snippets cannot answer the question. Do not repeat an "
    "unchanged query — the same query returns the same sources."
)

WEB_OPEN_DESCRIPTION = (
    "Read specific pages to answer one exact question. Pass the URLs you chose "
    "from search results, or a URL the user gave you, together with the "
    "question to extract — a question is required, and it is what ranks the "
    "returned passages.\n\n"
    "Returns bounded, source-addressed excerpts rather than whole pages, plus "
    "a record for each URL that could not be read. Open only the pages whose "
    "search snippets were insufficient."
)

IMAGE_SEARCH_DESCRIPTION = (
    "Find a picture of one visual subject. This is the only way an image "
    "reaches your answer, so reach for it whenever the reader would benefit "
    "from seeing the thing — not only when you need sources.\n\n"
    "The query reaches the image provider exactly as written, so write it as "
    "the search you would type yourself:\n"
    "- Name the part the reader has to look at, not the product that contains "
    "it. A question about warning lights wants the instrument cluster, not the "
    "vehicle; a question about a setting wants that settings screen, not the "
    "console. Naming the product returns its most photographed view, which is "
    "rarely the thing being explained.\n"
    "- Resolve what the user referred to. 'my scooter', 'this game', 'it' "
    "cannot be searched — recover the named thing from the conversation first.\n"
    "- Write it in the language the subject is documented in, which is not "
    "necessarily the language of the answer. A product, place or interface "
    "known mainly in one country is photographed and captioned there; a global "
    "subject is best served in English however the user writes.\n"
    "- Name the form the question calls for — asked what a thing is, its "
    "identity image (logo, key art, cover, poster); asked how it works or "
    "looks in use, a photo, screenshot, diagram, map or chart.\n"
    "- Add the year or version when what matters is how the subject looks now: "
    "the image provider has no recency filter, so the query text is the only "
    "way to ask for a current picture. Set time_range as well when the user "
    "asked for something current; a picture crawled before that window is "
    "dropped.\n"
    "- No question words, and no disambiguator the search itself does not "
    "need.\n\n"
    "One call, one subject, one figure. Call again with a different query for "
    "each further subject the answer needs, and let the answer decide how many "
    "that is — a subject introduced from scratch often wants its identity art "
    "and a shot of it in use, while a how-to usually wants the one screen "
    "being described. Repeating a subject returns nothing new.\n\n"
    "Set intent='gallery' only when the user asks to see or compare several "
    "instances — a roster, a set of logos, colour options. Otherwise leave it "
    "unset: the default places up to two images beside the prose they support. "
    "Never state how many images you want; the layout decides.\n\n"
    "Selected images appear in your available rich items. Not every call "
    "produces one, and a complete answer never depends on an image. A gallery "
    "arrives as ONE grid item with one marker."
)


class WebProviderError(RuntimeError):
    """A structured provider failure surfaced through a product tool."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        self.retryable = bool(retryable)
        super().__init__(str(message or "Web provider failed")[:300])


class WebOpenInput(BaseModel):
    urls: list[AnyHttpUrl] = Field(
        min_length=1,
        max_length=10,
        description="Pages to read. Clamped to the configured maximum per call.",
    )
    question: str = Field(
        min_length=3,
        max_length=500,
        description=(
            "The exact question to answer from these pages. Required: it is sent "
            "to the extractor and it ranks the passages that come back."
        ),
    )


class ImageSearchInput(BaseModel):
    query: str = Field(
        min_length=2,
        max_length=300,
        description="The concrete visual subject, written as a search.",
    )
    intent: Literal["figure", "gallery"] | None = Field(
        default=None,
        description=(
            "Layout: 'figure' (default) for one image beside the prose, "
            "'gallery' for a grid when the user asks to compare several "
            "instances. Never state a count."
        ),
    )
    max_images: int | None = Field(
        default=None,
        ge=1,
        le=8,
        description="Upper bound on a gallery grid. Ignored in figure mode.",
    )
    time_range: Literal["day", "week", "month", "year"] | None = Field(
        default=None,
        description=(
            "Recency window the picture must satisfy. Set it only when the user "
            "asked for something current."
        ),
    )


def create_web_search_tool(
    *,
    tavily_tool: Any | None = None,
    extract_tool: Any | None = None,
    tool_scope: str | None = None,
    clock: Callable[[], datetime] | None = None,
) -> StructuredTool:
    """Build ``web_search``. Providers and the clock are injected in tests.

    ``extract_tool`` is accepted and never called. Auto-extracting every hit is
    what made one question cost four page fetches; the parameter exists so the
    test asserting it stays uncalled has something to assert against.
    """

    async def _search(
        query: str,
        objective: str,
        freshness: str = "timeless",
        start_date: Any = None,
        end_date: Any = None,
        locale: str | None = None,
        include_domains: list[str] | None = None,
        max_results: int = 5,
    ) -> str:
        denied = _denied_in_client_only(tool_scope)
        if denied is not None:
            return denied
        try:
            normalized = normalize_web_search(
                WebSearchRequest(
                    query=query,
                    objective=objective,
                    freshness=freshness,
                    start_date=start_date,
                    end_date=end_date,
                    locale=locale,
                    include_domains=include_domains or [],
                    max_results=max_results,
                ),
                now=(clock or _default_clock)(),
                configured_max_results=int(settings.web_search_max_results),
            )
        except (WebQueryError, ValidationError) as exc:
            return _error_payload(
                str(exc),
                retryable=False,
                error_type="invalid_request",
                hint="Fix the search intent and call again; the provider was not contacted.",
            )

        args = tavily_search_args(normalized)
        scope = _search_scope(args)
        budget = get_research_budget(get_tool_context().conversation_id)
        reused = (
            budget.find_reuse(normalized.query, scope=scope)
            if settings.research_budget_enabled
            else None
        )
        if reused is not None:
            return _project_search(normalized, reused, reused=True, budget=budget)
        if settings.research_budget_enabled and not budget.reserve_search(
            normalized.query, scope=scope
        ):
            return _budget_spent_payload(budget)

        try:
            raw = await _call_provider(tavily_tool, "tavily", "tavily_search", args)
        except WebProviderError as exc:
            logger.warning("Web search provider failed: %s", exc)
            return _error_payload(str(exc), retryable=exc.retryable)
        except Exception as exc:
            logger.warning("Web search failed: %s", exc)
            return _error_payload(str(exc))
        budget.record_search(normalized.query, raw, scope=scope)
        return _project_search(normalized, raw, reused=False, budget=budget)

    return _internal_tool(
        _search,
        name="web_search",
        description=WEB_SEARCH_DESCRIPTION,
        args_schema=WebSearchRequest,
        tool_scope=tool_scope,
    )


def create_web_open_tool(
    *,
    extract_tool: Any | None = None,
    tool_scope: str | None = None,
) -> StructuredTool:
    """Build ``web_open``. The extractor is injected in tests."""

    async def _open(urls: list[Any], question: str) -> str:
        denied = _denied_in_client_only(tool_scope)
        if denied is not None:
            return denied
        focus = str(question or "").strip()
        if len(focus) < 3:
            return _error_payload(
                "web_open requires the exact question to extract.",
                retryable=False,
                error_type="invalid_request",
                hint="State the fact you need from these pages, then call again.",
            )
        requested = [str(url).strip() for url in urls if str(url or "").strip()]
        requested = requested[: max(1, int(settings.web_open_max_urls))]
        if not requested:
            return _error_payload(
                "web_open requires at least one URL.",
                retryable=False,
                error_type="invalid_request",
                hint="Pass URLs discovered by web_search or supplied by the user.",
            )

        args = {
            "urls": requested,
            "query": focus,
            "chunks_per_source": 3,
            "include_images": False,
        }
        try:
            raw = await _call_provider(extract_tool, "tavily", "tavily_extract", args)
        except WebProviderError as exc:
            logger.warning("Web open provider failed: %s", exc)
            return _error_payload(str(exc), retryable=exc.retryable)
        except Exception as exc:
            logger.warning("Web open failed: %s", exc)
            return _error_payload(str(exc))
        return _project_extract(raw, question=focus, requested=requested)

    return _internal_tool(
        _open,
        name="web_open",
        description=WEB_OPEN_DESCRIPTION,
        args_schema=WebOpenInput,
        tool_scope=tool_scope,
    )


def create_image_search_tool(
    *,
    brave_tool: Any | None = None,
    tavily_tool: Any | None = None,
    tool_scope: str | None = None,
) -> StructuredTool:
    """Build ``image_search``. It never touches the text provider.

    ``tavily_tool`` is accepted and ignored so a caller wiring all three tools
    from one place cannot accidentally couple a picture to a web search; the
    test that asserts it stays uncalled is the point.
    """

    async def _image_search(
        query: str,
        intent: str | None = None,
        max_images: int | None = None,
        time_range: str | None = None,
    ) -> str:
        denied = _denied_in_client_only(tool_scope)
        if denied is not None:
            return denied
        subject = str(query or "").strip()
        context = get_tool_context()
        if not _image_path_open(context.rich_response_capable):
            record_discovery_outcome("skipped")
            return _image_payload(subject, 0, "Image discovery is unavailable for this request.")

        budget = get_research_budget(context.conversation_id)
        if not budget.reserve_image_search(subject):
            record_discovery_outcome("skipped")
            return _image_payload(
                subject,
                0,
                "This subject was already searched in this turn; its picture, if one "
                "was found, is already in your available rich items.",
            )

        selected = await _discover(
            brave_tool=brave_tool,
            image_query=subject,
            intent=intent,
            max_images=max_images,
            time_range=time_range,
        )
        budget.record_image_search(selected)
        if selected:
            offer_selected_images(selected)
        return _image_payload(
            subject,
            len(selected),
            "The selected image is in your available rich items; place it with its marker."
            if selected
            else "No suitable image was found for this subject. Answer without one.",
        )

    return _internal_tool(
        _image_search,
        name="image_search",
        description=IMAGE_SEARCH_DESCRIPTION,
        args_schema=ImageSearchInput,
        tool_scope=tool_scope,
    )


def _internal_tool(
    coroutine: Any,
    *,
    name: str,
    description: str,
    args_schema: type[BaseModel],
    tool_scope: str | None,
) -> StructuredTool:
    return StructuredTool.from_function(
        coroutine=coroutine,
        name=name,
        description=description,
        args_schema=args_schema,
        metadata={
            "tool_origin": "internal",
            "qualified_tool_id": f"internal::{name}",
            "tool_scope": str(tool_scope or "default"),
        },
    )


def _default_clock() -> datetime:
    """The server's own wall clock, in the configured display timezone.

    Authoritative on purpose: the model never has to call a time tool before
    searching, and a prompt that tells it to would be a second, weaker source
    of truth for the same fact.
    """

    name = str(getattr(settings, "runtime_time_context_timezone", "UTC") or "UTC")
    try:
        return datetime.now(ZoneInfo(name))
    except Exception:
        from datetime import timezone

        return datetime.now(timezone.utc)


def _image_path_open(rich_response_capable: bool) -> bool:
    if not settings.remote_image_enrichment_enabled:
        return False
    if not settings.inline_rich_response_enabled:
        return False
    # A selected candidate reaches the answer only through the rich-item
    # inventory, which the graph withholds from a request that never advertised
    # the capability. Discovering one anyway spends a provider call on output
    # that is discarded.
    return bool(rich_response_capable)


async def _discover(
    *,
    brave_tool: Any | None,
    image_query: str,
    intent: str | None,
    max_images: int | None,
    time_range: str | None,
) -> list[dict[str, Any]]:
    if brave_tool is None:
        brave_tool = await _resolve_tool("brave_image_search", "brave_image_search")
    return await discover_images(
        brave_tool=brave_tool,
        image_query=image_query,
        image_intent=intent,
        time_range=time_range,
        max_images=max_images,
    )


async def _call_provider(
    injected: Any | None,
    server_name: str,
    tool_name: str,
    args: dict[str, Any],
) -> str:
    tool = injected or await _resolve_tool(server_name, tool_name)
    if tool is None:
        raise WebProviderError(f"{tool_name} is unavailable", retryable=False)
    return _raise_for_provider_error(
        provider_result_text(await tool.ainvoke(args), tool_name=tool_name)
    )


async def _resolve_tool(server_name: str, tool_name: str) -> Any | None:
    """Find one MCP tool, or None. Absence is reported by whoever needed it."""

    context = get_tool_context()
    if is_client_only_scope(device_id=context.device_id, tool_scope=context.tool_scope):
        return None
    try:
        from .mcp_registry import get_global_mcp_manager

        manager = await get_global_mcp_manager()
        for tool in await manager.get_server_tools(server_name):
            if getattr(tool, "name", None) == tool_name:
                return tool
    except Exception as exc:
        logger.debug("MCP tool %s unavailable: %s", tool_name, type(exc).__name__)
    return None


def _raise_for_provider_error(result_text: str) -> str:
    try:
        payload = json.loads(result_text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return result_text
    if isinstance(payload, dict) and payload.get("error"):
        raise WebProviderError(str(payload["error"]), retryable=bool(payload.get("retryable")))
    return result_text


def _search_scope(args: dict[str, Any]) -> tuple[Any, ...]:
    """The controls that make two identical query strings different searches."""

    return tuple(
        str(args.get(key)) if args.get(key) is not None else None
        for key in ("topic", "start_date", "end_date", "max_results", "include_domains")
    )


def _project_search(
    normalized: Any,
    raw: str,
    *,
    reused: bool,
    budget: Any,
) -> str:
    """Reduce the provider payload to what a model can act on, and bound it."""

    try:
        payload = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        payload = {}
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in payload.get("results") or []:
        if not isinstance(entry, dict):
            continue
        url = str(entry.get("url") or "").strip()
        key = _canonical_url(url)
        if not key or key in seen:
            continue
        seen.add(key)
        results.append(
            {
                "title": str(entry.get("title") or "")[:300],
                "url": url,
                "published_date": entry.get("published_date"),
                "snippet": str(entry.get("content") or "")[:_SNIPPET_MAX_CHARS],
                "score": entry.get("score"),
            }
        )

    envelope: dict[str, Any] = {
        "query": normalized.query,
        "objective": normalized.objective,
        "freshness": normalized.freshness,
        "start_date": normalized.start_date.isoformat() if normalized.start_date else None,
        "end_date": normalized.end_date.isoformat() if normalized.end_date else None,
        "results": results,
        "total_results": len(results),
        "omitted_results": 0,
        "reused": reused,
        "searches_used": budget.search_calls,
    }
    return _fit(envelope, int(settings.web_search_result_max_chars))


def _fit(envelope: dict[str, Any], budget_chars: int) -> str:
    """Drop trailing results until the serialized payload fits the budget.

    Results are dropped whole rather than shortened: half a snippet with its
    URL still attached reads as a complete source and is the easiest way for a
    model to cite something the page never said.
    """

    total = len(envelope["results"])
    while True:
        envelope["total_results"] = len(envelope["results"])
        envelope["omitted_results"] = total - len(envelope["results"])
        serialized = json.dumps(envelope, ensure_ascii=False)
        if len(serialized) <= budget_chars or not envelope["results"]:
            return serialized
        envelope["results"].pop()


def _project_extract(raw: str, *, question: str, requested: list[str]) -> str:
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        payload = {}
    results = payload.get("results") if isinstance(payload, dict) else None
    failures = payload.get("failed_results") if isinstance(payload, dict) else None

    envelope: dict[str, Any] = {
        "question": question,
        "requested_urls": requested,
        "failed": _failure_records(failures),
    }
    budget = max(1, int(settings.web_open_max_chars))
    # The excerpt budget is what remains after the envelope, so the returned
    # string honors the configured cap rather than the cap plus overhead.
    overhead = len(json.dumps(envelope, ensure_ascii=False))
    focused = select_focused_excerpts(
        json.dumps(results if isinstance(results, list) else [], ensure_ascii=False),
        objective=question,
        max_excerpts=max(1, int(settings.web_open_max_excerpts)),
        max_chars=max(_MIN_EXCERPT_BUDGET, budget - overhead),
        # The extractor was handed this same question and returned the chunks it
        # judged relevant. Exact term matching can still score every one of them
        # at zero — "which release date is stated" shares no token with
        # "released on 14 March" — and discarding provider-ranked evidence over
        # a word ending would lose the answer the fetch already paid for.
        fallback_to_leading=True,
    )
    envelope.update(
        {
            "excerpts": [item.model_dump(exclude_none=True) for item in focused.excerpts],
            "total_candidates": focused.total_candidates,
            "omitted_candidates": focused.omitted_candidates,
            "truncated": focused.truncated,
            "note": focused.note,
        }
    )
    return json.dumps(envelope, ensure_ascii=False)


def _failure_records(failures: Any) -> list[dict[str, str]]:
    """Per-URL failures only. Provider diagnostics are not model context."""

    records: list[dict[str, str]] = []
    for entry in failures or []:
        if not isinstance(entry, dict):
            continue
        url = str(entry.get("url") or "").strip()
        if not url:
            continue
        reason = str(entry.get("error") or entry.get("message") or "extraction failed")
        records.append({"url": url, "error": reason[:_FAILURE_REASON_MAX_CHARS]})
        if len(records) >= _MAX_FAILURE_RECORDS:
            break
    return records


def _canonical_url(url: str) -> str | None:
    try:
        parsed = urlsplit(str(url or "").strip())
        host = str(parsed.hostname or "").lower()
    except (UnicodeError, ValueError):
        return None
    if not host:
        return None
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((parsed.scheme.lower(), host, path, "", ""))


def _image_payload(query: str, selected: int, note: str) -> str:
    return json.dumps({"query": query, "selected": selected, "note": note}, ensure_ascii=False)


def _budget_spent_payload(budget: Any) -> str:
    return json.dumps(
        {
            "status": "error",
            "error_type": "budget_exhausted",
            "retryable": False,
            "searches_used": budget.search_calls,
            "hint": (
                "The per-turn search budget is spent. Answer from the sources already "
                "gathered in this turn, or open one of them with web_open; another "
                "search would return the same results."
            ),
        },
        ensure_ascii=False,
    )


def _denied_in_client_only(tool_scope: str | None) -> str | None:
    context = get_tool_context()
    bound_scope = str(getattr(tool_scope, "value", tool_scope) or "default")
    if bound_scope == "client_only" or is_client_only_scope(
        device_id=context.device_id,
        tool_scope=context.tool_scope,
    ):
        return _error_payload(
            "Server web access is unavailable in client-only tool scope.",
            retryable=False,
            error_type="permission_error",
            hint="Use a suitable tool from the active client device instead.",
        )
    return None


def _error_payload(
    message: str,
    *,
    retryable: bool = True,
    error_type: str = "provider_error",
    hint: str = "The web is temporarily unavailable. Say so rather than guessing.",
) -> str:
    return json.dumps(
        {
            "status": "error",
            "error_type": error_type,
            "retryable": retryable,
            "hint": hint,
            "message": message[:500],
        },
        ensure_ascii=False,
    )


__all__ = [
    "IMAGE_SEARCH_DESCRIPTION",
    "WEB_OPEN_DESCRIPTION",
    "WEB_SEARCH_DESCRIPTION",
    "ImageSearchInput",
    "WebOpenInput",
    "WebProviderError",
    "create_image_search_tool",
    "create_web_open_tool",
    "create_web_search_tool",
]
