"""Provider-neutral web tools backed by the current turn's research session."""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import AnyHttpUrl, BaseModel, Field

from .tool_context import get_tool_context
from .tool_scope import is_client_only_scope
from .web_query_contract import WebSearchRequest
from .web_research.contracts import ResearchRequest

logger = logging.getLogger(__name__)

WEB_SEARCH_DESCRIPTION = (
    "Search the web for source evidence. State a concrete query and objective. "
    "Use freshness='recent' for current information and freshness='as_of' with "
    "end_date for a historical cutoff; never guess the current year. Results "
    "have stable S# IDs: cite supported claims with [[source:S#]].\n\n"
    "Set visual_intent on the same call whenever the answer would show the subject "
    "rather than only describe it, and provide image_query naming the exact part, screen, "
    "identity art, photo, diagram, map, or chart to inspect. Resolve pronouns "
    "from the conversation and add a version or year when appearance changes. "
    "Validated candidates are shown privately on the next model call; no image "
    "is published unless selected with [[image:I#]].\n\n"
    "Each result lists only the sources that search newly admitted, with "
    "new_source_count and total_source_count. An empty list with a non-zero "
    "total means this search added nothing to evidence you already hold, not "
    "that the web is empty. If failures reports source_quota_exhausted, this "
    "turn's source budget is full: answer from the sources you already have "
    "instead of searching again."
)

WEB_OPEN_DESCRIPTION = (
    "Read specific sources to answer one exact question. Pass S# IDs returned by "
    "web_search, or public page URLs, plus the question to extract. Returns "
    "bounded source-addressed evidence. Open only pages whose search snippets "
    "were insufficient."
)


class ProductWebSearchInput(WebSearchRequest):
    mode: Literal["quick", "agentic"] = Field(
        default="quick",
        description="Requested depth; server policy owns the actual turn mode.",
    )
    visual_intent: Literal["none", "figure", "comparison", "gallery"] = Field(
        default="none",
        description="Visual evidence to gather alongside the sources; 'none' returns no images.",
    )
    image_query: str | None = Field(
        default=None,
        min_length=2,
        max_length=300,
        description="Concrete visual subject when visual_intent is not 'none'.",
    )


class WebOpenInput(BaseModel):
    urls: list[str | AnyHttpUrl] = Field(
        min_length=1,
        max_length=10,
        description="S# source IDs or public pages to read.",
    )
    question: str = Field(
        min_length=3,
        max_length=500,
        description="The exact question to answer from these pages.",
    )


def create_web_search_tool(*, tool_scope: str | None = None) -> StructuredTool:
    async def search(
        query: str,
        objective: str,
        freshness: str = "timeless",
        start_date: Any = None,
        end_date: Any = None,
        locale: str | None = None,
        include_domains: list[str] | None = None,
        max_results: int = 5,
        mode: str = "quick",
        visual_intent: str = "none",
        image_query: str | None = None,
    ) -> str:
        denied = _denied_in_client_only("web_search", tool_scope)
        if denied is not None:
            return denied
        session = get_tool_context().web_research_session
        if session is None:
            return _unavailable("web_search")
        request = ResearchRequest(
            query=query,
            objective=objective,
            mode=session.mode,
            freshness=freshness,
            start_date=start_date,
            end_date=end_date,
            locale=locale,
            include_domains=tuple(include_domains or ()),
            visual_intent=visual_intent,
            image_query=image_query,
        )
        bundle = await session.search(request)
        log_web_tool_call("web_search", outcome=bundle.status)
        return _project(bundle, searches_used=session.budget.search_calls)

    return _internal_tool(
        search,
        name="web_search",
        description=WEB_SEARCH_DESCRIPTION,
        args_schema=ProductWebSearchInput,
        tool_scope=tool_scope,
    )


def create_web_open_tool(*, tool_scope: str | None = None) -> StructuredTool:
    async def open_sources(urls: list[Any], question: str) -> str:
        denied = _denied_in_client_only("web_open", tool_scope)
        if denied is not None:
            return denied
        session = get_tool_context().web_research_session
        if session is None:
            return _unavailable("web_open")
        bundle = await session.open([str(value) for value in urls], question)
        log_web_tool_call("web_open", outcome=bundle.status)
        return _project(bundle, searches_used=session.budget.search_calls)

    return _internal_tool(
        open_sources,
        name="web_open",
        description=WEB_OPEN_DESCRIPTION,
        args_schema=WebOpenInput,
        tool_scope=tool_scope,
    )


#: Snippet ceilings by source status. A triage snippet is bounded hard, but
#: ``web_open`` exists to read pages "whose search snippets were insufficient",
#: so capping a deliberate read at the same bound would gut the deep-read path.
#: Worst case for one operation is an agentic open of 4 pages at 3,000 = 12,000
#: characters, under ``tool_result_offload_threshold_chars`` (16,000), so the
#: deep read stays in the transcript instead of being offloaded to a blob.
_SNIPPET_BOUNDS = {"opened": 3000}
_DEFAULT_SNIPPET_BOUND = 1200


def _project(bundle: Any, *, searches_used: int) -> str:
    """Expose public source evidence; candidate metadata remains session-private."""

    operation_ids = set(bundle.operation_source_ids)
    public_sources = [
        source for source in bundle.sources if source.source_id in operation_ids
    ]
    payload = {
        "status": bundle.status,
        "mode": bundle.mode,
        "operation_index": bundle.operation_index,
        "searches_used": max(0, int(searches_used)),
        "sources": [
            {
                "source_id": source.source_id,
                "title": source.title,
                "url": str(source.url),
                "snippet": (source.snippet or "")[
                    : _SNIPPET_BOUNDS.get(source.status, _DEFAULT_SNIPPET_BOUND)
                ]
                or None,
                "published_at": source.published_at.isoformat() if source.published_at else None,
                "status": source.status,
            }
            for source in public_sources
        ],
        # Without these, {"status":"success","sources":[]} is a riddle -- and
        # that shape is legitimate when a search returns only pages the
        # session already knows.
        "new_source_count": len(public_sources),
        "total_source_count": len(bundle.sources),
        "failures": [
            {
                "operation": failure.operation,
                "provider": failure.provider,
                "code": failure.code,
                "retryable": failure.retryable,
            }
            for failure in bundle.failures
        ],
        "reused": bundle.reused,
        "omitted_source_count": bundle.omitted_source_count,
        "omitted_image_count": bundle.omitted_image_count,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


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


def _denied_in_client_only(operation: str, tool_scope: str | None) -> str | None:
    context = get_tool_context()
    bound_scope = str(getattr(tool_scope, "value", tool_scope) or "default")
    if bound_scope != "client_only" and not is_client_only_scope(
        device_id=context.device_id, tool_scope=context.tool_scope
    ):
        return None
    log_web_tool_call(operation, outcome="permission_denied")
    return json.dumps(
        {
            "status": "error",
            "error_type": "permission_error",
            "retryable": False,
            "message": "Server web access is unavailable in client-only scope.",
        }
    )


def _unavailable(operation: str) -> str:
    log_web_tool_call(operation, outcome="session_unavailable")
    return json.dumps(
        {
            "status": "error",
            "error_type": "session_unavailable",
            "retryable": False,
            "message": "Web research is unavailable for this invocation.",
        }
    )


def log_web_tool_call(operation: str, *, outcome: str, **counts: int) -> None:
    fields = {"operation": operation, "outcome": outcome, **counts}
    logger.info("web_tool_call %s", " ".join(f"{key}={value}" for key, value in fields.items()))


__all__ = [
    "ProductWebSearchInput",
    "WEB_OPEN_DESCRIPTION",
    "WEB_SEARCH_DESCRIPTION",
    "WebOpenInput",
    "create_web_open_tool",
    "create_web_search_tool",
    "log_web_tool_call",
]
