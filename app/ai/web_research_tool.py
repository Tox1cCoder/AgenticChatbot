"""The single model-facing research operation.

The model asks one question and optionally names a visual subject. The server
decides everything else: whether to hit the network at all, whether to look for
an image, and whether any image it found may be shown. Prompted coordination of
two providers proved unreliable — a trace shows three sequential text searches
and no image search at all — so the sequencing lives here instead.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..core.config import settings
from .research_budget import get_research_budget
from .tool_context import get_tool_context
from .tool_result_rendering import provider_result_text
from .verified_image_sink import offer_verified_images

logger = logging.getLogger(__name__)

_DESCRIPTION = (
    "Research the web. Returns a synthesized answer plus ranked sources with URLs.\n\n"
    "This tool automatically considers a verified image. Set image_query only to "
    "make the visual subject more precise than the factual query: one concrete "
    "subject, no question words, plus a disambiguator or a form word (photo, "
    "diagram, map, chart) when it matters.\n\n"
    "Set skip_images=true only when a visual cannot support the answer.\n\n"
    "Set image_intent='gallery' only when the user asks to see or compare several "
    "instances — a roster, a set of logos, colour options. Otherwise leave it "
    "unset: the default places up to two images beside the prose they support. "
    "Never state how many images you want; the layout decides.\n\n"
    "Approved images appear in your available rich items. Not every call produces "
    "one, and a complete answer never depends on an image. A gallery arrives as "
    "ONE grid item with one marker."
)


class WebResearchInput(BaseModel):
    query: str = Field(description="The factual research query.")
    image_query: str | None = Field(
        default=None,
        description=(
            "Short concrete visual subject. Set it whenever seeing the thing would "
            "help the reader; omit for abstract topics."
        ),
    )
    image_intent: Literal["figure", "gallery"] | None = Field(
        default=None,
        description=(
            "Layout: 'figure' (default) for up to two images beside the prose, "
            "'gallery' for a grid when the user asks to see several instances or "
            "to compare things. Never state a count."
        ),
    )
    skip_images: bool = Field(
        default=False,
        description="Set true only when an image cannot help the answer.",
    )
    max_results: int | None = Field(default=None, description="Optional result count.")
    search_depth: str | None = Field(default=None, description="Optional Tavily depth.")


def create_web_research_tool(
    *,
    tavily_tool: Any | None = None,
    brave_tool: Any | None = None,
    web_image_service: Any | None = None,
    verifier_model: Any | None = None,
    recorder: Any | None = None,
) -> StructuredTool:
    """Build the ``web_research`` tool. Dependencies are injected in tests."""

    async def _research(
        query: str,
        image_query: str | None = None,
        image_intent: str | None = None,
        max_results: int | None = None,
        search_depth: str | None = None,
        skip_images: bool = False,
    ) -> str:
        context = get_tool_context()
        budget = get_research_budget(context.conversation_id)
        visual_query = str(image_query or query).strip()
        wants_image = bool(visual_query) and not skip_images

        reused = budget.find_reuse(query) if settings.research_budget_enabled else None
        search_task: asyncio.Task[str] | None = None
        if reused is None:
            # reserve_search claims the slot in one step; a bare check here would
            # race a concurrent web_research call across the await below.
            if settings.research_budget_enabled and not budget.reserve_search(query):
                return _budget_reused_payload(budget)
            search_task = asyncio.create_task(
                _run_search(tavily_tool, query, max_results, search_depth)
            )

        image_task: asyncio.Task[list[dict[str, Any]]] | None = None
        if wants_image and _image_path_open(budget, context.rich_response_capable):
            image_task = asyncio.create_task(
                _discover_and_verify(
                    brave_tool=brave_tool,
                    web_image_service=web_image_service,
                    verifier_model=verifier_model,
                    user_request=query,
                    image_query=visual_query,
                    factual_query=query,
                    image_intent=image_intent,
                    recorder=recorder,
                )
            )

        if search_task is not None:
            try:
                search_text = await search_task
            except Exception as exc:
                if image_task is not None:
                    image_task.cancel()
                logger.warning("Research search failed: %s", exc)
                return _error_payload(str(exc))
            budget.record_search(query, search_text)
            search_reused = False
        else:
            search_text = reused or ""
            search_reused = True

        approved = await _collect_images(image_task, budget, wants_image)
        if approved:
            offer_verified_images(approved)
        return _with_research_meta(search_text, reused=search_reused, budget=budget)

    return StructuredTool.from_function(
        coroutine=_research,
        name="web_research",
        description=_DESCRIPTION,
        args_schema=WebResearchInput,
        metadata={
            "tool_origin": "internal",
            "qualified_tool_id": "internal::web_research",
        },
    )


def _image_path_open(budget: Any, rich_response_capable: bool) -> bool:
    if not settings.vision_image_verification_enabled:
        return False
    if not settings.inline_rich_response_enabled:
        return False
    if not rich_response_capable:
        # An approved candidate reaches the answer only through the rich-item
        # inventory, which the graph withholds from a request that never
        # advertised the capability. Discovering and verifying one anyway spends
        # a Brave call, a thumbnail batch and a billed vision call on output
        # that is discarded.
        return False
    return budget.may_image_search()


async def _collect_images(
    image_task: asyncio.Task[list[dict[str, Any]]] | None,
    budget: Any,
    wants_image: bool,
) -> list[dict[str, Any]]:
    from .image_verification_flow import record_image_outcome

    if image_task is None:
        cached = budget.image_result() if wants_image else []
        if not cached:
            # Explicit opt-out or a closed server gate. Bounded and unlogged:
            # the path was never asked to run, so nothing failed.
            record_image_outcome("skipped")
        return cached
    try:
        approved = await image_task
    except Exception:
        # The flow classifies and reports every expected failure itself, so
        # anything arriving here is a programming error. Factual research still
        # survives it.
        logger.exception("Image enrichment failed unexpectedly")
        approved = []
    budget.record_image_search(approved)
    return approved


async def _run_search(
    tavily_tool: Any | None,
    query: str,
    max_results: int | None,
    search_depth: str | None,
) -> str:
    tool = tavily_tool
    if tool is None:
        tool = await _resolve_tool("tavily", "tavily_search")
    if tool is None:
        raise RuntimeError("tavily_search is unavailable")
    args: dict[str, Any] = {"query": query}
    if max_results is not None:
        args["max_results"] = max_results
    if search_depth is not None:
        args["search_depth"] = search_depth
    return provider_result_text(await tool.ainvoke(args), tool_name="tavily_search")


async def _discover_and_verify(
    *,
    brave_tool: Any | None,
    web_image_service: Any | None,
    verifier_model: Any | None,
    user_request: str,
    image_query: str,
    factual_query: str,
    image_intent: str | None = None,
    recorder: Any | None = None,
) -> list[dict[str, Any]]:
    """Return public candidate dicts for approved images, or an empty list."""

    from .image_verification_flow import discover_and_verify_images

    if brave_tool is None:
        brave_tool = await _resolve_tool("brave_image_search", "brave_image_search")
    if web_image_service is None:
        web_image_service = _from_container("web_image_service")

    return await discover_and_verify_images(
        brave_tool=brave_tool,
        web_image_service=web_image_service,
        verifier_model=verifier_model,
        user_request=user_request,
        image_query=image_query,
        factual_query=factual_query,
        image_intent=image_intent,
        recorder=recorder,
    )


async def _resolve_tool(server_name: str, tool_name: str) -> Any | None:
    """Find one MCP tool, or None. Absence is reported by whoever needed it."""

    try:
        from .mcp_registry import get_global_mcp_manager

        manager = await get_global_mcp_manager()
        for tool in await manager.get_server_tools(server_name):
            if getattr(tool, "name", None) == tool_name:
                return tool
    except Exception as exc:
        logger.debug("MCP tool %s unavailable: %s", tool_name, type(exc).__name__)
    return None


def _from_container(provider_name: str) -> Any | None:
    """Resolve one DI provider off the process-wide container, or None.

    ``get_container()`` rather than ``Container()``: instantiating the
    declarative container builds a second ``Database`` singleton, and with it a
    second SQLAlchemy engine and connection pool, on every call.
    """

    try:
        from ..core.container import get_container

        return getattr(get_container(), provider_name)()
    except Exception as exc:
        logger.debug("DI provider %s unavailable: %s", provider_name, type(exc).__name__)
        return None


def _with_research_meta(search_text: str, *, reused: bool, budget: Any) -> str:
    try:
        payload = json.loads(search_text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return search_text
    if not isinstance(payload, dict):
        return search_text
    payload.pop("images", None)
    payload["research"] = {"reused": reused, "searches_used": budget.search_calls}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _budget_reused_payload(budget: Any) -> str:
    return json.dumps(
        {
            "research": {
                "reused": True,
                "searches_used": budget.search_calls,
                "note": (
                    "The per-turn research budget is spent. Answer from the results "
                    "already gathered in this turn; another search would return the same "
                    "sources."
                ),
            },
            "accumulated_results": budget.accumulated(),
        },
        ensure_ascii=False,
        indent=2,
    )


def _error_payload(message: str) -> str:
    return json.dumps(
        {
            "status": "error",
            "error_type": "provider_error",
            "retryable": True,
            "hint": "Research is temporarily unavailable. Say so rather than guessing.",
            "message": message[:500],
        },
        ensure_ascii=False,
    )
