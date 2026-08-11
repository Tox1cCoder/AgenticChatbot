"""The single model-facing research operation.

The model asks one question and may refine the visual subject. The server owns
the rest: whether to hit the network at all, whether to look for an image, and
which provider-selected results may be shown.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..core.config import settings
from .image_discovery_flow import discover_images, record_discovery_outcome
from .research_budget import get_research_budget
from .selected_image_sink import offer_selected_images
from .tool_context import get_tool_context
from .tool_result_rendering import provider_result_text
from .tool_scope import is_client_only_scope

logger = logging.getLogger(__name__)

_DESCRIPTION = (
    "Research the web. Returns ranked sources with URLs and relevant content for you "
    "to synthesize into the final answer.\n\n"
    "Use topic='news' for current events. Set time_range only when the user "
    "explicitly requests a recency window. time_range also bounds the picture: "
    "inside a declared window, an image whose page was last crawled before it is "
    "dropped.\n\n"
    "This tool automatically considers a provider-selected image. Set image_query only to "
    "make the visual subject more precise than the factual query: one concrete "
    "subject, no question words, plus a disambiguator or a form word (photo, "
    "diagram, map, chart) when it matters. Add the year or version when what "
    "matters is how the subject looks now — the image provider has no recency "
    "filter, so the query text is the only way to ask for a current picture.\n\n"
    "Set skip_images=true only when a visual cannot support the answer.\n\n"
    "Set image_intent='gallery' only when the user asks to see or compare several "
    "instances — a roster, a set of logos, colour options. Otherwise leave it "
    "unset: the default places up to two images beside the prose they support. "
    "Never state how many images you want; the layout decides.\n\n"
    "Selected images appear in your available rich items. Not every call produces "
    "one, and a complete answer never depends on an image. A gallery arrives as "
    "ONE grid item with one marker."
)


class ResearchProviderError(RuntimeError):
    """A structured provider failure surfaced through the research boundary."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        self.retryable = bool(retryable)
        super().__init__(str(message or "Research provider failed")[:300])


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
    topic: Literal["general", "news", "finance"] | None = Field(
        default=None,
        description="Optional Tavily topic for general, news, or finance research.",
    )
    time_range: Literal["day", "week", "month", "year"] | None = Field(
        default=None,
        description="Optional Tavily recency window when the requested recency is clear.",
    )
    max_results: int | None = Field(default=None, description="Optional result count.")
    search_depth: str | None = Field(default=None, description="Optional Tavily depth.")


def create_web_research_tool(
    *,
    tavily_tool: Any | None = None,
    brave_tool: Any | None = None,
    tool_scope: str | None = None,
) -> StructuredTool:
    """Build the ``web_research`` tool. Dependencies are injected in tests."""

    async def _research(
        query: str,
        image_query: str | None = None,
        image_intent: str | None = None,
        topic: Literal["general", "news", "finance"] | None = None,
        time_range: Literal["day", "week", "month", "year"] | None = None,
        max_results: int | None = None,
        search_depth: str | None = None,
        skip_images: bool = False,
    ) -> str:
        context = get_tool_context()
        bound_scope = str(getattr(tool_scope, "value", tool_scope) or "default")
        if bound_scope == "client_only" or is_client_only_scope(
            device_id=context.device_id,
            tool_scope=context.tool_scope,
        ):
            return _error_payload(
                "Server research is unavailable in client-only tool scope.",
                retryable=False,
                error_type="permission_error",
            )
        budget = get_research_budget(context.conversation_id)
        visual_query = str(image_query or query).strip()
        wants_image = bool(visual_query) and not skip_images
        tavily_scope = (topic, time_range, max_results, search_depth)

        reused = (
            budget.find_reuse(query, scope=tavily_scope)
            if settings.research_budget_enabled
            else None
        )
        search_task: asyncio.Task[str] | None = None
        if reused is None:
            # reserve_search claims the slot in one step; a bare check here would
            # race a concurrent web_research call across the await below.
            if settings.research_budget_enabled and not budget.reserve_search(
                query, scope=tavily_scope
            ):
                return _budget_reused_payload(budget)
            search_task = asyncio.create_task(
                _run_search(
                    tavily_tool,
                    query,
                    max_results,
                    search_depth,
                    topic,
                    time_range,
                )
            )

        image_task: asyncio.Task[list[dict[str, Any]]] | None = None
        if wants_image and _reserve_image_path(budget, context.rich_response_capable):
            image_task = asyncio.create_task(
                _discover_selected(
                    brave_tool=brave_tool,
                    image_query=visual_query,
                    image_intent=image_intent,
                    time_range=time_range,
                )
            )

        if search_task is not None:
            try:
                search_text = await search_task
            except asyncio.CancelledError:
                await _cancel_and_wait(image_task)
                raise
            except ResearchProviderError as exc:
                await _cancel_and_wait(image_task)
                logger.warning("Research provider failed: %s", exc)
                return _error_payload(str(exc), retryable=exc.retryable)
            except Exception as exc:
                await _cancel_and_wait(image_task)
                logger.warning("Research search failed: %s", exc)
                return _error_payload(str(exc))
            budget.record_search(query, search_text, scope=tavily_scope)
            search_reused = False
        else:
            search_text = reused or ""
            search_reused = True

        selected = await _collect_images(image_task, budget, wants_image)
        if selected:
            offer_selected_images(selected)
        return _with_research_meta(search_text, reused=search_reused, budget=budget)

    return StructuredTool.from_function(
        coroutine=_research,
        name="web_research",
        description=_DESCRIPTION,
        args_schema=WebResearchInput,
        metadata={
            "tool_origin": "internal",
            "qualified_tool_id": "internal::web_research",
            "tool_scope": str(tool_scope or "default"),
        },
    )


def _reserve_image_path(budget: Any, rich_response_capable: bool) -> bool:
    if not settings.remote_image_enrichment_enabled:
        return False
    if not settings.inline_rich_response_enabled:
        return False
    if not rich_response_capable:
        # A selected candidate reaches the answer only through the rich-item
        # inventory, which the graph withholds from a request that never
        # advertised the capability. Discovering one anyway spends a Brave call
        # on output that is discarded.
        return False
    return budget.reserve_image_search()


async def _collect_images(
    image_task: asyncio.Task[list[dict[str, Any]]] | None,
    budget: Any,
    wants_image: bool,
) -> list[dict[str, Any]]:
    if image_task is None:
        cached = budget.image_result() if wants_image else []
        if not cached:
            # Explicit opt-out or a closed server gate. Bounded and unlogged:
            # the path was never asked to run, so nothing failed.
            record_discovery_outcome("skipped")
        return cached
    try:
        selected = await image_task
    except Exception:
        # The flow classifies and reports expected provider failures itself, so
        # anything arriving here is a programming error. Factual research still
        # survives it.
        logger.exception("Image enrichment failed unexpectedly")
        selected = []
    budget.record_image_search(selected)
    return selected


async def _cancel_and_wait(task: asyncio.Task[Any] | None) -> None:
    """Cancel a sibling provider call and consume its terminal outcome."""

    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:
        # The factual failure remains the result of this combined operation;
        # awaiting here exists to finish and consume sibling cleanup.
        pass


async def _run_search(
    tavily_tool: Any | None,
    query: str,
    max_results: int | None,
    search_depth: str | None,
    topic: Literal["general", "news", "finance"] | None,
    time_range: Literal["day", "week", "month", "year"] | None,
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
    if topic is not None:
        args["topic"] = topic
    if time_range is not None:
        args["time_range"] = time_range
    return _raise_for_provider_error(
        provider_result_text(await tool.ainvoke(args), tool_name="tavily_search")
    )


async def _discover_selected(
    *,
    brave_tool: Any | None,
    image_query: str,
    image_intent: str | None = None,
    time_range: str | None = None,
) -> list[dict[str, Any]]:
    """Return provider-selected candidate dicts, or an empty list.

    The recency window the model declared for the facts also bounds the picture:
    a window that makes a month-old source stale makes a month-old photograph of
    the same subject stale too.
    """

    if brave_tool is None:
        brave_tool = await _resolve_tool("brave_image_search", "brave_image_search")
    return await discover_images(
        brave_tool=brave_tool,
        image_query=image_query,
        image_intent=image_intent,
        time_range=time_range,
    )


async def _resolve_tool(server_name: str, tool_name: str) -> Any | None:
    """Find one MCP tool, or None. Absence is reported by whoever needed it."""

    context = get_tool_context()
    if is_client_only_scope(
        device_id=context.device_id,
        tool_scope=context.tool_scope,
    ):
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


def _raise_for_provider_error(result_text: str) -> str:
    try:
        payload = json.loads(result_text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return result_text
    if isinstance(payload, dict) and payload.get("error"):
        raise ResearchProviderError(
            str(payload["error"]), retryable=bool(payload.get("retryable"))
        )
    return result_text


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


def _error_payload(
    message: str,
    *,
    retryable: bool = True,
    error_type: str = "provider_error",
) -> str:
    return json.dumps(
        {
            "status": "error",
            "error_type": error_type,
            "retryable": retryable,
            "hint": "Research is temporarily unavailable. Say so rather than guessing.",
            "message": message[:500],
        },
        ensure_ascii=False,
    )
