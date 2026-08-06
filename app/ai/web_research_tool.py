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
from .verified_image_sink import offer_verified_images

logger = logging.getLogger(__name__)

_DESCRIPTION = (
    "Research the web. Returns a synthesized answer plus ranked sources with URLs.\n\n"
    "Set image_query to a short, concrete visual subject when a picture would help "
    "the reader see what the answer is about — a product, device, place, building, "
    "artwork, organism, vehicle, or screen. Write the subject yourself: no question "
    "words, one subject, plus a disambiguator or a form word (photo, diagram, map, "
    "chart) when it matters.\n\n"
    "Leave image_query unset for abstract subjects (code, math, policy, definitions, "
    "planning) and whenever you are unsure whether an image would help. An uncertain "
    "image decision uses no image_query at all.\n\n"
    "Set image_intent='gallery' when the user asks to SEE several instances or to "
    "compare things — a roster, a set of logos, colour options, a lineup. Otherwise "
    "leave it unset: the default places up to two images beside the prose they "
    "support. Never state how many images you want; the layout decides, and only "
    "images verified against the subject survive.\n\n"
    "Approved images appear in your available rich items. Not every image_query "
    "produces one, and a complete answer never depends on an image. A gallery "
    "arrives as ONE grid item with one marker."
)


class WebResearchInput(BaseModel):
    query: str = Field(description="The factual research query.")
    image_query: str | None = Field(
        default=None,
        description="Short concrete visual subject, or omit when an image would not help.",
    )
    image_intent: Literal["figure", "gallery"] | None = Field(
        default=None,
        description=(
            "Layout: 'figure' (default) for up to two images beside the prose, "
            "'gallery' for a grid when the user asks to see several instances or "
            "to compare things. Never state a count."
        ),
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
    ) -> str:
        conversation_id = get_tool_context().conversation_id
        budget = get_research_budget(conversation_id)
        wants_image = bool(str(image_query or "").strip())

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
        if wants_image and _image_path_open(budget):
            image_task = asyncio.create_task(
                _discover_and_verify(
                    brave_tool=brave_tool,
                    web_image_service=web_image_service,
                    verifier_model=verifier_model,
                    user_request=query,
                    image_query=str(image_query).strip(),
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


def _image_path_open(budget: Any) -> bool:
    if not settings.vision_image_verification_enabled:
        return False
    if not settings.inline_rich_response_enabled:
        return False
    return budget.may_image_search()


async def _collect_images(
    image_task: asyncio.Task[list[dict[str, Any]]] | None,
    budget: Any,
    wants_image: bool,
) -> list[dict[str, Any]]:
    if image_task is None:
        return budget.image_result() if wants_image else []
    try:
        approved = await image_task
    except Exception as exc:
        logger.debug("Image path abandoned: %s", type(exc).__name__)
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
    return str(await tool.ainvoke(args))


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

    try:
        async with asyncio.timeout(float(settings.image_verification_deadline_seconds)):
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
    except Exception as exc:
        logger.debug("Image verification abandoned: %s", type(exc).__name__)
        return []


async def _resolve_tool(server_name: str, tool_name: str) -> Any | None:
    try:
        from .mcp_registry import get_global_mcp_manager

        manager = get_global_mcp_manager()
        for tool in await manager.get_server_tools(server_name):
            if getattr(tool, "name", None) == tool_name:
                return tool
    except Exception as exc:
        logger.warning("MCP tool %s unavailable: %s", tool_name, exc)
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
