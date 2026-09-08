"""Keep the OpenAI tool payload inside the limits that provider enforces.

OpenAI validates the `tools` array before it generates anything and rejects the
*whole request* with 400 `string_above_max_length` if any
`function.description` exceeds 1024 characters. Gemini has no equivalent limit,
so a description written long enough to guide Gemini well makes every
OpenAI call with tools fail — while the configured fallback quietly answers on
Gemini instead and the only trace is a warning saying "after a provider error".

That is what happened on 2026-09-08: three of the search agent's eight tools
were over the limit (2547, 1285 and 3027 characters), so every tool-bearing
call 400'd in under two seconds and only the tool-free forced-synthesis call
ever succeeded on OpenAI.

This clamp is a safety net, not a cure. A truncated description is degraded
guidance, and the right fix for a description that needs 3000 characters is to
write a shorter one. :func:`oversized_tool_descriptions` exists so that debt
can be reported rather than silently absorbed.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "OPENAI_FUNCTION_DESCRIPTION_MAX_CHARS",
    "clamp_openai_tool_descriptions",
    "oversized_tool_descriptions",
]

#: OpenAI's documented maximum for `tools[].function.description`. Exceeding it
#: is a 400 on the entire request, not a warning and not a per-tool omission.
OPENAI_FUNCTION_DESCRIPTION_MAX_CHARS = 1024

#: Where a truncated description may end. Ordered by preference: a paragraph
#: break reads as a deliberate stop, a sentence end almost as well, and a word
#: boundary is merely better than cutting mid-token.
_BOUNDARIES = ("\n\n", ". ", ".\n", "; ", "\n", " ")

#: How far back from the limit a boundary may be found before a hard cut is
#: preferable. Searching the whole string would let one early full stop throw
#: away most of the guidance.
_BOUNDARY_SEARCH_WINDOW = 320


def _as_openai_spec(tool: Any) -> dict[str, Any]:
    """One tool in OpenAI's wire shape, without touching the original.

    ``BaseTool`` objects are shared and cached across providers, so converting
    a copy is the difference between clamping a payload and shortening the
    description Gemini receives too.
    """
    if isinstance(tool, dict):
        return copy.deepcopy(tool)

    from langchain_core.utils.function_calling import convert_to_openai_tool

    return convert_to_openai_tool(tool)


def _truncate(description: str) -> str:
    """Shorten to the limit, ending at the latest sensible boundary."""
    hard_cut = description[:OPENAI_FUNCTION_DESCRIPTION_MAX_CHARS]
    floor = max(0, OPENAI_FUNCTION_DESCRIPTION_MAX_CHARS - _BOUNDARY_SEARCH_WINDOW)

    for boundary in _BOUNDARIES:
        index = hard_cut.rfind(boundary)
        if index >= floor:
            # Keep the punctuation, drop the separator that followed it.
            kept = hard_cut[: index + len(boundary)].rstrip()
            if kept:
                return kept
    return hard_cut


def oversized_tool_descriptions(tools: list[Any]) -> dict[str, int]:
    """Tool name to description length, for descriptions over the limit.

    The length is reported, not just the name: "over by 261" and "over by 2003"
    are different amounts of rewriting, and an operator deciding what to shorten
    needs to know which.
    """
    report: dict[str, int] = {}
    for tool in tools:
        function = _as_openai_spec(tool).get("function") or {}
        description = function.get("description")
        if not isinstance(description, str):
            continue
        if len(description) > OPENAI_FUNCTION_DESCRIPTION_MAX_CHARS:
            report[str(function.get("name") or "?")] = len(description)
    return report


def clamp_openai_tool_descriptions(tools: list[Any]) -> list[dict[str, Any]]:
    """The tool payload, in OpenAI's shape, with every description within limit.

    Returns dicts rather than the original objects so nothing shared is
    mutated. A description already within the limit is passed through
    byte-for-byte: a clamp that rewrites compliant text would be an unannounced
    prompt change.
    """
    clamped: list[dict[str, Any]] = []
    for tool in tools:
        spec = _as_openai_spec(tool)
        function = spec.get("function")
        if not isinstance(function, dict):
            clamped.append(spec)
            continue

        description = function.get("description")
        if (
            isinstance(description, str)
            and len(description) > OPENAI_FUNCTION_DESCRIPTION_MAX_CHARS
        ):
            shortened = _truncate(description)
            logger.warning(
                "Truncated the OpenAI description for tool %r from %d to %d characters; "
                "OpenAI rejects the whole request above %d. Shorten the tool's own "
                "description rather than relying on this.",
                function.get("name"),
                len(description),
                len(shortened),
                OPENAI_FUNCTION_DESCRIPTION_MAX_CHARS,
            )
            function = {**function, "description": shortened}
            spec = {**spec, "function": function}

        clamped.append(spec)
    return clamped
