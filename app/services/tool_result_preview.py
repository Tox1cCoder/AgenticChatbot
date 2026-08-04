"""Budgeted, structure-aware previews for offloaded tool results.

A blind character prefix of a JSON tool result keeps whichever key happens to
be serialized first and silently discards the rest. Research payloads paid for
that: the model received image metadata and no facts, and re-ran the search.
This module spends the preview budget on the fields that answer the question.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

_IDENTITY_KEYS = ("provider", "operation", "query", "tool", "total_results")
_RESULT_VERBATIM_KEYS = ("index", "title", "url", "score")


@dataclass(frozen=True, slots=True)
class ToolResultPreview:
    """The inline text for an offloaded result and what it left out."""

    text: str
    omitted_arrays: tuple[tuple[str, int], ...] = ()
    omitted_results: int = 0
    structured: bool = False


def build_tool_result_preview(
    output_text: str,
    *,
    budget_chars: int,
    answer_share: float = 0.25,
    min_result_content_chars: int = 200,
) -> ToolResultPreview:
    """Return a preview of at most ``budget_chars`` characters."""

    budget = max(1, int(budget_chars))
    parsed = _parse_object_with_results(output_text)
    if parsed is None:
        return ToolResultPreview(text=output_text[:budget].rstrip())

    results = parsed["results"]
    shell = _build_shell(parsed, budget=budget, answer_share=answer_share)
    omitted_arrays = _omitted_arrays(parsed)

    floor = max(1, int(min_result_content_chars))
    kept, share = _fit_results(results, shell=shell, budget=budget, floor=floor)
    shell["results"] = [_shrink_result(entry, share) for entry in results[:kept]]

    text = _dump(shell)
    if len(text) > budget:
        # Even an empty results array doesn't fit the shell (identity keys +
        # capped answer) within budget. _fit_results guarantees any kept > 0
        # allocation fits exactly, so this only fires when kept == 0 and the
        # bare shell itself overflows. Slicing the JSON here would produce
        # invalid text mislabeled structured=True; falling back to a plain
        # character prefix of the original output is honest about what fits.
        return ToolResultPreview(text=output_text[:budget].rstrip(), structured=False)

    return ToolResultPreview(
        text=text,
        omitted_arrays=omitted_arrays,
        omitted_results=max(0, len(results) - kept),
        structured=True,
    )


def _parse_object_with_results(output_text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(output_text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(parsed, dict) or not isinstance(parsed.get("results"), list):
        return None
    return parsed


def _build_shell(parsed: dict[str, Any], *, budget: int, answer_share: float) -> dict[str, Any]:
    """Build the non-results portion of the preview: identity scalars and answer."""

    shell: dict[str, Any] = {"results": []}
    for key in _IDENTITY_KEYS:
        value = parsed.get(key)
        if value is not None and not isinstance(value, (list, dict)):
            shell[key] = value

    answer = parsed.get("answer")
    if isinstance(answer, str) and answer:
        shell["answer"] = answer[: max(1, int(budget * float(answer_share)))]
    return shell


def _omitted_arrays(parsed: dict[str, Any]) -> tuple[tuple[str, int], ...]:
    return tuple(
        (key, len(value))
        for key, value in parsed.items()
        if key != "results" and isinstance(value, list)
    )


def _fit_results(
    results: list[Any], *, shell: dict[str, Any], budget: int, floor: int
) -> tuple[int, int]:
    """Return how many results fit in the budget and the content chars each gets.

    Per-result JSON overhead (keys, quotes, commas, indentation) is measured by
    dumping the shell with empty-string content placeholders rather than
    estimated, because that punctuation is independent of content length. A
    naive estimate based on the empty-results shell alone undercounts it and
    produces a preview that overflows ``budget`` once real content is filled in.
    """

    kept = len(results)
    while kept > 0:
        scaffold = dict(shell)
        scaffold["results"] = [_shrink_result(entry, 0) for entry in results[:kept]]
        content_budget = budget - len(_dump(scaffold))
        share = content_budget // kept
        if share >= floor:
            return kept, share
        kept -= 1
    return 0, floor


def _shrink_result(entry: Any, content_chars: int) -> dict[str, Any]:
    source = entry if isinstance(entry, dict) else {}
    shrunk: dict[str, Any] = {
        key: source[key] for key in _RESULT_VERBATIM_KEYS if key in source
    }
    content = source.get("content")
    if isinstance(content, str):
        shrunk["content"] = content[:content_chars]
    return shrunk


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)
