"""Budgeted, structure-aware previews for offloaded tool results.

A blind character prefix of a JSON tool result keeps whichever key happens to
be serialized first and silently discards the rest. Research payloads paid for
that: the model received image metadata and no facts, and re-ran the search.
This module spends the preview budget on the fields that answer the question.

The builder runs against every offloaded tool output, not only web search, so it
must not assume a search shape. It keeps every scalar and every string of a
result entry and shortens only long text, which is why ``tavily_extract`` keeps
its ``raw_content`` and ``tavily_map`` keeps its plain-string entries.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_IDENTITY_KEYS = ("provider", "operation", "query", "tool", "total_results")
_NEVER_TRUNCATED_KEYS = frozenset({"index", "title", "url", "score"})

# json.dumps(..., ensure_ascii=False) writes each of these as two characters.
_TWO_CHAR_ESCAPES = frozenset('"\\\n\r\t\b\f')


@dataclass(frozen=True, slots=True)
class ToolResultPreview:
    """The inline text for an offloaded result and what it left out."""

    text: str
    omitted_arrays: tuple[tuple[str, int], ...] = ()
    omitted_results: int = 0
    structured: bool = False
    omitted_keys: tuple[str, ...] = ()
    omitted_result_keys: tuple[str, ...] = ()
    shortened_keys: tuple[str, ...] = ()


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
    shell, answer_shortened = _build_shell(parsed, budget=budget, answer_share=answer_share)
    omitted_arrays, omitted_keys = _omitted_top_level(parsed, shell)

    floor = max(1, int(min_result_content_chars))
    kept, share = _fit_results(results, shell=shell, budget=budget, floor=floor)
    originals = results[:kept]
    shell["results"] = [_shrink_result(entry, share) for entry in originals]
    dropped_result_keys, shortened_result_keys = _result_losses(originals, shell["results"])

    text = _dump(shell)
    if len(text) > budget:
        # Reached only when the shell alone (identity keys plus the capped
        # answer) overflows the budget, so _fit_results kept nothing: for any
        # kept > 0 the allocation is measured in serialized characters and
        # therefore fits exactly. Slicing the JSON dump here would produce
        # invalid text mislabeled structured=True, so fall back to a plain
        # character prefix, and say so — a silent fallback makes the whole
        # feature inert without any signal.
        logger.warning(
            "Structured tool-result preview overflowed its %d-char budget with "
            "%d of %d results kept; falling back to a character prefix.",
            budget,
            kept,
            len(results),
        )
        return ToolResultPreview(text=output_text[:budget].rstrip(), structured=False)

    return ToolResultPreview(
        text=text,
        omitted_arrays=omitted_arrays,
        omitted_results=max(0, len(results) - kept),
        structured=True,
        omitted_keys=omitted_keys,
        omitted_result_keys=dropped_result_keys,
        shortened_keys=answer_shortened + shortened_result_keys,
    )


def _parse_object_with_results(output_text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(output_text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(parsed, dict) or not isinstance(parsed.get("results"), list):
        return None
    return parsed


def _build_shell(
    parsed: dict[str, Any], *, budget: int, answer_share: float
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Build the non-results portion of the preview: identity scalars and answer."""

    shell: dict[str, Any] = {"results": []}
    for key in _IDENTITY_KEYS:
        value = parsed.get(key)
        if value is not None and not isinstance(value, (list, dict)):
            shell[key] = value

    answer = parsed.get("answer")
    if not isinstance(answer, str) or not answer:
        return shell, ()

    kept = _encoded_prefix(answer, max(1, int(budget * float(answer_share))))
    shell["answer"] = kept
    return shell, ("answer",) if len(kept) < len(answer) else ()


def _omitted_top_level(
    parsed: dict[str, Any], shell: dict[str, Any]
) -> tuple[tuple[tuple[str, int], ...], tuple[str, ...]]:
    """Report every top-level key the preview drops, not only the arrays.

    Under-reporting is what made a lossy preview misleading: a notice that named
    two omitted arrays while silently discarding ``usage`` and ``request_id``
    told the model its omissions were trivial.
    """

    arrays: list[tuple[str, int]] = []
    keys: list[str] = []
    for key, value in parsed.items():
        if key == "results" or key in shell:
            continue
        if _is_empty(value):
            continue
        if isinstance(value, list):
            arrays.append((key, len(value)))
        else:
            keys.append(key)
    return tuple(arrays), tuple(keys)


def _is_empty(value: Any) -> bool:
    """Whether dropping ``value`` loses nothing worth reporting."""

    return value is None or (isinstance(value, (str, list, dict)) and not value)


def _fit_results(
    results: list[Any], *, shell: dict[str, Any], budget: int, floor: int
) -> tuple[int, int]:
    """Return how many results fit in the budget and the content chars each gets.

    Per-result JSON overhead (keys, quotes, commas, indentation) is measured by
    dumping the shell with empty-string content placeholders rather than
    estimated, because that punctuation is independent of content length. A
    naive estimate based on the empty-results shell alone undercounts it and
    produces a preview that overflows ``budget`` once real content is filled in.

    The search is a bisection over a bounded range rather than a walk down from
    ``len(results)``. An answer needs ``kept * floor <= content_budget <
    budget``, so ``kept`` can never exceed ``budget // floor``; walking from
    ``len(results)`` re-dumped the scaffold once per surplus entry, which cost
    26 seconds of blocked event loop on an 8000-entry payload.
    """

    high = min(len(results), max(1, budget // floor))
    low = 1
    best = (0, floor)
    while low <= high:
        # share() is non-increasing in kept: more entries mean less content
        # budget shared between more of them, so the fit predicate is monotone
        # and bisection finds the same answer the linear walk did.
        middle = (low + high) // 2
        share = _content_share(results, shell=shell, budget=budget, kept=middle)
        if share >= floor:
            best = (middle, share)
            low = middle + 1
        else:
            high = middle - 1
    return best


def _content_share(results: list[Any], *, shell: dict[str, Any], budget: int, kept: int) -> int:
    scaffold = dict(shell)
    scaffold["results"] = [_shrink_result(entry, 0) for entry in results[:kept]]
    return (budget - len(_dump(scaffold))) // kept


def _shrink_result(entry: Any, content_chars: int) -> Any:
    """Shrink one entry to at most ``content_chars`` serialized text characters.

    Every scalar survives and only long text is shortened, so an entry shape the
    builder has never seen keeps its substance. Nested containers are dropped
    because their size is unbounded and unbudgetable; the notice names them.
    """

    if isinstance(entry, str):
        return _encoded_prefix(entry, content_chars)
    if not isinstance(entry, dict):
        return entry

    retained: dict[str, Any] = {}
    shrinkable: dict[str, str] = {}
    for key, value in entry.items():
        if isinstance(value, (list, dict)):
            continue
        if isinstance(value, str) and key not in _NEVER_TRUNCATED_KEYS:
            shrinkable[key] = value
        retained[key] = value

    fitted = _fit_text_values(shrinkable, content_chars)
    return {key: fitted.get(key, value) for key, value in retained.items()}


def _fit_text_values(values: dict[str, str], budget: int) -> dict[str, str]:
    """Spend ``budget`` serialized characters across ``values``, shortest first.

    Shortest-first means a short field survives whole and its surplus flows to
    the long ones, so a 40 KB ``raw_content`` is shortened while the
    30-character ``published_date`` beside it is not.
    """

    if not values:
        return {}
    remaining = max(0, budget)
    ordered = sorted(values.items(), key=lambda item: _encoded_length(item[1], remaining))
    fitted: dict[str, str] = {}
    for position, (key, text) in enumerate(ordered):
        share = remaining // (len(ordered) - position)
        kept = _encoded_prefix(text, share)
        fitted[key] = kept
        remaining -= _encoded_length(kept, share)
    return fitted


def _result_losses(
    originals: list[Any], shrunk: list[Any]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Name the keys dropped from, and shortened inside, the kept entries."""

    dropped: dict[str, None] = {}
    shortened: dict[str, None] = {}
    for original, final in zip(originals, shrunk, strict=True):
        if not isinstance(original, dict) or not isinstance(final, dict):
            continue
        for key, value in original.items():
            if key not in final:
                dropped[key] = None
            elif isinstance(value, str) and len(final[key]) < len(value):
                shortened[key] = None
    return tuple(dropped), tuple(shortened)


def _encoded_prefix(text: str, budget: int) -> str:
    """Longest prefix of ``text`` whose serialized form fits ``budget`` chars.

    The budget is spent on serialized characters, and json.dumps expands a
    quote, a backslash or a newline to two. Measuring raw characters here
    overflowed the budget on ordinary scraped prose — two quotes anywhere in the
    retained text were enough — and threw the whole structured preview away.
    """

    if budget <= 0:
        return ""
    total = 0
    for index, char in enumerate(text):
        total += _escaped_char_cost(char)
        if total > budget:
            return text[:index]
    return text


def _encoded_length(text: str, cap: int) -> int:
    """Serialized length of ``text``, abandoned once it passes ``cap``.

    Bounding the walk keeps the cost proportional to the budget rather than to
    a multi-megabyte field that will be shortened to a few hundred characters.
    """

    total = 0
    for char in text:
        total += _escaped_char_cost(char)
        if total > cap:
            return total
    return total


def _escaped_char_cost(char: str) -> int:
    if char in _TWO_CHAR_ESCAPES:
        return 2
    if char < " ":
        return 6  # json writes any other control character as \uXXXX
    return 1


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)
