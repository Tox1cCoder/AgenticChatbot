"""
Shared tool search scoring logic for server and client catalogs.

Both McpToolCatalog and ClientToolCatalog use identical ranking logic.
Centralizing it here prevents drift and makes threshold changes easy to reason about.

Scoring model:
- Exact name match:    +100.0
- Name prefix match:   +50.0
- Name substring:      +30.0
- IDF-weighted token overlap with name/description/args: variable (never 0 unless no overlap)
- Description presence bonus: REMOVED (was unconditional +0.1; irrelevant to relevance)

Thresholds (configured via Settings):
- min_relevance_score: floor for including a result (tools below this are excluded)
- autoload_min_relevance_score: floor for autoloading a result
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from .text_normalization import tokenize_text
from .tool_search_profiles import (
    QueryIntent,
    ToolCapabilityProfile,
    infer_query_intent,
    infer_tool_profile,
)

_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "not",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


def score_tool(
    *,
    tool_name: str,
    description: str,
    arg_names: list[str],
    query_lower: str,
    query_tokens: set[str],
    doc_freq: Counter,
    total_docs: int,
) -> float:
    """
    Score a single tool against a query.

    Returns a non-negative float. Higher is more relevant.
    Returns 0.0 for zero token overlap and no name signal.
    """
    score = 0.0
    tool_name_lower = tool_name.lower()

    # Name-based signals (highest priority, position-independent)
    if tool_name_lower == query_lower:
        score += 100.0
    elif tool_name_lower.startswith(query_lower):
        score += 50.0
    elif query_lower in tool_name_lower:
        score += 30.0

    # IDF-weighted token overlap across name + description + args
    searchable = f"{tool_name} {description} {' '.join(arg_names)}"
    tool_tokens = set(tokenize_text(searchable))

    overlap = query_tokens & tool_tokens
    if overlap:
        for token in overlap:
            df = doc_freq.get(token, 1)
            idf = 1.0 / (1.0 + df / max(total_docs, 1))
            score += idf * 10.0

    # NOTE: no description presence bonus (+0.1 removed) — irrelevant to relevance
    return score


def build_query_tokens(query: str) -> tuple[str, set[str]]:
    """
    Normalize and tokenize a query for scoring.

    Returns:
        (query_lower, query_tokens_set)
    """
    query_lower = query.lower().strip()
    raw_tokens = set(tokenize_text(query))
    query_tokens = {token for token in raw_tokens if token not in _QUERY_STOPWORDS}
    if not query_tokens:
        query_tokens = raw_tokens
    return query_lower, query_tokens


def rank_and_filter(
    scored: list[tuple[Any, float]],
    *,
    min_relevance_score: float,
) -> list[tuple[Any, float]]:
    """
    Sort by score descending, then filter out below-threshold results.

    Args:
        scored: List of (tool_descriptor, score) pairs.
        min_relevance_score: Minimum score to include a tool.

    Returns:
        Sorted and filtered list.
    """
    # Sort by score descending, then by tool name for stability
    scored.sort(key=lambda x: (-x[1], getattr(x[0], "tool_name", "")))
    return [(tool, s) for tool, s in scored if s >= min_relevance_score]


# ---------------------------------------------------------------------------
# Intent-aware field-weighted ranking (Tool Search Accuracy plan)
#
# This is the new ranking API. It replaces raw token overlap with capability
# intent matching plus field-weighted lexical overlap (name/required-args/args/
# description). Callers receive confidence and autoload eligibility directly.
# The legacy score_tool()/rank_and_filter() above remain as a compatibility
# shim until all catalogs/tests migrate to rank_tool_candidates().
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSearchScore:
    tool: Any
    score: float
    confidence: str
    match_reasons: list[str]
    profile: ToolCapabilityProfile
    autoload_eligible: bool


def rank_tool_candidates(
    *,
    query: str,
    candidates: Iterable[Any],
    min_relevance_score: float | None = None,
    high_confidence_margin: float = 8.0,
) -> list[ToolSearchScore]:
    intent = infer_query_intent(query)
    scored = [_score_candidate(intent, tool) for tool in candidates]
    scored = [item for item in scored if item.score >= (min_relevance_score or 1.0)]
    scored.sort(key=lambda item: (-item.score, getattr(item.tool, "tool_name", "")))

    if not scored:
        return []

    top_score = scored[0].score
    second_score = scored[1].score if len(scored) > 1 else 0.0
    margin = top_score - second_score
    result: list[ToolSearchScore] = []
    for index, item in enumerate(scored):
        confidence = _confidence_for(item.score, margin if index == 0 else 0.0)
        result.append(
            ToolSearchScore(
                tool=item.tool,
                score=item.score,
                confidence=confidence,
                match_reasons=item.match_reasons[:2],
                profile=item.profile,
                autoload_eligible=(
                    index == 0 and confidence == "high" and margin >= high_confidence_margin
                ),
            )
        )
    return result


def _score_candidate(intent: QueryIntent, tool: Any) -> ToolSearchScore:
    tool_name = str(getattr(tool, "tool_name", getattr(tool, "name", "")) or "")
    profile = infer_tool_profile(
        tool_name=tool_name,
        server_name=str(getattr(tool, "server_name", "") or ""),
        description=str(getattr(tool, "description", "") or ""),
        arg_names=list(getattr(tool, "arg_names", []) or []),
        required_arg_names=list(getattr(tool, "required_arg_names", []) or []),
    )

    score = 0.0
    reasons: list[str] = []

    capability_overlap = intent.capabilities & profile.capabilities
    if capability_overlap:
        score += 40.0 * len(capability_overlap)
        reasons.append(f"matches {sorted(capability_overlap)[0]} intent")

    name_overlap = intent.tokens & profile.name_tokens
    if name_overlap:
        score += 14.0 * len(name_overlap)
        reasons.append(f"name match: {', '.join(sorted(name_overlap)[:2])}")

    required_arg_overlap = intent.tokens & profile.required_arg_tokens
    if required_arg_overlap:
        score += 12.0 * len(required_arg_overlap)
        reasons.append(f"required arg match: {', '.join(sorted(required_arg_overlap)[:2])}")

    arg_overlap = intent.tokens & profile.arg_tokens
    if arg_overlap:
        score += 6.0 * len(arg_overlap)
        reasons.append(f"arg match: {', '.join(sorted(arg_overlap)[:2])}")

    description_overlap = intent.tokens & profile.description_tokens
    if description_overlap:
        score += min(4.0, 1.0 * len(description_overlap))

    score += _capability_specific_adjustment(intent, profile)

    return ToolSearchScore(
        tool=tool,
        score=score,
        confidence=_confidence_for(score, 0.0),
        match_reasons=reasons or ["weak lexical match"],
        profile=profile,
        autoload_eligible=False,
    )


def _capability_specific_adjustment(
    intent: QueryIntent,
    profile: ToolCapabilityProfile,
) -> float:
    score = 0.0
    if "shell_exec" in intent.capabilities:
        if "shell_exec" in profile.capabilities:
            score += 35.0
        if "file_search" in profile.capabilities:
            score -= 18.0
        # Tuned vs plan: process_interaction is shell-adjacent, so under a
        # shell_exec intent it earns a positive bump (ranking it above
        # config-style tools) instead of a penalty. This matches the plan's
        # Target Public Result Shape, which shows interact_with_process as the
        # medium-confidence #2 result for "run shell command". config_read is
        # intentionally NOT penalized here: weak config matches stay visible at
        # a low score rather than being filtered out entirely.
        if "process_interaction" in profile.capabilities:
            score += 30.0

    if "file_write" in intent.capabilities:
        if "file_write" in profile.capabilities:
            score += 30.0
        if "file_edit" in profile.capabilities and "edit" not in intent.tokens and "patch" not in intent.tokens:
            score -= 10.0

    if "file_edit" in intent.capabilities and "file_edit" in profile.capabilities:
        score += 30.0

    if "file_search" in intent.capabilities and "file_search" in profile.capabilities:
        score += 30.0

    if "config_read" in intent.capabilities and "config_read" in profile.capabilities:
        score += 25.0

    return score


def _confidence_for(score: float, margin: float) -> str:
    if score >= 60.0 and margin >= 8.0:
        return "high"
    if score >= 25.0:
        return "medium"
    return "low"
