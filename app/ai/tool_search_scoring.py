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
from typing import Any

from .text_normalization import filter_stopwords, tokenize_text


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
    tool_tokens = set(filter_stopwords(list(tool_tokens)))

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
        (query_lower, query_tokens_set_without_stopwords)
    """
    query_lower = query.lower().strip()
    raw_tokens = tokenize_text(query)
    filtered = filter_stopwords(raw_tokens)
    # Keep the filtered set; if all tokens were stopwords, fall back to raw
    query_tokens = set(filtered) if filtered else set(raw_tokens)
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
