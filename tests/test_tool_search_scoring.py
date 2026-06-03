"""Tests for shared tool search scoring logic (Phase 2)."""

from __future__ import annotations

from collections import Counter

import pytest

from app.ai.tool_search_scoring import (
    build_query_tokens,
    rank_and_filter,
    score_tool,
)

# ---------------------------------------------------------------------------
# score_tool: basic scoring contracts
# ---------------------------------------------------------------------------


def test_exact_name_match_scores_highest():
    doc_freq: Counter = Counter()
    s = score_tool(
        tool_name="search",
        description="Web search tool.",
        arg_names=["query"],
        query_lower="search",
        query_tokens={"search"},
        doc_freq=doc_freq,
        total_docs=1,
    )
    assert s >= 100.0


def test_zero_overlap_scores_zero_without_name_signal():
    """A query with no token overlap and no name match must score 0."""
    doc_freq: Counter = Counter({"banana": 1, "fruit": 1})
    s = score_tool(
        tool_name="execute_shell",
        description="Run shell commands on the host machine.",
        arg_names=["command"],
        query_lower="banana",
        query_tokens={"banana"},  # 'banana' not in tool tokens
        doc_freq=doc_freq,
        total_docs=5,
    )
    # 'banana' does not appear in tool_name, description, or arg_names
    assert s == 0.0


def test_description_presence_no_bonus():
    """Removing the +0.1 description bonus: two tools with identical names but
    one has a description should NOT differ solely because of the bonus."""
    doc_freq: Counter = Counter()
    s_with_desc = score_tool(
        tool_name="unique_tool_xyz",
        description="Some description present here.",
        arg_names=[],
        query_lower="unique_tool_xyz",
        query_tokens={"unique", "tool", "xyz"},
        doc_freq=doc_freq,
        total_docs=1,
    )
    s_no_desc = score_tool(
        tool_name="unique_tool_xyz",
        description="",
        arg_names=[],
        query_lower="unique_tool_xyz",
        query_tokens={"unique", "tool", "xyz"},
        doc_freq=doc_freq,
        total_docs=1,
    )
    # Both should have the same base name score (100 for exact match)
    # Any difference must come from token overlap (description tokens),
    # NOT from the removed +0.1 bonus
    # With no description tokens matching query tokens, scores must be equal
    assert s_with_desc == s_no_desc


# ---------------------------------------------------------------------------
# build_query_tokens: stopword filtering
# ---------------------------------------------------------------------------


def test_build_query_tokens_filters_stopwords():
    _, tokens = build_query_tokens("search the web for news")
    # 'the', 'for' are stopwords
    assert "the" not in tokens
    assert "for" not in tokens
    # 'search', 'web', 'news' are meaningful
    assert "search" in tokens


def test_build_query_tokens_falls_back_when_all_stopwords():
    """If query is entirely stopwords, fall back to raw tokens."""
    query_lower, tokens = build_query_tokens("to be or not to be")
    assert len(tokens) > 0


# ---------------------------------------------------------------------------
# rank_and_filter: threshold enforcement
# ---------------------------------------------------------------------------


class _FakeTool:
    def __init__(self, name: str):
        self.tool_name = name


def test_rank_and_filter_excludes_below_threshold():
    scored = [
        (_FakeTool("strong_tool"), 5.0),
        (_FakeTool("weak_tool"), 0.1),
        (_FakeTool("zero_tool"), 0.0),
    ]
    result = rank_and_filter(scored, min_relevance_score=0.5)
    names = [t.tool_name for t, _ in result]
    assert "strong_tool" in names
    assert "weak_tool" not in names
    assert "zero_tool" not in names


def test_rank_and_filter_empty_when_all_below_threshold():
    scored = [
        (_FakeTool("a"), 0.3),
        (_FakeTool("b"), 0.4),
    ]
    result = rank_and_filter(scored, min_relevance_score=0.5)
    assert result == []


def test_rank_and_filter_sorts_descending():
    scored = [
        (_FakeTool("low"), 1.0),
        (_FakeTool("high"), 50.0),
        (_FakeTool("mid"), 10.0),
    ]
    result = rank_and_filter(scored, min_relevance_score=0.0)
    names = [t.tool_name for t, _ in result]
    assert names == ["high", "mid", "low"]


# ---------------------------------------------------------------------------
# Phase 2 integration: is_loaded truthful when no autoload threshold met
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_is_loaded_false_when_autoload_score_below_threshold(monkeypatch):
    """Tools that score below autoload_min_relevance_score must have is_loaded=False
    even if they appear in search results (above min_relevance_score)."""
    from app.ai.mcp_tool_catalog import ToolDescriptor
    from app.ai.tool_context import ToolContext
    from app.ai.tool_search_tool import _execute_tool_search

    class FakeServerCatalog:
        def search_scored(self, query=None, top_k=5, server_name=None, allowlist=None):
            # Return a tool with a very low score (above min_relevance but below autoload_min)
            desc = ToolDescriptor(
                tool_name="banana_tool",
                server_name="exotic_server",
                description="A tool about bananas.",
                arg_names=[],
                required_arg_names=[],
                schema_fingerprint="fp-banana",
            )
            return [(desc, 0.6)]  # above min_relevance=0.5, below autoload_min=2.0

        def search(self, query=None, top_k=5, server_name=None, allowlist=None):
            desc = ToolDescriptor(
                tool_name="banana_tool",
                server_name="exotic_server",
                description="A tool about bananas.",
                arg_names=[],
                required_arg_names=[],
                schema_fingerprint="fp-banana",
            )
            return [desc]

        def is_ambiguous(self, tool_name):
            return False

        def get_server_inventory(self, allowlist=None):
            return []

    class DeferredStateStub:
        def __init__(self):
            self.autoload_calls = 0

        def autoload(self, **kwargs):
            self.autoload_calls += 1
            return []  # nothing actually loaded

        def autoload_client_tools(self, **kwargs):
            return []

    deferred_state = DeferredStateStub()

    async def fake_get_global_mcp_manager():
        return object()

    async def fake_get_tool_catalog(_manager):
        return FakeServerCatalog()

    monkeypatch.setattr(
        "app.ai.tool_search_tool.get_global_mcp_manager", fake_get_global_mcp_manager
    )
    monkeypatch.setattr("app.ai.tool_search_tool.get_tool_catalog", fake_get_tool_catalog)
    monkeypatch.setattr("app.ai.tool_search_tool.get_deferred_tool_state", lambda: deferred_state)
    monkeypatch.setattr(
        "app.ai.tool_search_tool.get_tool_context",
        lambda: ToolContext(
            conversation_id="conv-1",
            user_id=None,
            agent_key="chat",
            device_id=None,
        ),
    )

    # Set thresholds: min=0.5, autoload_min=2.0
    monkeypatch.setattr("app.ai.tool_search_tool.settings.mcp_tool_search_min_relevance_score", 0.5)
    monkeypatch.setattr(
        "app.ai.tool_search_tool.settings.mcp_tool_search_autoload_min_relevance_score", 2.0
    )

    result = await _execute_tool_search(query="banana")

    # Tool appears in results (score 0.6 >= min_relevance 0.5)
    assert len(result["results"]) == 1
    assert result["results"][0]["tool_name"] == "banana_tool"
    # But is_loaded must be False (score 0.6 < autoload_min 2.0)
    assert result["results"][0]["is_loaded"] is False
    # And deferred state autoload was never called (or called with empty refs)
    assert deferred_state.autoload_calls == 0


def test_exact_tool_name_query_is_high_confidence_and_autoload_eligible():
    """Searching a tool by its exact/near-exact name is a high-confidence match.

    Deferred discovery relies on autoload: an agent (notably a custom agent with
    no pinned tools) can only invoke a server tool after tool_search autoloads
    it. Autoload only fires for the high-confidence top result, so a query that
    names the tool must rank ``high`` and be ``autoload_eligible`` — otherwise
    the tool is never bound and the agent gives up / hands off.
    """
    from app.ai.mcp_tool_catalog import ToolDescriptor
    from app.ai.tool_search_scoring import rank_tool_candidates

    tools = [
        ToolDescriptor(
            tool_name="tavily_search",
            server_name="tavily",
            description="Search the web for current news and information.",
            arg_names=["query", "max_results"],
            required_arg_names=["query"],
            schema_fingerprint="fp-tavily",
        ),
        ToolDescriptor(
            tool_name="vector_store_search",
            server_name="rag",
            description="Search an internal vector store of documents.",
            arg_names=["query"],
            required_arg_names=["query"],
            schema_fingerprint="fp-vector",
        ),
    ]

    ranked = rank_tool_candidates(query="tavily_search", candidates=tools)

    assert ranked[0].tool.tool_name == "tavily_search"
    assert ranked[0].confidence == "high"
    assert ranked[0].autoload_eligible is True


def test_single_generic_name_token_match_does_not_reach_autoload():
    """A one-token generic name match must stay below the autoload bar.

    Covering a single common name token (e.g. query "search" vs a tool literally
    named "search") is a weaker signal than naming a specific multi-token tool,
    and must not be auto-bound on its own.
    """
    from app.ai.mcp_tool_catalog import ToolDescriptor
    from app.ai.tool_search_scoring import rank_tool_candidates

    tools = [
        ToolDescriptor(
            tool_name="search",
            server_name="generic",
            description="A generic search tool.",
            arg_names=["query"],
            required_arg_names=["query"],
            schema_fingerprint="fp-generic-search",
        ),
    ]

    ranked = rank_tool_candidates(query="search", candidates=tools)

    assert ranked[0].tool.tool_name == "search"
    assert ranked[0].autoload_eligible is False


def test_score_metadata_marks_weak_description_only_matches_low_confidence():
    from app.ai.mcp_tool_catalog import ToolDescriptor
    from app.ai.tool_search_scoring import rank_tool_candidates

    tools = [
        ToolDescriptor(
            tool_name="get_config",
            server_name="desktop_commander",
            description="Configuration includes blocked shell commands.",
            arg_names=[],
            required_arg_names=[],
            schema_fingerprint="fp-config",
        ),
        ToolDescriptor(
            tool_name="start_process",
            server_name="desktop_commander",
            description="Start a terminal process.",
            arg_names=["command", "timeout_ms", "shell"],
            required_arg_names=["command"],
            schema_fingerprint="fp-process",
        ),
    ]

    ranked = rank_tool_candidates(query="run shell command", candidates=tools)

    assert ranked[0].tool.tool_name == "start_process"
    assert ranked[0].confidence == "high"
    assert ranked[1].tool.tool_name == "get_config"
    assert ranked[1].confidence in {"low", "medium"}
    assert ranked[1].autoload_eligible is False
