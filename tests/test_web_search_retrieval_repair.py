"""Regressions for the Worlds-results trace's incomplete retrieval."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.ai.research_budget import ResearchBudget, research_budget_from_state
from app.ai.tool_context import tool_execution_context
from app.ai.web_query_contract import WebSearchRequest, normalize_web_search, tavily_search_args
from app.ai.web_research.contracts import ResearchRequest, ResearchScope, SourceRecord
from app.ai.web_research.grounding import GroundingParser
from app.ai.web_research.providers import ProviderResolver, TavilyTextSearchProvider
from app.ai.web_research.service import WebResearchService
from app.ai.web_tools import create_web_search_tool

NOW = datetime(2026, 10, 26, 8, tzinfo=UTC)
SCOPE = ResearchScope(conversation_id="retrieval", user_id="test", logical_turn_id="turn")


class SearchBackend:
    """Stand in for the external MCP call, leaving the real adapter intact."""

    def __init__(self):
        self.calls: list[dict] = []
        self.next_results: list[dict] | None = None

    async def ainvoke(self, args):
        self.calls.append(args)
        results = self.next_results
        self.next_results = None
        if results is None:
            results = [
                {
                    "url": f"https://scores.example/{len(self.calls)}/{rank}",
                    "title": "Tournament evidence",
                    "content": "Completed fixture: Example A 2-1 Example B (test data).",
                }
                for rank in range(args["max_results"])
            ]
        return json.dumps({"results": results})


def session_for(backend, budget=None, *, mode="quick"):
    return WebResearchService(
        resolver=ProviderResolver(text=(TavilyTextSearchProvider(backend),)), now=lambda: NOW
    ).new_session(SCOPE, budget or ResearchBudget(), mode=mode)


async def search(session, query, **kwargs):
    with tool_execution_context(web_research_session=session):
        return json.loads(
            await create_web_search_tool().ainvoke(
                {"query": query, "objective": "Find completed match scores", **kwargs}
            )
        )


@pytest.mark.asyncio
async def test_full_first_search_does_not_block_targeted_results_retrieval():
    backend = SearchBackend()
    backend.next_results = [
        {
            "url": f"https://previews.example/article/{i}",
            "title": "Worlds preview and rankings",
            "content": "Predictions only; no completed scores.",
        }
        for i in range(5)
    ]
    session = session_for(backend)
    first = await search(session, "League of Legends Worlds latest results standings October 26")
    followup = await search(session, "Worlds 2026 October 25 26 Swiss results wins losses")

    assert first["new_source_count"] == 5
    assert followup["status"] == "success"
    assert followup["searches_used"] == 2
    assert followup["sources"][0]["source_id"] == "S6"
    assert "Completed fixture" in followup["sources"][0]["snippet"]
    assert str(session.source_registry.resolve("S1").url).startswith("https://previews.")
    assert str(session.source_registry.resolve("S6").url).startswith("https://scores.")
    grounded = GroundingParser(session).resolve("Completed fixture [[source:S6]]")
    assert grounded.source_ids == ("S6",)
    assert "https://scores.example/2/0" in grounded.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requested", "expected", "mode"),
    [(1, 1, "quick"), (500, 5, "quick"), (1, 1, "agentic"), (500, 8, "agentic")],
)
async def test_public_result_count_controls_provider_and_returned_evidence(
    requested, expected, mode
):
    backend = SearchBackend()
    result = await search(
        session_for(backend, mode=mode), "current tournament results", max_results=requested
    )

    assert backend.calls[0]["max_results"] == expected
    assert len(result["sources"]) == expected


@pytest.mark.asyncio
async def test_remaining_capacity_counts_valid_unique_sources_instead_of_payload_positions():
    backend = SearchBackend()
    session = session_for(backend)
    for i in range(12):
        await search(session, f"subject {i}")
    await search(session, "remaining evidence", max_results=3)
    assert len(session.source_registry.records) == 63
    original = session.source_registry.resolve("S1")
    backend.next_results = [
        {"url": "https://127.0.0.1/private", "content": "invalid"},
        {"url": str(original.url), "content": "already known"},
        {"url": "https://scores.example/final", "content": "new completed result"},
        {"url": "https://scores.example/overflow", "content": "over capacity"},
    ]
    last = await search(session, "final missing scores")

    assert backend.calls[-1]["max_results"] == 1
    assert [item["source_id"] for item in last["sources"]] == ["S64"]
    assert last["sources"][0]["url"] == "https://scores.example/final"
    assert session.source_registry.resolve("S1") == original
    assert len(session.source_registry.records) == 64
    exhausted = await search(session, "unfilled followup question")
    assert exhausted["failures"][0]["code"] == "source_quota_exhausted"
    assert len(backend.calls) == 14


@pytest.mark.parametrize("topic", ["general", "news"])
def test_current_search_category_is_independent_of_date_bounds(topic):
    normalized = normalize_web_search(
        WebSearchRequest(
            query="Worlds 2026 results", objective="Find scores", freshness="recent", topic=topic
        ),
        now=NOW,
        configured_max_results=5,
    )
    args = tavily_search_args(normalized)
    assert args["topic"] == topic
    assert args["end_date"] == "2026-10-26"


def test_invalid_search_topic_is_rejected_at_public_boundary():
    with pytest.raises(ValidationError):
        create_web_search_tool().args_schema(
            query="Worlds results", objective="Find scores", topic="unknown"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("restore", [False, True])
async def test_changing_topic_allows_new_retrieval_while_exact_repeats_remain_refused(restore):
    backend = SearchBackend()
    session = session_for(backend)
    await search(session, "Worlds 2026 results", topic="news", freshness="recent")
    if restore:
        session = session_for(backend, research_budget_from_state(session.budget.to_state()))
    repeat = await search(session, "Worlds 2026 results", topic="news", freshness="recent")
    general = await search(session, "Worlds 2026 results", topic="general", freshness="recent")

    assert repeat["failures"][0]["code"] == "duplicate_query"
    assert general["status"] == "success"
    assert general["new_source_count"] == 5
    assert [call["topic"] for call in backend.calls] == ["news", "general"]


@pytest.mark.asyncio
@pytest.mark.parametrize("freshness", ["timeless", "recent"])
async def test_deduplication_survives_json_persistence_with_domain_filters(freshness):
    backend = SearchBackend()
    session = session_for(backend)
    args = {"freshness": freshness, "include_domains": ["scores.example"]}
    await search(session, "Worlds 2026 results", **args)
    wire = json.loads(json.dumps(session.budget.to_state()))
    restored = session_for(backend, research_budget_from_state(wire))
    repeat = await search(restored, "Worlds 2026 results", **args)
    assert repeat["failures"][0]["code"] == "duplicate_query"
    assert len(backend.calls) == 1


@pytest.mark.asyncio
async def test_pre_upgrade_search_scope_keeps_its_historical_news_category():
    wire = {
        "schema_version": 1,
        "searched": [
            {
                "tokens": ["worlds", "2026", "results"],
                "scope": ["recent", None, "2026-10-26", [], "none", ""],
            }
        ],
        "image_subjects": [],
        "epochs_recorded": 1,
    }
    backend = SearchBackend()
    session = session_for(backend, research_budget_from_state(wire))
    repeated_news = await search(session, "Worlds 2026 results", freshness="recent", topic="news")
    general = await search(session, "Worlds 2026 results", freshness="recent", topic="general")
    assert repeated_news["failures"][0]["code"] == "duplicate_query"
    assert general["new_source_count"] == 5
    assert len(backend.calls) == 1


@pytest.mark.asyncio
async def test_carried_sources_do_not_reset_turn_capacity_on_continue():
    backend = SearchBackend()
    session = session_for(backend)
    session.import_sources(
        [
            SourceRecord(
                source_id=f"S{i}",
                url=f"https://previous.example/{i}",
                status="search_result",
                provider="tavily",
                query_index=i,
            )
            for i in range(1, 65)
        ]
    )
    result = await search(session, "additional missing evidence")
    assert result["failures"][0]["code"] == "source_quota_exhausted"
    assert len(session.source_registry.records) == 64
    assert backend.calls == []


@pytest.mark.asyncio
async def test_none_mode_never_calls_a_search_provider():
    backend = SearchBackend()
    session = WebResearchService(
        resolver=ProviderResolver(text=(TavilyTextSearchProvider(backend),)), now=lambda: NOW
    ).new_session(SCOPE, ResearchBudget(), mode="none")
    result = await session.search(ResearchRequest(query="current results", objective="Find scores"))
    assert result.sources == ()
    assert backend.calls == []
