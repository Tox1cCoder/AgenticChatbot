from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.ai.research_budget import ResearchBudget
from app.ai.tool_context import get_tool_context, tool_execution_context
from app.ai.web_research.contracts import ProviderSource, ResearchRequest, ResearchScope
from app.ai.web_research.providers import ProviderResolver
from app.ai.web_research.service import WebResearchService
from app.ai.web_tools import create_web_open_tool, create_web_search_tool


class _Session:
    mode = "quick"

    def __init__(self, bundle) -> None:
        self.bundle = bundle
        self.budget = SimpleNamespace(search_calls=1)
        self.requests: list[ResearchRequest] = []
        self.opens: list[tuple[list[str], str]] = []

    async def search(self, request: ResearchRequest):
        self.requests.append(request)
        return self.bundle

    async def open(self, urls, question):
        self.opens.append((list(urls), question))
        return self.bundle


class _SequenceText:
    """Returns a different result set per search, with a deliberate overlap."""

    name = "tavily"
    health_key = "tavily:test"

    def __init__(self, *cohorts: tuple[str, ...]) -> None:
        self.cohorts = list(cohorts)
        self.calls = 0

    async def search(self, _request, *, query_index: int):
        cohort = self.cohorts[min(self.calls, len(self.cohorts) - 1)]
        self.calls += 1
        return tuple(
            ProviderSource(
                provider="tavily",
                url=url,
                title=f"Page {rank}",
                snippet="evidence " * 200,
                rank=rank,
                query_index=query_index,
            )
            for rank, url in enumerate(cohort, start=1)
        )


class _Opener:
    name = "tavily"
    health_key = "tavily:test"

    async def open(self, urls, question, *, query_index: int):
        return tuple(
            ProviderSource(
                provider="tavily",
                url=url,
                title="Opened page",
                snippet="deep read " * 400,
                rank=rank,
                query_index=query_index,
            )
            for rank, url in enumerate(urls, start=1)
        )


def _real_session():
    service = WebResearchService(
        resolver=ProviderResolver(
            text=(
                _SequenceText(
                    ("https://a.test/one", "https://b.test/two"),
                    ("https://b.test/two", "https://c.test/three"),
                ),
            ),
            openers=(_Opener(),),
        ),
        now=lambda: datetime(2026, 9, 16, tzinfo=timezone.utc),
    )
    scope = ResearchScope(
        conversation_id=str(uuid4()), user_id=str(uuid4()), logical_turn_id=str(uuid4())
    )
    return service.new_session(scope, ResearchBudget(), mode="quick")


def _search_args(query: str) -> dict:
    return {"query": query, "objective": "find the evidence", "max_results": 5}


@pytest.mark.asyncio
async def test_each_search_returns_only_the_sources_it_newly_admitted() -> None:
    """A tool result used to repeat the whole accumulated registry."""

    session = _real_session()

    with tool_execution_context(web_research_session=session):
        tool = create_web_search_tool()
        first = json.loads(await tool.ainvoke(_search_args("first distinct subject")))
        second = json.loads(await tool.ainvoke(_search_args("wholly unrelated matter")))

    assert {item["source_id"] for item in first["sources"]} == {"S1", "S2"}
    assert {item["source_id"] for item in second["sources"]} == {"S3"}
    assert second["new_source_count"] == 1
    assert second["total_source_count"] == 3


@pytest.mark.asyncio
async def test_open_reports_the_page_it_opened_even_though_it_was_already_known() -> None:
    """The delta is explicit, not inferred from ``query_index``."""

    session = _real_session()

    with tool_execution_context(web_research_session=session):
        await create_web_search_tool().ainvoke(_search_args("first distinct subject"))
        opened = json.loads(
            await create_web_open_tool().ainvoke(
                {"urls": ["S1"], "question": "What does the page say?"}
            )
        )

    assert [item["source_id"] for item in opened["sources"]] == ["S1"]
    assert opened["sources"][0]["status"] == "opened"


@pytest.mark.asyncio
async def test_snippets_are_bounded_by_status_so_a_deep_read_survives() -> None:
    session = _real_session()

    with tool_execution_context(web_research_session=session):
        searched = json.loads(
            await create_web_search_tool().ainvoke(_search_args("first distinct subject"))
        )
        opened = json.loads(
            await create_web_open_tool().ainvoke(
                {"urls": ["S1"], "question": "What does the page say?"}
            )
        )

    assert len(searched["sources"][0]["snippet"]) == 1200
    assert len(opened["sources"][0]["snippet"]) == 3000


@pytest.mark.asyncio
async def test_web_search_uses_the_exact_session_from_tool_context() -> None:
    bundle = SimpleNamespace(
        status="success",
        mode="quick",
        operation_index=1,
        sources=(
            SimpleNamespace(
                source_id="S1",
                title="Release notes",
                url="https://example.test/release",
                snippet="Released today.",
                published_at=None,
                status="search_result",
            ),
        ),
        images=(SimpleNamespace(candidate_id="I1", delivery_url="/web-images/private"),),
        failures=(),
        reused=False,
        omitted_source_count=0,
        omitted_image_count=0,
        operation_source_ids=("S1",),
    )
    session = _Session(bundle)

    with tool_execution_context(
        conversation_id="conversation",
        user_id="user",
        agent_key="search",
        web_research_session=session,
    ):
        assert get_tool_context().web_research_session is session
        raw = await create_web_search_tool().ainvoke(
            {
                "query": "Aurora release",
                "objective": "Find the release date",
                "freshness": "recent",
                "visual_intent": "figure",
                "image_query": "Aurora interface",
            }
        )

    assert session.requests[0].visual_intent == "figure"
    assert session.requests[0].image_query == "Aurora interface"
    public = json.loads(raw)
    assert public["sources"][0]["source_id"] == "S1"
    assert "images" not in public
    assert "I1" not in raw
    assert "/web-images/" not in raw


def test_nested_tool_contexts_restore_their_own_research_session() -> None:
    first = object()
    second = object()

    with tool_execution_context(web_research_session=first):
        assert get_tool_context().web_research_session is first
        with tool_execution_context(web_research_session=second):
            assert get_tool_context().web_research_session is second
        assert get_tool_context().web_research_session is first


@pytest.mark.asyncio
async def test_web_open_uses_the_same_turn_session() -> None:
    bundle = SimpleNamespace(
        status="success",
        mode="quick",
        operation_index=2,
        sources=(),
        images=(),
        failures=(),
        reused=False,
        omitted_source_count=0,
        omitted_image_count=0,
        operation_source_ids=(),
    )
    session = _Session(bundle)

    with tool_execution_context(web_research_session=session):
        await create_web_open_tool().ainvoke(
            {"urls": ["https://example.test/release"], "question": "When was it released?"}
        )

    assert session.opens == [(["https://example.test/release"], "When was it released?")]
