"""How many times a turn searches is the model's call, not a config's.

The per-turn search *count* cap is gone. It never once refused a turn that
deserved refusing -- three separate reports traced to a stale default, a
mislabelled refusal, and a leaked reservation -- while every occurrence cost a
real answer. What remains is the part that was always doing the useful work:
deduplication, which refuses a query this turn already ran, already has a
result for, or already failed at the provider.

The backstop is unchanged and is not search-specific:
``generation_soft_tool_calls_per_epoch`` bounds every tool call in an epoch, so
an unbounded search loop still terminates.

The model paces itself from ``searches_used``, which every successful search
returns.
"""

from __future__ import annotations

import json

import pytest

from app.ai.research_budget import ResearchBudget
from app.core.config import Settings


def _budget() -> ResearchBudget:
    return ResearchBudget(near_duplicate_threshold=0.75, max_image_searches=3)


def test_the_search_count_setting_is_gone():
    assert "research_max_search_calls_per_turn" not in Settings.model_fields


def test_a_budget_has_no_search_count_ceiling():
    assert not hasattr(ResearchBudget(), "max_search_calls")


SUBJECTS = (
    "T1 LCK 2026 roster",
    "vietnam coffee export tonnage",
    "james webb telescope latest images",
    "bank of japan policy rate decision",
    "rust async runtime comparison",
    "antarctic sea ice extent trend",
    "premier league fixture congestion",
    "solid state battery manufacturing yield",
    "tokyo apartment rent index",
    "postgres logical replication limits",
    "olive oil harvest spain drought",
    "quantum error correction threshold",
)


def test_many_distinct_searches_are_all_allowed():
    """The case that used to fail: a multi-facet question.

    Twelve genuinely unrelated subjects, well past any cap that ever existed.
    They have to be unrelated: near-duplicate detection is still on, and
    mechanically-numbered variants of one sentence trip it, correctly.
    """
    budget = _budget()

    for subject in SUBJECTS:
        assert budget.reserve_search(subject) is None, f"{subject!r} was refused"
        budget.record_search(subject, "RESULT")

    assert budget.search_calls == len(SUBJECTS)


def test_a_repeat_is_still_refused():
    budget = _budget()
    budget.reserve_search("population of vietnam")
    budget.record_search("population of vietnam", "RESULT")

    assert budget.reserve_search("population of vietnam") == "duplicate_query"


def test_a_near_duplicate_is_still_refused():
    budget = _budget()
    budget.reserve_search("population of vietnam 2024")
    budget.record_search("population of vietnam 2024", "RESULT")

    assert budget.reserve_search("vietnam population 2024") == "duplicate_query"


def test_a_query_that_failed_at_the_provider_is_still_refused():
    budget = _budget()
    budget.reserve_search("broken query")
    budget.record_failed_search("broken query")

    assert budget.reserve_search("broken query") == "duplicate_query"


def test_an_in_flight_query_is_still_refused():
    """Two identical searches running at once are pure waste."""
    budget = _budget()
    assert budget.reserve_search("same subject twice") is None

    assert budget.reserve_search("same subject twice") == "duplicate_query"


def test_a_prior_epochs_query_is_still_refused():
    from app.ai.research_budget import research_budget_from_state

    budget = _budget()
    budget.reserve_search("population of vietnam")
    budget.record_search("population of vietnam", "RESULT")

    restored = research_budget_from_state(budget.to_state())

    assert restored.reserve_search("population of vietnam") == "duplicate_query"
    assert restored.reserve_search("an entirely different subject") is None


@pytest.mark.asyncio
async def test_the_refusal_payload_no_longer_claims_a_spent_budget():
    from app.ai.tool_context import tool_execution_context
    from app.ai.web_research.contracts import ResearchScope
    from app.ai.web_research.providers import ProviderResolver, TavilyTextSearchProvider
    from app.ai.web_research.service import WebResearchService
    from app.ai.web_tools import create_web_search_tool

    class _Fake:
        async def ainvoke(self, args: dict) -> str:
            return json.dumps({"results": []})

    budget = _budget()
    session = WebResearchService().new_session(
        ResearchScope(
            conversation_id="99999999-9999-9999-9999-999999999999",
            user_id="11111111-1111-1111-1111-111111111111",
            logical_turn_id="turn",
        ),
        budget,
        mode="quick",
        resolver=ProviderResolver(text=(TavilyTextSearchProvider(_Fake()),)),
    )
    tool = create_web_search_tool()
    with tool_execution_context(web_research_session=session):
        await tool.ainvoke({"query": "alpha", "objective": "find alpha"})
        payload = json.loads(
            await tool.ainvoke({"query": "alpha", "objective": "find alpha again"})
        )

    assert payload["failures"][0]["code"] == "duplicate_query"
    assert "search_limit" not in payload
    assert payload["searches_used"] == 1


@pytest.mark.asyncio
async def test_a_successful_search_reports_the_running_count_for_pacing():
    """The only number the model gets, and the one the prompt tells it to watch."""
    import asyncio

    from app.ai.tool_context import tool_execution_context
    from app.ai.web_research.contracts import ResearchScope
    from app.ai.web_research.providers import ProviderResolver, TavilyTextSearchProvider
    from app.ai.web_research.service import WebResearchService
    from app.ai.web_tools import create_web_search_tool

    conversation = "99999999-9999-9999-9999-999999999999"

    class _Fake:
        name = "tavily_search"

        async def ainvoke(self, args: dict) -> str:
            return json.dumps({"results": [], "total_results": 0, "answer": ""})

    session = WebResearchService().new_session(
        ResearchScope(
            conversation_id=conversation,
            user_id="11111111-1111-1111-1111-111111111111",
            logical_turn_id="turn",
        ),
        _budget(),
        mode="quick",
        resolver=ProviderResolver(text=(TavilyTextSearchProvider(_Fake()),)),
    )
    tool = create_web_search_tool()
    with tool_execution_context(web_research_session=session):
        first = json.loads(
            await tool.ainvoke({"query": "first subject", "objective": "find a fact"})
        )
        second = json.loads(
            await tool.ainvoke(
                {"query": "a wholly separate second subject", "objective": "find another"}
            )
        )

    assert first["searches_used"] == 1
    assert second["searches_used"] == 2
    assert "search_limit" not in second
    assert asyncio.iscoroutinefunction(_Fake.ainvoke)


def test_both_web_capable_agents_carry_the_same_search_policy():
    """`chat` and `search` both bind `web_search` (see BaseAgent).

    Chat had no web guidance at all, so half the agents that can search had no
    policy telling them when to stop. One shared block means they cannot drift.
    """
    from app.ai.prompts import CHAT_SYSTEM_PROMPT, SEARCH_SYSTEM_PROMPT, WEB_RESEARCH_SNIPPET

    assert WEB_RESEARCH_SNIPPET in CHAT_SYSTEM_PROMPT
    assert WEB_RESEARCH_SNIPPET in SEARCH_SYSTEM_PROMPT


def test_the_policy_tells_the_model_it_is_the_one_deciding():
    """The cap is gone; the prompt has to carry the restraint it pretended to."""
    from app.ai.prompts import WEB_RESEARCH_SNIPPET

    text = WEB_RESEARCH_SNIPPET.lower()
    assert "searches_used" in WEB_RESEARCH_SNIPPET, "the model needs its running count named"
    assert "nothing limits how many times you may search" in text
    assert "web_open" in WEB_RESEARCH_SNIPPET, "opening a found source must beat re-searching"


def test_the_search_prompt_no_longer_pushes_only_toward_more_searching():
    """Three unconditional 'search more' bullets outweighed one stop rule."""
    from app.ai.prompts import SEARCH_SYSTEM_PROMPT

    routine_corroboration = "Verify important facts across multiple sources when possible"
    assert routine_corroboration not in SEARCH_SYSTEM_PROMPT
    assert "try a different angle" not in SEARCH_SYSTEM_PROMPT
    assert "Don't provide superficial answers" not in SEARCH_SYSTEM_PROMPT
    assert "Stop when independent sources support the answer" in SEARCH_SYSTEM_PROMPT
