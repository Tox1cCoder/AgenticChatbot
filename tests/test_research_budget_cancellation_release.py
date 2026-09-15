"""A cancelled search must not hold its slot for the rest of the turn.

``reserve_search`` claims a slot before the network call; ``record_search``
and ``record_failed_search`` release it after. Both are reached through
``except`` clauses, and ``asyncio.CancelledError`` is a ``BaseException`` --
so neither runs when the tool call is cancelled, and the reservation is held
forever.

Cancellation is routine, not exotic: ``tool_execution`` cancels the task on its
soft timeout, and a Stop or a client disconnect cancels it too. Each one
permanently consumed a search slot, so a turn reported ``budget_exhausted``
with ``searches_used`` *below* ``search_limit`` -- the arithmetic Thai saw as
``5`` of ``6``.

This is the same defect as ``test_research_budget_failure_release`` one path
over: that fix released on provider failure and stopped there.
"""

from __future__ import annotations

import asyncio

import pytest

from app.ai.research_budget import ResearchBudget
from app.ai.web_research.contracts import ResearchRequest, ResearchScope
from app.ai.web_research.providers import ProviderResolver, TavilyTextSearchProvider
from app.ai.web_research.service import WebResearchService

CONVERSATION_ID = "77777777-7777-7777-7777-777777777777"
class _Hanging:
    name = "tavily_search"

    async def ainvoke(self, args: dict) -> str:
        await asyncio.sleep(30)
        return "{}"


async def _cancel_a_search_mid_flight(budget: ResearchBudget):
    session = WebResearchService().new_session(
        ResearchScope(
            conversation_id=CONVERSATION_ID,
            user_id="11111111-1111-1111-1111-111111111111",
            logical_turn_id="turn",
        ),
        budget,
        mode="quick",
        resolver=ProviderResolver(text=(TavilyTextSearchProvider(_Hanging()),)),
    )
    task = asyncio.create_task(
        session.search(ResearchRequest(query="a query that hangs", objective="never returns"))
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_release_search_frees_a_slot_without_recording_anything():
    budget = ResearchBudget(near_duplicate_threshold=0.75)
    assert budget.reserve_search("abandoned topic") is None

    budget.release_search("abandoned topic")

    # The slot is back, the query is not remembered, and nothing was recorded.
    assert budget.search_calls == 0
    assert budget.reserve_search("abandoned topic") is None


def test_releasing_a_reservation_that_is_already_gone_is_harmless():
    budget = ResearchBudget(near_duplicate_threshold=0.75)
    budget.reserve_search("topic")
    budget.record_search("topic", "RESULT")

    budget.release_search("topic")

    assert budget.search_calls == 1


@pytest.mark.asyncio
async def test_a_cancelled_search_releases_its_slot():
    budget = ResearchBudget()
    await _cancel_a_search_mid_flight(budget)

    assert budget.search_calls == 0
    assert len(budget._in_flight) == 0, "the cancelled search still holds its slot"


@pytest.mark.asyncio
async def test_a_cancelled_search_does_not_refuse_a_later_unrelated_query():
    """The symptom: budget_exhausted with searches_used below search_limit."""
    budget = ResearchBudget()
    await _cancel_a_search_mid_flight(budget)

    assert budget.reserve_search("an entirely unrelated subject") is None
    budget.record_search("an entirely unrelated subject", "R")
    assert budget.reserve_search("a second unrelated subject") is None


@pytest.mark.asyncio
async def test_a_cancelled_query_is_not_remembered_as_failed():
    """A cancelled call never got the provider's verdict, unlike a failure.

    ``record_failed_search`` deliberately remembers a query so the model cannot
    retry a broken one. Cancellation is not the provider's answer, so the same
    query has to stay available -- asserted on the dedup memory itself, because
    a bare ``reserve_search`` here would use the default scope rather than the
    one the tool derives from its arguments, and would pass without meaning it.
    """
    budget = ResearchBudget()
    await _cancel_a_search_mid_flight(budget)

    assert budget._failed_searches == []
    assert budget._searches == []
