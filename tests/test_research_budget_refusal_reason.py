"""A refusal must name the cause the reservation actually refused for.

``reserve_search`` refuses for two different reasons -- the per-epoch call cap,
and a query this turn has already run or already failed -- and returned a bare
boolean, so the caller re-derived the reason from ``search_calls``. That
derivation cannot see the in-flight reservations the cap check counts, and the
model emits its searches in parallel, so a refusal caused by the cap was
reported as ``duplicate_query`` saying "The budget is not spent" -- which sends
the model to rephrase and burn further calls that are refused the same way.

The mirror of this mislabel is already covered by
``test_research_budget_failure_release``; this file owns the in-flight half and
the wording of the at-cap hint.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest

from app.ai import web_tools
from app.ai.research_budget import ResearchBudget, reset_research_budget
from app.ai.tool_context import clear_tool_context, tool_execution_context
from app.ai.web_tools import _budget_spent_payload, create_web_search_tool

CONVERSATION_ID = "66666666-6666-6666-6666-666666666666"
NOW = datetime(2026, 9, 4, 12, tzinfo=timezone.utc)


def _budget(cap: int = 2) -> ResearchBudget:
    return ResearchBudget(
        max_search_calls=cap,
        near_duplicate_threshold=0.75,
        max_image_searches=3,
    )


@pytest.fixture(autouse=True)
def _clean():
    clear_tool_context()
    reset_research_budget(conversation_id=CONVERSATION_ID)
    yield
    clear_tool_context()
    reset_research_budget(conversation_id=CONVERSATION_ID)


def test_a_refusal_caused_by_in_flight_reservations_names_the_cap():
    """Two searches in flight fill the cap; the third is refused by it."""
    budget = _budget(cap=2)
    assert budget.reserve_search("alpha topic") is None
    assert budget.reserve_search("beta topic") is None

    refusal = budget.reserve_search("gamma topic")

    assert refusal == "budget_exhausted"
    payload = json.loads(_budget_spent_payload(budget, refusal))
    assert payload["error_type"] == "budget_exhausted"
    assert payload["search_limit"] == 2


def test_a_genuine_duplicate_is_still_reported_as_a_duplicate():
    budget = _budget(cap=4)
    budget.reserve_search("population of vietnam")
    budget.record_search("population of vietnam", "RESULT")

    refusal = budget.reserve_search("population of vietnam")

    payload = json.loads(_budget_spent_payload(budget, refusal))
    assert payload["error_type"] == "duplicate_query"
    assert "budget is not spent" in payload["hint"]


def test_a_query_that_already_failed_is_reported_as_a_duplicate():
    budget = _budget(cap=4)
    budget.reserve_search("broken query")
    budget.record_failed_search("broken query")

    assert budget.reserve_search("broken query") == "duplicate_query"


def test_a_repeat_of_a_prior_epoch_is_reported_as_a_duplicate():
    budget = _budget(cap=4)
    budget._prior_searches.append((frozenset({"population", "of", "vietnam"}), ()))

    assert budget.reserve_search("population of vietnam") == "duplicate_query"


def test_the_at_cap_hint_does_not_claim_a_new_query_is_pointless():
    """A different query returns different sources; saying otherwise is false.

    That sentence is what tells the model the rest of its research plan is
    worthless, so it stops gathering rather than continuing next epoch.
    """
    budget = _budget(cap=1)
    budget.reserve_search("only topic")
    budget.record_search("only topic", "RESULT")

    refusal = budget.reserve_search("another topic")
    payload = json.loads(_budget_spent_payload(budget, refusal))

    assert payload["error_type"] == "budget_exhausted"
    assert "would return the same results" not in payload["hint"]


def test_a_granted_reservation_reserves_exactly_one_slot():
    budget = _budget(cap=2)

    assert budget.reserve_search("alpha topic") is None
    assert budget.reserve_search("beta topic") is None
    assert budget.reserve_search("gamma topic") == "budget_exhausted"


def test_parallel_web_search_calls_report_the_cap_not_a_duplicate(monkeypatch):
    """End to end, through the tool the model actually calls."""

    class _SlowSearch:
        name = "tavily_search"

        async def ainvoke(self, args: dict) -> str:
            await asyncio.sleep(0.05)
            return json.dumps({"results": [], "total_results": 0, "answer": ""})

    monkeypatch.setattr(web_tools.settings, "research_max_search_calls_per_turn", 2)
    reset_research_budget(conversation_id=CONVERSATION_ID)
    tool = create_web_search_tool(tavily_tool=_SlowSearch(), clock=lambda: NOW)

    async def _run() -> list[dict]:
        with tool_execution_context(
            conversation_id=CONVERSATION_ID, user_id="u1", agent_key="search"
        ):
            raw = await asyncio.gather(
                *(
                    tool.ainvoke(
                        {
                            "query": query,
                            "objective": "verify one distinct fact about the subject",
                        }
                    )
                    for query in (
                        "T1 2026 LCK standings schedule",
                        "T1 official 2026 roster Peyz Doran Oner",
                        "T1 2026 LCK Cup First Stand results",
                    )
                )
            )
        return [json.loads(item) for item in raw]

    payloads = asyncio.run(_run())
    refusals = [item for item in payloads if item.get("status") == "error"]

    assert len(refusals) == 1
    assert refusals[0]["error_type"] == "budget_exhausted"


def test_the_default_cap_covers_a_multi_facet_research_turn():
    """The execution ladder was quadrupled for chained research on 2026-09-10.

    The search cap was left at the value tuned on 2026-08-05, when a turn made
    one or two searches, so it became the binding constraint: a question with
    three facets was refused on its third distinct query.
    """
    from app.core.config import Settings

    assert Settings.model_fields["research_max_search_calls_per_turn"].default >= 6
