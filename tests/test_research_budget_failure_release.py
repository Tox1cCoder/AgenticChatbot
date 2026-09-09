"""A failed provider call must not spend the turn's search quota.

``reserve_search`` claims a slot before the network call and ``record_search``
releases it after. A failure used to reach neither, so the reservation was held
for the rest of the turn *by design* — the stated intent being that a provider
error must not buy the model a second attempt at the same broken query.

Holding the reservation achieves that, but it also consumes a slot. With the
default cap of 2, one success plus one failure reaches the cap, and every later
*unrelated* query is refused reporting ``searches_used: 1`` — a budget the turn
never actually spent. The retry guard is kept here; the collateral is not.
"""

from __future__ import annotations

import json

import pytest

from app.ai.research_budget import ResearchBudget
from app.ai.web_tools import _budget_spent_payload


def _budget(cap: int = 2) -> ResearchBudget:
    return ResearchBudget(
        max_search_calls=cap,
        near_duplicate_threshold=0.75,
        max_image_searches=3,
    )


def test_a_failed_search_frees_the_slot_for_a_different_query():
    budget = _budget()
    budget.reserve_search("first topic")
    budget.record_search("first topic", "RESULT")

    budget.reserve_search("second topic")
    budget.record_failed_search("second topic")

    assert budget.reserve_search("a third, unrelated topic") is True


def test_a_failed_query_is_still_refused_on_retry():
    """The guard the held reservation existed for must survive the fix."""
    budget = _budget()
    budget.reserve_search("broken query")
    budget.record_failed_search("broken query")

    assert budget.reserve_search("broken query") is False
    assert budget.reserve_search("broken  QUERY") is False


def test_a_failure_does_not_count_as_a_completed_search():
    budget = _budget()
    budget.reserve_search("only topic")
    budget.record_failed_search("only topic")

    assert budget.search_calls == 0


def test_the_cap_still_applies_to_successful_searches():
    budget = _budget(cap=2)
    budget.reserve_search("one")
    budget.record_search("one", "R1")
    budget.reserve_search("two")
    budget.record_search("two", "R2")

    assert budget.reserve_search("three") is False


@pytest.mark.parametrize(
    ("used", "cap", "expected"),
    [(2, 2, "budget_exhausted"), (1, 2, "duplicate_query"), (0, 2, "duplicate_query")],
)
def test_the_refusal_names_its_real_cause(used, cap, expected):
    """A repeated query reported as an exhausted budget is a false statement.

    It also misdirects the model: told the quota is gone it summarises, when
    the correct move is to rephrase and search again.
    """
    budget = _budget(cap=cap)
    for index in range(used):
        query = f"topic {index}"
        budget.reserve_search(query)
        budget.record_search(query, "R")

    payload = json.loads(_budget_spent_payload(budget))

    assert payload["error_type"] == expected
    assert payload["searches_used"] == used
    assert payload["search_limit"] == cap
