"""A failed provider call must not hold the reservation it claimed.

``reserve_search`` claims a slot before the network call and ``record_search``
releases it after. A failure used to reach neither, so the reservation was held
for the rest of the turn *by design* — the stated intent being that a provider
error must not buy the model a second attempt at the same broken query.

Holding the reservation achieves that, but it also leaves an entry in
``_in_flight`` that grows without bound and refuses near-duplicates of a query
that is no longer running. The retry guard is kept here; the collateral is not,
and ``_failed_searches`` is what actually enforces the guard.
"""

from __future__ import annotations

from app.ai.research_budget import ResearchBudget


def _budget() -> ResearchBudget:
    return ResearchBudget(near_duplicate_threshold=0.75, max_image_searches=3)


def test_a_failed_search_frees_the_slot_for_a_different_query():
    budget = _budget()
    budget.reserve_search("first topic")
    budget.record_search("first topic", "RESULT")

    budget.reserve_search("second topic")
    budget.record_failed_search("second topic")

    assert budget.reserve_search("a third, unrelated topic") is None


def test_a_failed_query_is_still_refused_on_retry():
    """The guard the held reservation existed for must survive the fix."""
    budget = _budget()
    budget.reserve_search("broken query")
    budget.record_failed_search("broken query")

    assert budget.reserve_search("broken query") is not None
    assert budget.reserve_search("broken  QUERY") is not None


def test_a_failure_does_not_count_as_a_completed_search():
    budget = _budget()
    budget.reserve_search("only topic")
    budget.record_failed_search("only topic")

    assert budget.search_calls == 0


def test_an_unrelated_query_is_never_refused_by_how_many_ran_before_it():
    """There is no count cap; only a repeat is refused."""
    budget = _budget()
    for subject in ("one distinct subject", "another wholly separate matter"):
        budget.reserve_search(subject)
        budget.record_search(subject, "R")

    assert budget.reserve_search("a third, unrelated topic entirely") is None
