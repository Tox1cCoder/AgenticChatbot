from __future__ import annotations

from app.ai.research_budget import (
    ResearchBudget,
    get_research_budget,
    near_duplicate,
    normalize_query_tokens,
    reset_research_budget,
)

TRACE_Q1 = "T1 League of Legends Esports team news roster 2026"
TRACE_Q2 = "T1 League of Legends team overview roster news 2026"
TRACE_Q3 = "T1 League of Legends current roster 2026 achievements lck summer"


def test_short_identifiers_survive_normalization():
    tokens = normalize_query_tokens("T1 F1 3M vs G2")

    assert {"t1", "f1", "3m", "g2"} <= tokens


def test_non_ascii_queries_normalize_without_losing_tokens():
    tokens = normalize_query_tokens("thông tin về T1")

    assert "t1" in tokens
    assert len(tokens) == 4


def test_trace_second_query_is_a_near_duplicate_of_the_first():
    assert near_duplicate(
        normalize_query_tokens(TRACE_Q1), normalize_query_tokens(TRACE_Q2), threshold=0.75
    )


def test_trace_third_query_is_genuinely_distinct():
    assert not near_duplicate(
        normalize_query_tokens(TRACE_Q1), normalize_query_tokens(TRACE_Q3), threshold=0.75
    )


def test_trace_produces_two_network_searches_and_one_reuse():
    budget = ResearchBudget(max_search_calls=2, near_duplicate_threshold=0.75)

    assert budget.find_reuse(TRACE_Q1) is None
    budget.record_search(TRACE_Q1, "first result")

    assert budget.find_reuse(TRACE_Q2) == "first result"

    assert budget.find_reuse(TRACE_Q3) is None
    assert budget.may_search(TRACE_Q3) is True
    budget.record_search(TRACE_Q3, "third result")

    assert budget.search_calls == 2


def test_a_fourth_distinct_query_is_refused_and_returns_accumulated_results():
    budget = ResearchBudget(max_search_calls=2, near_duplicate_threshold=0.75)
    budget.record_search("alpha topic one", "A")
    budget.record_search("beta topic two", "B")

    assert budget.may_search("gamma topic three") is False
    assert budget.accumulated() == ["A", "B"]


def test_only_one_image_search_per_turn():
    budget = ResearchBudget(max_search_calls=2, near_duplicate_threshold=0.75)

    assert budget.may_image_search() is True
    budget.record_image_search([{"id": "image:verified:1"}])

    assert budget.may_image_search() is False
    assert budget.image_result() == [{"id": "image:verified:1"}]


def test_budget_is_per_conversation_and_resettable():
    first = get_research_budget("conv-a")
    first.record_search("alpha", "A")

    assert get_research_budget("conv-a") is first
    assert get_research_budget("conv-b") is not first

    reset_research_budget("conv-a")
    assert get_research_budget("conv-a").search_calls == 0


def test_missing_conversation_id_gets_an_isolated_budget():
    reset_research_budget(None)
    budget = get_research_budget(None)
    budget.record_search("alpha", "A")

    reset_research_budget(None)
    assert get_research_budget(None).search_calls == 0
