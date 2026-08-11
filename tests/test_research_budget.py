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
    tokens_q1 = normalize_query_tokens(TRACE_Q1)
    tokens_q2 = normalize_query_tokens(TRACE_Q2)

    # Pin the actual overlap so this fails if tokenization drifts, not only
    # if the near-duplicate verdict happens to flip.
    assert len(tokens_q1 & tokens_q2) == 8
    assert min(len(tokens_q1), len(tokens_q2)) == 9
    assert near_duplicate(tokens_q1, tokens_q2, threshold=0.75)


def test_trace_third_query_is_genuinely_distinct():
    tokens_q1 = normalize_query_tokens(TRACE_Q1)
    tokens_q3 = normalize_query_tokens(TRACE_Q3)

    assert len(tokens_q1 & tokens_q3) == 6
    assert min(len(tokens_q1), len(tokens_q3)) == 9
    assert not near_duplicate(tokens_q1, tokens_q3, threshold=0.75)


def test_trace_produces_two_network_searches_and_one_reuse():
    budget = ResearchBudget(max_search_calls=2, near_duplicate_threshold=0.75)

    assert budget.find_reuse(TRACE_Q1) is None
    budget.record_search(TRACE_Q1, "first result")

    assert budget.find_reuse(TRACE_Q2) == "first result"

    assert budget.find_reuse(TRACE_Q3) is None
    assert budget.reserve_search(TRACE_Q3) is True
    budget.record_search(TRACE_Q3, "third result")

    assert budget.search_calls == 2


def test_tavily_controls_scope_near_duplicate_reuse():
    budget = ResearchBudget(max_search_calls=2, near_duplicate_threshold=0.75)
    general_scope = (None, None, None, None)
    news_scope = ("news", "week", None, None)

    budget.record_search(TRACE_Q1, "general result", scope=general_scope)

    assert budget.find_reuse(TRACE_Q2, scope=general_scope) == "general result"
    assert budget.find_reuse(TRACE_Q2, scope=news_scope) is None
    assert budget.reserve_search(TRACE_Q2, scope=news_scope) is True


def test_a_fourth_distinct_query_is_refused_and_returns_accumulated_results():
    budget = ResearchBudget(max_search_calls=2, near_duplicate_threshold=0.75)
    budget.record_search("alpha topic one", "A")
    budget.record_search("beta topic two", "B")

    assert budget.reserve_search("gamma topic three") is False
    assert budget.accumulated() == ["A", "B"]


def test_reserve_search_enforces_the_cap_before_any_result_is_recorded():
    # Expresses the race sequentially: two reservations land before either
    # result is recorded, exactly like two parallel tool calls racing ahead
    # of their awaits. The cap must hold on reservations alone.
    budget = ResearchBudget(max_search_calls=2, near_duplicate_threshold=0.75)

    assert budget.reserve_search("alpha topic one") is True
    assert budget.reserve_search("beta topic two") is True
    assert budget.reserve_search("gamma topic three") is False
    assert budget.search_calls == 0


def test_reserve_search_refuses_a_near_duplicate_of_an_already_recorded_query():
    budget = ResearchBudget(max_search_calls=2, near_duplicate_threshold=0.75)
    budget.record_search(TRACE_Q1, "first result")

    assert budget.reserve_search(TRACE_Q2) is False


def test_record_search_releases_the_reservation_for_a_further_distinct_query():
    budget = ResearchBudget(max_search_calls=2, near_duplicate_threshold=0.75)

    assert budget.reserve_search("alpha topic one") is True
    budget.record_search("alpha topic one", "A")

    assert budget.search_calls == 1
    assert budget.reserve_search("beta topic two") is True


def test_a_distinct_image_subject_gets_its_own_search():
    """One image search per turn meant every picture in an answer came from one
    query, so an answer needing an official logo *and* a gameplay shot could
    only ever get two renderings of whichever one was asked for."""
    budget = ResearchBudget(
        max_search_calls=2, near_duplicate_threshold=0.75, max_image_searches=3
    )

    assert budget.reserve_image_search("Pokemon Unite logo") is True
    budget.record_image_search([{"id": "image:logo"}])

    assert budget.reserve_image_search("Pokemon Unite gameplay screenshot") is True
    budget.record_image_search([{"id": "image:gameplay"}])

    assert budget.image_result() == [{"id": "image:logo"}, {"id": "image:gameplay"}]


def test_a_repeated_image_subject_is_refused():
    """Asking twice for the same subject is how duplicates come back."""
    budget = ResearchBudget(
        max_search_calls=2, near_duplicate_threshold=0.75, max_image_searches=3
    )

    assert budget.reserve_image_search("Pokemon Unite gameplay") is True
    assert budget.reserve_image_search("Pokemon Unite gameplay") is False


def test_the_image_search_cap_bounds_the_turn():
    budget = ResearchBudget(
        max_search_calls=2, near_duplicate_threshold=0.75, max_image_searches=2
    )

    assert budget.reserve_image_search("first subject") is True
    assert budget.reserve_image_search("second subject") is True
    assert budget.may_image_search() is False
    assert budget.reserve_image_search("third subject") is False


def test_a_failed_image_search_does_not_buy_a_retry():
    """Matching reserve_search: a reservation is consumed on claim, so a
    provider failure cannot be retried into the same slot."""
    budget = ResearchBudget(
        max_search_calls=2, near_duplicate_threshold=0.75, max_image_searches=1
    )

    assert budget.reserve_image_search("subject") is True
    assert budget.reserve_image_search("different subject") is False


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
