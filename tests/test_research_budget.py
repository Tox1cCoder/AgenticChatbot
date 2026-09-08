from __future__ import annotations

import pytest

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
    budget = ResearchBudget(max_search_calls=2, near_duplicate_threshold=0.75, max_image_searches=3)

    assert budget.reserve_image_search("Pokemon Unite logo") is True
    budget.record_image_search([{"id": "image:logo"}])

    assert budget.reserve_image_search("Pokemon Unite gameplay screenshot") is True
    budget.record_image_search([{"id": "image:gameplay"}])

    assert budget.image_result() == [{"id": "image:logo"}, {"id": "image:gameplay"}]


def test_a_repeated_image_subject_is_refused():
    """Asking twice for the same subject is how duplicates come back."""
    budget = ResearchBudget(max_search_calls=2, near_duplicate_threshold=0.75, max_image_searches=3)

    assert budget.reserve_image_search("Pokemon Unite gameplay") is True
    assert budget.reserve_image_search("Pokemon Unite gameplay") is False


def test_the_image_search_cap_bounds_the_turn():
    budget = ResearchBudget(max_search_calls=2, near_duplicate_threshold=0.75, max_image_searches=2)

    assert budget.reserve_image_search("first subject") is True
    assert budget.reserve_image_search("second subject") is True
    assert budget.may_image_search() is False
    assert budget.reserve_image_search("third subject") is False


def test_a_failed_image_search_does_not_buy_a_retry():
    """Matching reserve_search: a reservation is consumed on claim, so a
    provider failure cannot be retried into the same slot."""
    budget = ResearchBudget(max_search_calls=2, near_duplicate_threshold=0.75, max_image_searches=1)

    assert budget.reserve_image_search("subject") is True
    assert budget.reserve_image_search("different subject") is False


def test_budget_is_per_conversation_and_resettable():
    first = get_research_budget(conversation_id="conv-a")
    first.record_search("alpha", "A")

    assert get_research_budget(conversation_id="conv-a") is first
    assert get_research_budget(conversation_id="conv-b") is not first

    reset_research_budget(conversation_id="conv-a")
    assert get_research_budget(conversation_id="conv-a").search_calls == 0


def test_missing_conversation_id_gets_an_isolated_budget():
    reset_research_budget(conversation_id=None)
    budget = get_research_budget(conversation_id=None)
    budget.record_search("alpha", "A")

    reset_research_budget(conversation_id=None)
    assert get_research_budget(conversation_id=None).search_calls == 0


# ----------------------------------------------------------------------
# keyed by logical turn, and carried across epochs (R4)
# ----------------------------------------------------------------------


def test_the_budget_is_keyed_by_logical_turn_not_conversation():
    """Two turns in one conversation must not share a dedup memory.

    Conversation keying meant a turn in flight shared its allowance and its
    "already searched" memory with any other turn for the same conversation.
    """
    first = get_research_budget(logical_turn_id="turn-1", conversation_id="conv-a")
    second = get_research_budget(logical_turn_id="turn-2", conversation_id="conv-a")

    assert first is not second


def test_the_same_turn_finds_the_same_budget():
    first = get_research_budget(logical_turn_id="turn-3", conversation_id="conv-a")
    first.record_search("alpha", "A")

    assert get_research_budget(logical_turn_id="turn-3", conversation_id="conv-a") is first


def test_a_caller_without_a_turn_falls_back_to_conversation_scope():
    """Not to one shared global bucket, which would leak between turns."""
    scoped = get_research_budget(conversation_id="conv-fallback")

    assert get_research_budget(logical_turn_id=None, conversation_id="conv-fallback") is scoped
    assert get_research_budget(conversation_id="conv-other") is not scoped


def test_the_accessors_are_keyword_only():
    """A positional call would read and write a different key than it meant.

    Before R4 the single positional parameter was the conversation id. Making
    the turn id positional instead would have let every existing call keep
    working against the wrong key — silent, and worse than a crash.
    """
    with pytest.raises(TypeError):
        get_research_budget("conv-a")  # type: ignore[misc]
    with pytest.raises(TypeError):
        reset_research_budget("conv-a")  # type: ignore[misc]


def test_a_persisted_accounting_round_trips():
    from app.ai.research_budget import research_budget_from_state

    budget = ResearchBudget(max_search_calls=2, near_duplicate_threshold=0.75)
    budget.record_search("population of vietnam 2024", "…", scope=("advanced", 5))

    restored = research_budget_from_state(budget.to_state())

    assert restored.searched_in_prior_epoch(
        "population of vietnam 2024", scope=("advanced", 5)
    )


def test_a_scope_survives_the_round_trip_as_a_tuple():
    """A list would never compare equal to a caller's tuple, so the guard
    would silently stop matching and every query would look new."""
    from app.ai.research_budget import research_budget_from_state

    budget = ResearchBudget(max_search_calls=2, near_duplicate_threshold=0.75)
    budget.record_search("alpha beta", "…", scope=("advanced", 5))

    restored = research_budget_from_state(budget.to_state())

    assert restored.searched_in_prior_epoch("alpha beta", scope=("advanced", 5)) is True
    # A different scope is a different search, before and after the round trip.
    assert restored.searched_in_prior_epoch("alpha beta", scope=("basic", 5)) is False


def test_a_continued_epoch_gets_its_call_cap_back():
    """The point of continuing. The allowance resets; the memory does not."""
    from app.ai.research_budget import research_budget_from_state

    budget = ResearchBudget(max_search_calls=2, near_duplicate_threshold=0.75)
    budget.record_search("alpha", "A")
    budget.record_search("beta", "B")
    assert budget.reserve_search("gamma") is False, "the first epoch should be spent"

    restored = research_budget_from_state(budget.to_state())

    assert restored.search_calls == 0
    assert restored.reserve_search("gamma") is True


def test_a_continued_epoch_cannot_re_run_a_query_the_last_one_made():
    """Otherwise Continue is a way around the cap that just refused it."""
    from app.ai.research_budget import research_budget_from_state

    budget = ResearchBudget(max_search_calls=2, near_duplicate_threshold=0.75)
    budget.record_search("population of vietnam", "…")

    restored = research_budget_from_state(budget.to_state())

    assert restored.reserve_search("population of vietnam") is False


def test_a_near_duplicate_of_a_prior_epochs_query_is_also_refused():
    from app.ai.research_budget import research_budget_from_state

    budget = ResearchBudget(max_search_calls=2, near_duplicate_threshold=0.75)
    budget.record_search("population of vietnam 2024", "…")

    restored = research_budget_from_state(budget.to_state())

    assert restored.reserve_search("vietnam population 2024") is False


def test_a_prior_epochs_query_has_no_result_to_reuse():
    """Only tokens are persisted, never result text.

    The next epoch receives the previous one's ToolMessages through
    ``carried_messages``, so the row does not need to carry the evidence — and
    a row is the wrong place to put unbounded provider output.
    """
    from app.ai.research_budget import research_budget_from_state

    budget = ResearchBudget(max_search_calls=2, near_duplicate_threshold=0.75)
    budget.record_search("alpha", "the full provider response")

    state = budget.to_state()

    assert "the full provider response" not in str(state)
    assert research_budget_from_state(state).find_reuse("alpha") is None


def test_an_image_subject_is_not_searched_twice_across_epochs():
    from app.ai.research_budget import research_budget_from_state

    budget = ResearchBudget(
        max_search_calls=2, near_duplicate_threshold=0.75, max_image_searches=1
    )
    assert budget.reserve_image_search("a red bicycle") is True

    restored = research_budget_from_state(budget.to_state())

    assert restored.reserve_image_search("a red bicycle") is False
    # But the slot itself is replenished for a different subject.
    assert restored.reserve_image_search("a blue canoe") is True


def test_the_epoch_count_advances_with_each_restore():
    from app.ai.research_budget import research_budget_from_state

    budget = ResearchBudget(max_search_calls=2, near_duplicate_threshold=0.75)
    budget.record_search("alpha", "A")

    second = research_budget_from_state(budget.to_state())
    third = research_budget_from_state(second.to_state())

    assert (budget.epochs_recorded, second.epochs_recorded, third.epochs_recorded) == (1, 2, 3)


def test_the_persisted_memory_is_bounded():
    """A row is not a place for unbounded growth."""
    from app.ai.research_budget import _MAX_PERSISTED_SEARCHES

    budget = ResearchBudget(max_search_calls=1000, near_duplicate_threshold=1.0)
    for index in range(_MAX_PERSISTED_SEARCHES + 25):
        budget.record_search(f"query number {index}", "…")

    assert len(budget.to_state()["searched"]) == _MAX_PERSISTED_SEARCHES


@pytest.mark.parametrize(
    "payload",
    [
        None,
        "not-an-object",
        {},
        {"schema_version": 999, "searched": []},
        {"schema_version": 1},
        {"schema_version": 1, "searched": "not-a-list"},
        {"schema_version": 1, "searched": ["not-an-object"]},
        {"schema_version": 1, "searched": [{"tokens": "abc", "scope": []}]},
        {"schema_version": 1, "searched": [{"tokens": [], "scope": "nope"}]},
        {"schema_version": 1, "searched": [], "image_subjects": "nope"},
        {"schema_version": 1, "searched": [], "image_subjects": ["not-a-list"]},
    ],
)
def test_an_unreadable_accounting_is_refused_not_emptied(payload):
    """R4's explicit requirement: fail rather than proceed.

    An empty budget is indistinguishable from a fresh turn with a full quota,
    so degrading to one would silently re-run every search the previous epoch
    already paid for — while looking like success.
    """
    from app.ai.research_budget import (
        ResearchAccountingUnreadable,
        research_budget_from_state,
    )

    with pytest.raises(ResearchAccountingUnreadable):
        research_budget_from_state(payload)


def test_a_turn_that_never_searched_snapshots_nothing():
    """``None``, not an empty payload.

    "Never searched" and "restored from a payload with no entries" must stay
    distinguishable, or a rehydration failure could not be told from a turn
    that simply had nothing to carry.
    """
    from app.ai.research_budget import snapshot_research_budget

    reset_research_budget(logical_turn_id="turn-quiet")
    get_research_budget(logical_turn_id="turn-quiet")

    assert snapshot_research_budget(logical_turn_id="turn-quiet") is None


def test_a_turn_that_searched_snapshots_its_memory():
    from app.ai.research_budget import snapshot_research_budget

    reset_research_budget(logical_turn_id="turn-busy")
    get_research_budget(logical_turn_id="turn-busy").record_search("alpha beta", "A")

    state = snapshot_research_budget(logical_turn_id="turn-busy")

    assert state is not None
    assert state["searched"]


def test_installing_an_accounting_replaces_the_live_budget():
    """Both Continue paths rehydrate, including a same-worker one.

    Reusing the in-memory entry for a same-worker Continue would give it a
    spent allowance while a cross-worker Continue got a fresh one — the two
    paths must not disagree.
    """
    from app.ai.research_budget import install_research_budget

    reset_research_budget(logical_turn_id="turn-install")
    live = get_research_budget(logical_turn_id="turn-install")
    live.record_search("alpha", "A")
    live.record_search("beta", "B")
    assert live.reserve_search("gamma") is False

    installed = install_research_budget(live.to_state(), logical_turn_id="turn-install")

    assert get_research_budget(logical_turn_id="turn-install") is installed
    assert installed.reserve_search("gamma") is True
    assert installed.reserve_search("alpha") is False
