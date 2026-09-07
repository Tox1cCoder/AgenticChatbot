"""One accountant for every execution path, not one middleware for some.

The soft budget exists so a turn that runs out of room still answers. Its whole
value is that the *last* call is reserved: tools are withheld, the model is told
evidence gathering has ended, and what comes back is a validated partial the
user can read and continue from. A limit that simply raises produces a generic
error instead.

This module is framework-free on purpose. Putting the counters in an
``AgentMiddleware`` would cover only the agents built through ``create_agent`` —
RAG runs the shared compiled graph and Planning runs parent nodes, so both
would silently have no budget at all. The middleware is a thin adapter over
this; RAG and Planning call the same object.
"""

from __future__ import annotations

import pytest

from app.ai.workflow.execution_budget import (
    ExecutionBudgetAccountant,
    ExecutionBudgetLimits,
    ExecutionBudgetState,
)


def _limits(**overrides) -> ExecutionBudgetLimits:
    values = {
        "soft_model_calls": 3,
        "hard_model_calls": 5,
        "soft_tool_calls": 4,
        "hard_tool_calls": 6,
        "total_epochs_per_turn": 3,
    }
    values.update(overrides)
    return ExecutionBudgetLimits(**values)


def _accountant(**overrides) -> ExecutionBudgetAccountant:
    return ExecutionBudgetAccountant(limits=_limits(**overrides))


# ----------------------------------------------------------------------
# limits
# ----------------------------------------------------------------------


def test_a_hard_limit_must_exceed_its_soft_limit():
    """Otherwise the framework's error fires before synthesis is reserved."""
    with pytest.raises(ValueError, match="hard_model_calls"):
        _limits(soft_model_calls=5, hard_model_calls=5)

    with pytest.raises(ValueError, match="hard_tool_calls"):
        _limits(soft_tool_calls=6, hard_tool_calls=6)


def test_limits_come_from_settings_by_name():
    class _Settings:
        generation_soft_model_calls_per_epoch = 7
        generation_hard_model_calls_per_epoch = 9
        generation_soft_tool_calls_per_epoch = 12
        generation_hard_tool_calls_per_epoch = 16
        generation_total_epochs_per_turn = 5

    limits = ExecutionBudgetLimits.from_settings(_Settings())

    assert limits.soft_model_calls == 7
    assert limits.hard_tool_calls == 16
    assert limits.total_epochs_per_turn == 5


# ----------------------------------------------------------------------
# tool budget
# ----------------------------------------------------------------------


def test_a_tool_call_below_the_soft_limit_runs():
    accountant = _accountant()

    for _ in range(3):
        assert accountant.note_tool_call().allowed is True

    assert accountant.state.tool_calls == 3
    assert accountant.state.forced_synthesis is False


def test_the_call_that_reaches_the_soft_limit_still_runs():
    """The threshold is the last tool call, not the first refusal."""
    accountant = _accountant(soft_tool_calls=4)

    decisions = [accountant.note_tool_call() for _ in range(4)]

    assert all(decision.allowed for decision in decisions)
    assert accountant.state.tool_calls == 4


def test_the_next_tool_call_after_the_soft_limit_is_refused_and_forces_synthesis():
    accountant = _accountant(soft_tool_calls=4)
    for _ in range(4):
        accountant.note_tool_call()

    decision = accountant.note_tool_call()

    assert decision.allowed is False
    assert decision.forced_synthesis is True
    assert accountant.state.forced_synthesis is True
    assert accountant.state.exhausted_by == "tool_calls"


def test_a_refused_tool_call_is_not_counted():
    """It never ran. Counting it would misreport what the turn spent."""
    accountant = _accountant(soft_tool_calls=4)
    for _ in range(4):
        accountant.note_tool_call()

    accountant.note_tool_call()

    assert accountant.state.tool_calls == 4


def test_every_further_tool_call_stays_refused():
    accountant = _accountant(soft_tool_calls=1)
    accountant.note_tool_call()

    assert accountant.note_tool_call().allowed is False
    assert accountant.note_tool_call().allowed is False


# ----------------------------------------------------------------------
# model budget
# ----------------------------------------------------------------------


def test_model_calls_below_the_soft_limit_keep_their_tools():
    accountant = _accountant(soft_model_calls=3)

    first = accountant.note_model_call()
    second = accountant.note_model_call()

    assert first.tools_suppressed is False
    assert second.tools_suppressed is False


def test_the_last_model_call_in_the_budget_is_the_reserved_synthesis():
    """Reaching the soft limit does not fail the turn; it answers it."""
    accountant = _accountant(soft_model_calls=3)
    accountant.note_model_call()
    accountant.note_model_call()

    decision = accountant.note_model_call()

    assert decision.allowed is True
    assert decision.tools_suppressed is True
    assert decision.forced_synthesis is True
    assert accountant.state.exhausted_by == "model_calls"


def test_a_forced_synthesis_from_the_tool_budget_suppresses_the_next_model_call(
):
    """The two budgets share one outcome: answer now, with no tools."""
    accountant = _accountant(soft_tool_calls=1)
    accountant.note_tool_call()
    accountant.note_tool_call()

    decision = accountant.note_model_call()

    assert decision.allowed is True
    assert decision.tools_suppressed is True
    assert accountant.state.exhausted_by == "tool_calls"


def test_the_first_exhaustion_reason_is_the_one_reported():
    """Two limits can both be reached; the answer names why it stopped first."""
    accountant = _accountant(soft_tool_calls=1, soft_model_calls=2)
    accountant.note_tool_call()
    accountant.note_tool_call()
    accountant.note_model_call()
    accountant.note_model_call()

    assert accountant.state.exhausted_by == "tool_calls"


# ----------------------------------------------------------------------
# hard limit
# ----------------------------------------------------------------------


def test_a_hard_limit_is_recorded_as_a_budget_outcome_not_an_error():
    """The framework's own limit raising is a defect, not a user-facing error.

    It means the soft budget failed to reserve synthesis, and the honest
    recovery is a deterministic partial answer from the evidence already
    gathered — not ``agent_execution_limit`` shown to a client.
    """
    accountant = _accountant()

    state = accountant.note_hard_limit()

    assert state.exhausted_by == "hard_limit"
    assert state.forced_synthesis is True


def test_a_hard_limit_overrides_an_earlier_soft_reason():
    accountant = _accountant(soft_tool_calls=1)
    accountant.note_tool_call()
    accountant.note_tool_call()

    state = accountant.note_hard_limit()

    assert state.exhausted_by == "hard_limit"


# ----------------------------------------------------------------------
# epochs
# ----------------------------------------------------------------------


def test_a_new_epoch_resets_the_per_epoch_counters():
    accountant = _accountant(soft_tool_calls=1)
    accountant.note_tool_call()
    accountant.note_tool_call()

    accountant.begin_epoch(1)

    assert accountant.state.execution_epoch == 1
    assert accountant.state.tool_calls == 0
    assert accountant.state.forced_synthesis is False
    assert accountant.state.exhausted_by is None


def test_a_new_epoch_keeps_the_cumulative_turn_totals():
    """R4. Continue replenishes the epoch's room, not the turn's history.

    Without this, Continue is an unlimited budget: press it enough times and
    the per-epoch cap means nothing.
    """
    accountant = _accountant()
    accountant.note_tool_call()
    accountant.note_tool_call()
    accountant.note_model_call()

    accountant.begin_epoch(1)

    assert accountant.state.turn_tool_calls == 2
    assert accountant.state.turn_model_calls == 1
    assert accountant.state.epochs_used == 2


def test_the_turn_totals_keep_accumulating_across_epochs():
    accountant = _accountant()
    accountant.note_tool_call()
    accountant.begin_epoch(1)
    accountant.note_tool_call()

    assert accountant.state.turn_tool_calls == 2
    assert accountant.state.tool_calls == 1


def test_a_turn_that_has_used_every_epoch_reports_it():
    accountant = _accountant(total_epochs_per_turn=2)

    accountant.begin_epoch(1)

    assert accountant.epochs_remaining == 0
    assert accountant.can_begin_epoch is False


def test_a_turn_with_epochs_left_reports_that_too():
    accountant = _accountant(total_epochs_per_turn=3)

    assert accountant.epochs_remaining == 2
    assert accountant.can_begin_epoch is True


# ----------------------------------------------------------------------
# persistence
# ----------------------------------------------------------------------


def test_the_state_round_trips_so_another_worker_can_resume_it():
    """A Continue may be served by a process that never saw epoch 1."""
    accountant = _accountant()
    accountant.note_tool_call()
    accountant.note_model_call()
    stored = accountant.state.model_dump(mode="json")

    restored = ExecutionBudgetAccountant(
        limits=_limits(), state=ExecutionBudgetState.model_validate(stored)
    )

    assert restored.state == accountant.state


def test_an_absent_stored_state_starts_a_fresh_budget():
    accountant = ExecutionBudgetAccountant(limits=_limits(), state=None)

    assert accountant.state.model_calls == 0
    assert accountant.state.execution_epoch == 0


def test_the_state_is_json_safe_for_the_generation_row():
    accountant = _accountant()
    accountant.note_hard_limit()

    stored = accountant.state.model_dump(mode="json")

    assert stored["exhausted_by"] == "hard_limit"
    assert isinstance(stored["forced_synthesis"], bool)


def test_the_defaults_match_the_settings_declaration():
    """Two copies of a default is two ladders, and only one is validated."""
    from app.ai.workflow.execution_budget import DEFAULT_LIMITS
    from app.core.config import Settings

    for name, fallback in DEFAULT_LIMITS.items():
        assert Settings.model_fields[name].default == fallback


def test_a_partial_settings_object_still_yields_a_valid_ladder():
    limits = ExecutionBudgetLimits.from_settings(object())

    assert limits.hard_model_calls > limits.soft_model_calls
    assert limits.hard_tool_calls > limits.soft_tool_calls
