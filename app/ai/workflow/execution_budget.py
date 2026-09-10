"""Per-epoch execution accounting, shared by every path that runs a model.

A turn that exhausts its budget should still answer. That is the whole point of
a *soft* limit: it stops one call short, withholds the tools, tells the model
that evidence gathering has ended, and what comes back is a validated partial
the user can read and continue from. A limit that simply raises produces a
generic error and throws away everything the turn already gathered.

Framework-free by design. The obvious home for these counters is an
``AgentMiddleware``, and that was the plan's first draft — but it would cover
only the specialists built through ``create_agent``. ``rag_agent`` runs the
shared compiled RAG graph, and Planning runs parent graph nodes with delegated
workers; both would have had no budget at all, silently. So the accountant is a
plain object, the middleware is a thin adapter over it, and RAG and Planning
call the same methods.

Two counters, one outcome. Whichever limit is reached first sets
``forced_synthesis``, and from then on every model call is tool-free until the
epoch changes. ``exhausted_by`` keeps the *first* reason, because that is the
one that explains the answer — except a hard limit, which overrides everything
since it means this module failed to reserve the synthesis call at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

__all__ = [
    "BudgetDecision",
    "ExecutionBudgetAccountant",
    "ExecutionBudgetLimits",
    "ExecutionBudgetState",
    "ExhaustionReason",
]

ExhaustionReason = Literal["model_calls", "tool_calls", "hard_limit"]

#: Fallbacks for a settings object that does not declare a field. Kept here
#: rather than importing ``Settings`` so this module stays free of app config;
#: a test asserts they equal the declared defaults.
DEFAULT_LIMITS: dict[str, int] = {
    "generation_soft_model_calls_per_epoch": 24,
    "generation_hard_model_calls_per_epoch": 28,
    "generation_soft_tool_calls_per_epoch": 48,
    "generation_hard_tool_calls_per_epoch": 56,
    "generation_total_epochs_per_turn": 10,
}


def _setting(settings: Any, name: str) -> int:
    value = getattr(settings, name, None)
    return int(DEFAULT_LIMITS[name] if value is None else value)

#: Appended to the reserved synthesis call. Server-owned: a model that could
#: talk itself out of answering would defeat the reservation.
FORCED_SYNTHESIS_INSTRUCTION = (
    "Evidence gathering has ended for this execution epoch. Answer now using "
    "only the evidence already present. State uncertainty and missing facts "
    "explicitly. Do not request, promise, or imply another automatic tool call."
)


class ExecutionBudgetState(BaseModel):
    """What one epoch has spent, and what the turn has spent overall.

    The per-epoch counters reset on Continue; the ``turn_*`` totals do not.
    Without that split, Continue is an unlimited budget — press it enough times
    and the per-epoch cap means nothing.
    """

    execution_epoch: int = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    forced_synthesis: bool = False
    exhausted_by: ExhaustionReason | None = None

    turn_model_calls: int = Field(default=0, ge=0)
    turn_tool_calls: int = Field(default=0, ge=0)
    epochs_used: int = Field(default=1, ge=1)


class ExecutionBudgetLimits(BaseModel):
    """The ladder. Every hard rung must sit above its soft rung."""

    soft_model_calls: int = Field(ge=1)
    hard_model_calls: int = Field(ge=2)
    soft_tool_calls: int = Field(ge=1)
    hard_tool_calls: int = Field(ge=2)
    total_epochs_per_turn: int = Field(ge=1)

    @model_validator(mode="after")
    def _hard_exceeds_soft(self) -> ExecutionBudgetLimits:
        # If they were equal the framework's own limit would raise on the very
        # call this module reserved for the answer, which is the failure the
        # soft budget exists to prevent.
        if self.hard_model_calls <= self.soft_model_calls:
            raise ValueError("hard_model_calls must exceed soft_model_calls")
        if self.hard_tool_calls <= self.soft_tool_calls:
            raise ValueError("hard_tool_calls must exceed soft_tool_calls")
        return self

    @classmethod
    def from_settings(cls, settings: Any) -> ExecutionBudgetLimits:
        """Read the ladder, tolerating a settings object that lacks a field.

        Partial settings doubles are the established pattern in this codebase
        (``SpecialistFactory._limit`` reads them the same way), and a test that
        cares about one limit should not have to declare five. The real
        ``Settings`` always carries all of them, and its cross-field validator
        is what guarantees each hard rung sits above its soft one --
        ``test_the_defaults_match_the_settings_declaration`` pins these
        fallbacks to it.
        """
        return cls(
            soft_model_calls=_setting(settings, "generation_soft_model_calls_per_epoch"),
            hard_model_calls=_setting(settings, "generation_hard_model_calls_per_epoch"),
            soft_tool_calls=_setting(settings, "generation_soft_tool_calls_per_epoch"),
            hard_tool_calls=_setting(settings, "generation_hard_tool_calls_per_epoch"),
            total_epochs_per_turn=_setting(settings, "generation_total_epochs_per_turn"),
        )


@dataclass(frozen=True)
class BudgetDecision:
    """What the caller must do about the call it is asking about."""

    allowed: bool
    tools_suppressed: bool = False
    forced_synthesis: bool = False
    reason: ExhaustionReason | None = None


class ExecutionBudgetAccountant:
    """Counts one epoch's calls and decides when to reserve the answer."""

    def __init__(
        self,
        *,
        limits: ExecutionBudgetLimits,
        state: ExecutionBudgetState | None = None,
    ) -> None:
        self._limits = limits
        self._state = state.model_copy(deep=True) if state is not None else ExecutionBudgetState()

    @property
    def state(self) -> ExecutionBudgetState:
        return self._state

    @property
    def limits(self) -> ExecutionBudgetLimits:
        return self._limits

    # ------------------------------------------------------------------
    # accounting
    # ------------------------------------------------------------------

    def note_tool_call(self) -> BudgetDecision:
        """Ask whether one tool call may run. Call this *before* running it.

        A refusal is not an error: the caller pairs a synthetic ``ToolMessage``
        with the call it refused, so the transcript stays valid, and the next
        model call becomes the answer.
        """
        if self._state.forced_synthesis:
            return BudgetDecision(
                allowed=False, forced_synthesis=True, reason=self._state.exhausted_by
            )
        if self._state.tool_calls >= self._limits.soft_tool_calls:
            self._force("tool_calls")
            return BudgetDecision(allowed=False, forced_synthesis=True, reason="tool_calls")
        self._state.tool_calls += 1
        self._state.turn_tool_calls += 1
        return BudgetDecision(allowed=True)

    def note_model_call(self) -> BudgetDecision:
        """Ask how one model call must be made. Always allowed.

        Refusing a model call would leave the turn with no answer at all, which
        is the outcome this budget exists to avoid. What changes is whether the
        call carries tools.
        """
        if not self._state.forced_synthesis and (
            self._state.model_calls >= self._limits.soft_model_calls - 1
        ):
            self._force("model_calls")
        self._state.model_calls += 1
        self._state.turn_model_calls += 1
        suppressed = self._state.forced_synthesis
        return BudgetDecision(
            allowed=True,
            tools_suppressed=suppressed,
            forced_synthesis=suppressed,
            reason=self._state.exhausted_by,
        )

    def note_hard_limit(self) -> ExecutionBudgetState:
        """Record that the framework's own limit fired.

        This means the soft budget did not reserve the answer, so it overrides
        whatever reason was recorded before: the useful signal is that the
        ladder was misconfigured or bypassed, not which rung was touched first.
        """
        logger.warning(
            "Execution hard limit reached in epoch %s after %s model and %s tool calls",
            self._state.execution_epoch,
            self._state.model_calls,
            self._state.tool_calls,
        )
        self._state.forced_synthesis = True
        self._state.exhausted_by = "hard_limit"
        return self._state

    # ------------------------------------------------------------------
    # epochs
    # ------------------------------------------------------------------

    def begin_epoch(self, epoch: int) -> ExecutionBudgetState:
        """Give the next epoch its room back, and only its room."""
        self._state.execution_epoch = int(epoch)
        self._state.model_calls = 0
        self._state.tool_calls = 0
        self._state.forced_synthesis = False
        self._state.exhausted_by = None
        self._state.epochs_used = int(epoch) + 1
        return self._state

    @property
    def epochs_remaining(self) -> int:
        return max(0, self._limits.total_epochs_per_turn - self._state.epochs_used)

    @property
    def can_begin_epoch(self) -> bool:
        return self.epochs_remaining > 0

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _force(self, reason: ExhaustionReason) -> None:
        self._state.forced_synthesis = True
        if self._state.exhausted_by is None:
            self._state.exhausted_by = reason
