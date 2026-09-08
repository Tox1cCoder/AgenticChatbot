"""Pausing a validated partial answer, and resuming it with its evidence.

The pause is deliberately *after* validation. An answer the user is shown and
offered Continue on has passed exactly the checks a finished answer passes;
pausing earlier would mean offering to continue an unchecked draft.

The harder half is what Continue hands the next epoch. A specialist's next
invocation is assembled from history plus the current-turn slice
(``SpecialistFactory._invocation_messages``), and everything an epoch produced
is sliced off into ``outcome.provenance.private_messages`` -- so resetting the
counters and jumping back to the specialist gives epoch 2 the original question
and no evidence. It would then redo the work epoch 1 already paid for, with a
budget that has just been refilled. :func:`carry_messages` is what prevents
that.

Carrying has one hard rule: a tool call and the result answering it travel
together or not at all. Providers reject a transcript containing an unanswered
tool call, and an orphaned ``ToolMessage`` is a request error rather than a
merely degraded prompt -- so a half-round is dropped, not repaired.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

__all__ = [
    "ContinuationPausePayload",
    "ContinuationResume",
    "carry_messages",
    "make_continuation_pause_node",
    "pairs_are_intact",
    "pending_continuation_payload",
]


class ContinuationPausePayload(BaseModel):
    """What the client is shown when a turn pauses at its budget.

    ``type`` is load-bearing. Tool approval and this both arrive as LangGraph
    interrupts, and confusing them would ask a human to approve a tool call
    that does not exist -- so the discriminator is a literal, not a convention.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["execution_budget_exhausted"] = "execution_budget_exhausted"
    generation_id: str
    logical_turn_id: str
    execution_epoch: int = Field(ge=0)
    active_agent_id: str
    validated_content: str
    budget: dict[str, Any] = Field(default_factory=dict)


class ContinuationResume(BaseModel):
    """The decision a resumed pause carries back into the graph."""

    model_config = ConfigDict(frozen=True)

    action: Literal["continue", "stop"]
    continuation_id: str
    expected_epoch: int = Field(ge=0)


def carry_messages(produced: list[Any] | tuple[Any, ...]) -> list[BaseMessage]:
    """The evidence from one epoch, in a shape the next one can be sent.

    Kept: complete tool rounds -- the assistant turn requesting a call and the
    ``ToolMessage`` answering it.

    Dropped:

    * the epoch's final answer, which is already persisted and shown, so
      carrying it would repeat it to the model as if it were still deciding;
    * any human message, because the question is already in the current-turn
      slice and a second copy invites the model to answer it twice;
    * any half-round, because a provider rejects an unanswered tool call and an
      orphaned result is a request error rather than a weaker prompt.

    Offloaded results are carried exactly as they are -- a ``blob_id``
    reference stays a reference. Rehydrating it here would undo the offload the
    size limit required in the first place.
    """
    messages = [message for message in produced if isinstance(message, BaseMessage)]
    answered = {
        str(getattr(message, "tool_call_id", "") or "")
        for message in messages
        if isinstance(message, ToolMessage)
    }

    carried: list[BaseMessage] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            # Appended only alongside the request it answers, which the branch
            # below has already decided to keep.
            if _requested_ids(carried) & {str(message.tool_call_id or "")}:
                carried.append(message)
            continue
        if not isinstance(message, AIMessage):
            continue
        requested = _tool_call_ids(message)
        if not requested:
            # A plain assistant turn is this epoch's answer, not its evidence.
            continue
        if not requested <= answered:
            logger.info(
                "Dropped an unanswered tool round from the carried transcript: %s",
                sorted(requested - answered),
            )
            continue
        carried.append(message)

    return carried


def pairs_are_intact(messages: list[BaseMessage] | tuple[BaseMessage, ...]) -> bool:
    """Whether every tool call in ``messages`` has its answer, and vice versa."""
    requested: set[str] = set()
    answered: set[str] = set()
    for message in messages:
        if isinstance(message, ToolMessage):
            answered.add(str(message.tool_call_id or ""))
        elif isinstance(message, AIMessage):
            requested |= _tool_call_ids(message)
    return requested == answered


def _tool_call_ids(message: Any) -> set[str]:
    return {
        str(call.get("id") or "")
        for call in (getattr(message, "tool_calls", None) or [])
        if isinstance(call, dict)
    }


def _requested_ids(messages: list[BaseMessage]) -> set[str]:
    ids: set[str] = set()
    for message in messages:
        if isinstance(message, AIMessage):
            ids |= _tool_call_ids(message)
    return ids


def make_continuation_pause_node(*, interrupt_fn: Any = None) -> Any:
    """Build the node that offers Continue on a validated partial answer.

    Reached only from ``validate_output``, and only for an outcome whose budget
    was exhausted. Everything it hands the client has therefore already passed
    the same validation a finished answer passes.

    Every refusal goes to ``finalize`` rather than raising. This node sits on
    the only path a paused turn can leave by, so raising here would strand the
    turn active with nothing running -- and ``finalize`` is the single terminal
    boundary the whole graph is built around.

    ``interrupt_fn`` is injected so the decision can be scripted in a test;
    production passes LangGraph's own ``interrupt``.
    """
    from langgraph.types import Command, interrupt

    resume_with = interrupt_fn or interrupt

    async def continuation_pause(state: dict[str, Any], runtime: Any = None) -> Any:
        from app.ai.workflow.specialists import resolve_node_for_agent_id

        outcome = state.get("agent_outcome")
        active_agent_id = state.get("active_agent_id")
        epoch = int(state.get("execution_epoch") or 0)
        payload = ContinuationPausePayload(
            generation_id=str(state.get("generation_id") or ""),
            logical_turn_id=str(state.get("logical_turn_id") or ""),
            execution_epoch=epoch,
            active_agent_id=str(active_agent_id or ""),
            validated_content=_validated_content(outcome),
            budget=dict(state.get("execution_budget") or {}),
        )

        decision = resume_with(payload.model_dump(mode="json"))
        action, expected_epoch = _read_decision(decision)

        if action != "continue":
            return Command(update={"execution_phase": "finalizing"}, goto="finalize")
        if expected_epoch != epoch:
            # A Continue issued against an epoch this turn has left. Running it
            # would open a second epoch on top of whatever already moved.
            logger.warning(
                "Refused a continuation for epoch %s; the turn is at epoch %s",
                expected_epoch,
                epoch,
            )
            return Command(update={"execution_phase": "finalizing"}, goto="finalize")

        node = resolve_node_for_agent_id(active_agent_id)
        if not node:
            logger.error("Cannot continue: no graph node for agent %r", active_agent_id)
            return Command(update={"execution_phase": "finalizing"}, goto="finalize")

        return Command(
            update={
                "execution_epoch": epoch + 1,
                "execution_budget": None,
                "execution_phase": "executing",
                "carried_messages": carry_messages(
                    list(getattr(getattr(outcome, "provenance", None), "private_messages", ()))
                ),
            },
            # Straight back to the agent the turn already chose. Routing again
            # could land a continuation on a different specialist than the one
            # whose evidence it is carrying.
            goto=node,
        )

    return continuation_pause


def _validated_content(outcome: Any) -> str:
    message = getattr(getattr(outcome, "response", None), "message", None)
    return str(getattr(message, "content", "") or "")


def _read_decision(decision: Any) -> tuple[str, int]:
    """Read a resume value, defaulting to stop for anything unrecognised.

    Defaulting to stop is the safe direction: it keeps the validated partial
    the user already has, where guessing "continue" would spend another epoch
    on a decision nobody made.
    """
    if isinstance(decision, ContinuationResume):
        return decision.action, decision.expected_epoch
    if isinstance(decision, dict):
        action = str(decision.get("action") or "stop")
        try:
            return action, int(decision.get("expected_epoch") or 0)
        except (TypeError, ValueError):
            return action, -1
    logger.warning("Unreadable continuation resume value of type %s", type(decision).__name__)
    return "stop", -1


def pending_continuation_payload(snapshot: Any) -> ContinuationPausePayload | None:
    """The live budget pause on a checkpoint, or ``None``.

    Deliberately the mirror of ``hitl_config.pending_interrupt_payload``, and
    deliberately disjoint from it: that one matches on ``action_requests`` and
    this one on the ``type`` literal, so a budget pause can never be presented
    to a human as a tool approval, nor an approval resumed as a budget
    decision. ``tests/test_workflow_continuation.py`` asserts both directions.

    Never raises. A checkpoint is read on every resume, so an unparseable value
    has to mean "not a pause" rather than strand the turn.
    """
    for item in _live_interrupts(snapshot):
        value = getattr(item, "value", None)
        if not isinstance(value, dict):
            continue
        if value.get("type") != "execution_budget_exhausted":
            continue
        try:
            return ContinuationPausePayload.model_validate(value)
        except Exception:
            logger.warning("Ignored an unreadable continuation pause payload")
            return None
    return None


def _live_interrupts(snapshot: Any) -> list[Any]:
    """Interrupts whose task has not produced a result yet.

    ``task.result is None`` is the whole test. LangGraph keeps reporting an
    answered interrupt on both the snapshot and its task, so anything looser
    would re-present a pause the user already decided.
    """
    tasks = list(getattr(snapshot, "tasks", None) or ())
    if tasks:
        return [
            item
            for task in tasks
            if getattr(task, "result", None) is None
            for item in (getattr(task, "interrupts", None) or ())
        ]
    return list(getattr(snapshot, "interrupts", None) or ())
