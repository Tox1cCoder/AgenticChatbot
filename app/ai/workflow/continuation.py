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
    "pairs_are_intact",
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
