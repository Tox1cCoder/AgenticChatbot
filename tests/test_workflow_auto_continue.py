"""A long plan continues itself instead of stopping to ask.

The budget ladder was built to pause: an epoch that spends its allowance writes
a validated partial answer, offers Continue, and waits for a human to redeem it
through ``POST /messages/continue``. For a plan that runs for hours -- the way
Claude Code or Codex runs one -- that pause is the thing that "stops in the
middle of the plan".

Auto-continue takes the branch the pause node already had. Nothing else about
the ladder changes: the per-epoch counters still reset, evidence is still
carried, and when the turn runs out of epochs ``_is_continuable`` still returns
False and the partial answer is still what the user gets. What goes away is the
wait.

Stop is unaffected and remains the brake -- it is polled by the streaming
consumer at every tool and subagent boundary, not by this node.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from app.ai.workflow.continuation import make_continuation_pause_node
from app.ai.workflow.contracts import OutcomeProvenance, ResponseOutcome


def _outcome() -> ResponseOutcome:
    produced = [
        AIMessage(content="", tool_calls=[{"id": "c1", "name": "web_search", "args": {}}]),
        ToolMessage(content="evidence", tool_call_id="c1", name="web_search"),
        AIMessage(content="partial answer"),
    ]
    return ResponseOutcome(
        agent_id="chat_agent",
        response=AgentResponse(
            agent_type=AgentType.CHAT,
            agent_id="chat_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="partial answer"),
            metadata={"execution_budget": {"exhausted_by": "tool_calls"}},
        ),
        provenance=OutcomeProvenance(
            output_policy_ids=("public_content",), private_messages=produced
        ),
    )


def _paused_state(**overrides):
    state = {
        "agent_outcome": _outcome(),
        "active_agent_id": "chat_agent",
        "execution_epoch": 0,
        "execution_budget": {"exhausted_by": "tool_calls"},
        "turn_identity": None,
        "conversation_id": "conversation-1",
        "user_id": "user-1",
    }
    state.update(overrides)
    return state


class _Interrupt:
    """Records whether the turn stopped to ask, and answers Continue if it did."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, payload):
        self.calls.append(payload)
        return {"action": "continue", "expected_epoch": payload.get("execution_epoch", 0)}


@pytest.mark.asyncio
async def test_auto_continue_never_stops_to_ask():
    interrupt = _Interrupt()
    node = make_continuation_pause_node(interrupt_fn=interrupt, auto_continue=True)

    command = await node(_paused_state())

    assert interrupt.calls == [], "the turn paused for a human it did not need"
    assert command.goto == "chat_agent"
    assert command.update["execution_epoch"] == 1
    assert command.update["execution_phase"] == "executing"


@pytest.mark.asyncio
async def test_auto_continue_still_resets_the_epoch_budget():
    """The ladder is unchanged -- each epoch gets its own room and only its own."""
    node = make_continuation_pause_node(interrupt_fn=_Interrupt(), auto_continue=True)

    command = await node(_paused_state())

    assert command.update["execution_budget"] is None


@pytest.mark.asyncio
async def test_auto_continue_carries_the_evidence_forward():
    """Work already done must survive the rollover, or the epoch is wasted."""
    from app.ai.workflow.continuation import pairs_are_intact

    node = make_continuation_pause_node(interrupt_fn=_Interrupt(), auto_continue=True)

    command = await node(_paused_state())
    carried = command.update["carried_messages"]

    assert pairs_are_intact(carried) is True
    assert any(isinstance(message, ToolMessage) for message in carried)


@pytest.mark.asyncio
async def test_auto_continue_resumes_from_whatever_epoch_the_turn_reached():
    """Epoch 7 continues into 8. Nothing here assumes the first rollover."""
    node = make_continuation_pause_node(interrupt_fn=_Interrupt(), auto_continue=True)

    command = await node(_paused_state(execution_epoch=7))

    assert command.update["execution_epoch"] == 8


@pytest.mark.asyncio
async def test_asking_is_still_what_happens_when_auto_continue_is_off():
    """The pause is not deleted, only skipped -- the flag must restore it."""
    interrupt = _Interrupt()
    node = make_continuation_pause_node(interrupt_fn=interrupt, auto_continue=False)

    command = await node(_paused_state())

    assert len(interrupt.calls) == 1, "the turn should have offered Continue"
    assert command.update["execution_epoch"] == 1
