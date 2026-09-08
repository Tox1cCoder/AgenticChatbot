"""What a paused turn looks like on the wire.

Task 4 built the pause and a reader for it, but nothing *emitted* it. That gap
was not cosmetic: ``_finish_stream`` looks for a tool-approval interrupt, and a
budget pause is deliberately invisible to that reader (``pending_interrupt_payload``
matches on ``action_requests``, which a budget pause has none of). So the
snapshot had a next node, no interrupt event was produced, ``finalize`` had
never run to leave a response behind, and the turn fell through to
``NO_RESPONSE_GENERATED`` -- a validated partial answer reported to the client
as a generic failure.

The two pause kinds must stay separable in both directions. Projecting a budget
pause as an ``interrupt`` would ask a human to approve tool calls that do not
exist; projecting an approval request as ``continuation_available`` would offer
Continue on a turn that is waiting for a decision nobody has been asked for.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.ai.graph import MultiAgentWorkflow
from app.core.response_constants import NO_RESPONSE_GENERATED
from app.services.event_streaming.events import StreamEventType

pytestmark = pytest.mark.asyncio


class _Interrupt:
    def __init__(self, value: Any, interrupt_id: str = "i1") -> None:
        self.value = value
        self.id = interrupt_id


class _Task:
    """A checkpoint task. ``result is None`` is what makes its interrupt live.

    LangGraph keeps reporting an interrupt after it has been answered, so a
    reader that ignores ``result`` re-presents decisions the user already made.
    """

    def __init__(self, interrupts: list[_Interrupt], result: Any = None) -> None:
        self.interrupts = interrupts
        self.result = result


class _Snapshot:
    def __init__(
        self,
        tasks: list[_Task],
        *,
        next_nodes: tuple[str, ...] = ("chat_agent",),
        values: dict[str, Any] | None = None,
    ) -> None:
        self.tasks = tasks
        self.next = next_nodes
        self.values = values or {}


def _budget_pause(
    *,
    epoch: int = 0,
    content: str = "Here is what I found before running out of budget.",
) -> _Interrupt:
    return _Interrupt(
        {
            "type": "execution_budget_exhausted",
            "generation_id": "11111111-1111-1111-1111-111111111111",
            "logical_turn_id": "turn-7",
            "execution_epoch": epoch,
            "active_agent_id": "search_agent",
            "validated_content": content,
            "budget": {"model_calls": 7, "tool_calls": 12, "exhausted_by": "tool_calls"},
        }
    )


def _approval_pause() -> _Interrupt:
    return _Interrupt(
        {
            "action_requests": [
                {"tool_call_id": "call-1", "name": "send_email", "args": {"to": "a@b.test"}}
            ],
            "metadata": {"origin": "client"},
        }
    )


def _workflow(snapshot: _Snapshot) -> MultiAgentWorkflow:
    """A workflow whose only live behaviour is reading one checkpoint.

    ``__new__`` on purpose: constructing the real thing needs Qdrant, an
    embedding service and a model resolver, none of which this boundary touches.
    """
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.checkpointer = SimpleNamespace()

    async def aget_state(_config):
        return snapshot

    workflow.graph = SimpleNamespace(aget_state=aget_state)
    return workflow


async def _finish(snapshot: _Snapshot) -> list[Any]:
    from app.ai.graph import StreamProjectionContext

    workflow = _workflow(snapshot)
    ctx = StreamProjectionContext(last_emitted_agent="search_agent", suppress_tokens=False)
    return [
        event
        async for event in workflow._finish_stream(
            ctx, config={}, thread_id="thread-1", conversation_id="conv-1"
        )
    ]


# ----------------------------------------------------------------------
# a budget pause reaches the client as a typed continuation offer
# ----------------------------------------------------------------------


async def test_a_budget_pause_is_not_reported_as_a_failed_turn():
    """The regression this file exists for."""
    events = await _finish(_Snapshot([_Task([_budget_pause()])]))

    errors = [event for event in events if event.type == "error"]
    assert not errors, f"a validated partial was reported as an error: {errors}"
    assert [event.type for event in events] == ["continuation_available"]


async def test_the_continuation_event_carries_the_paused_identity():
    events = await _finish(_Snapshot([_Task([_budget_pause(epoch=2)])]))

    data = events[0].data
    assert data["generation_id"] == "11111111-1111-1111-1111-111111111111"
    assert data["logical_turn_id"] == "turn-7"
    assert data["execution_epoch"] == 2
    assert data["active_agent_id"] == "search_agent"
    assert data["thread_id"] == "thread-1"


async def test_the_continuation_event_carries_the_validated_partial_and_its_budget():
    """The graph's own event is internal, and the service needs both.

    ``message_service`` persists this content as the partial assistant message
    before any Continue is offered, and the budget is why the turn stopped.
    Task 6's *public* projection strips the content -- this one cannot, or there
    would be nothing to persist.
    """
    events = await _finish(_Snapshot([_Task([_budget_pause(content="Partial findings.")])]))

    data = events[0].data
    assert data["validated_content"] == "Partial findings."
    assert data["budget"]["exhausted_by"] == "tool_calls"


async def test_continuation_available_is_a_declared_canonical_event_type():
    """A type the enum does not declare cannot be projected by an adapter."""
    assert "continuation_available" in getattr(StreamEventType, "__args__", ())


# ----------------------------------------------------------------------
# the two pause kinds stay separable
# ----------------------------------------------------------------------


async def test_a_tool_approval_pause_still_emits_an_interrupt():
    events = await _finish(_Snapshot([_Task([_approval_pause()])]))

    assert [event.type for event in events] == ["interrupt"]
    assert events[0].data["pending_tool_calls"][0]["name"] == "send_email"


async def test_a_budget_pause_is_never_projected_as_a_tool_approval():
    events = await _finish(_Snapshot([_Task([_budget_pause()])]))

    assert all(event.type != "interrupt" for event in events)
    assert all("pending_tool_calls" not in event.data for event in events)


async def test_an_approval_pause_is_never_projected_as_a_continuation_offer():
    events = await _finish(_Snapshot([_Task([_approval_pause()])]))

    assert all(event.type != "continuation_available" for event in events)


async def test_an_approval_pause_wins_when_a_turn_somehow_holds_both():
    """A human waiting on a decision outranks an offer to continue.

    This should not arise -- the pause node is reached only from
    ``validate_output``, by which point tool execution is over. If it ever
    does, resolving the approval is the only move that unblocks the turn, so
    Continue must not be offered as an alternative to it.
    """
    snapshot = _Snapshot([_Task([_approval_pause()]), _Task([_budget_pause()])])

    events = await _finish(snapshot)

    assert [event.type for event in events] == ["interrupt"]


# ----------------------------------------------------------------------
# nothing else about the terminal path moved
# ----------------------------------------------------------------------


async def test_an_answered_budget_pause_is_not_re_offered():
    """A task with a result has already been decided."""
    snapshot = _Snapshot(
        [_Task([_budget_pause()], result={"action": "continue"})],
        next_nodes=(),
        values={"response": None},
    )

    events = await _finish(snapshot)

    assert all(event.type != "continuation_available" for event in events)
    assert [event.type for event in events] == ["error"]
    assert events[0].data["error"] == NO_RESPONSE_GENERATED


async def test_a_turn_with_no_pause_and_no_response_is_still_an_error():
    """The pause branch must not become a way to end a broken turn quietly."""
    events = await _finish(_Snapshot([], next_nodes=("chat_agent",)))

    assert [event.type for event in events] == ["error"]
    assert events[0].data["error"] == NO_RESPONSE_GENERATED
