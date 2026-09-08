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
from typing import Any, TypedDict

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


# ----------------------------------------------------------------------
# resuming the pause, through a real compiled graph
#
# These drive LangGraph's own `interrupt()` against a real checkpointer rather
# than mocking the resume seam. That distinction has already cost this
# repository once: a routing context passed through `config["context"]` was
# silently dropped by every real call and no test noticed, because every test
# mocked the delivery seam.
# ----------------------------------------------------------------------


def _paused_graph():
    """A two-node graph that pauses like the real one and then answers.

    ``pause`` is the real ``make_continuation_pause_node`` wired to LangGraph's
    own ``interrupt``; ``answer`` stands in for the specialist the decision
    routes back to.
    """
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
    from app.ai.workflow.continuation import make_continuation_pause_node
    from app.ai.workflow.contracts import OutcomeProvenance, ResponseOutcome

    outcome = ResponseOutcome(
        agent_id="chat_agent",
        response=AgentResponse(
            agent_type=AgentType.CHAT,
            agent_id="chat_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="partial answer"),
            metadata={"execution_budget": {"exhausted_by": "tool_calls"}},
        ),
        provenance=OutcomeProvenance(output_policy_ids=("public_content",)),
    )

    resumed: dict[str, Any] = {}

    async def enter(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "agent_outcome": outcome,
            "active_agent_id": "chat_agent",
            "execution_epoch": 0,
            "execution_budget": {"exhausted_by": "tool_calls"},
        }

    async def chat_agent(state: dict[str, Any]) -> dict[str, Any]:
        resumed["epoch"] = state.get("execution_epoch")
        resumed["budget"] = state.get("execution_budget")
        return {"response": outcome.response, "execution_phase": "finalizing"}

    async def finalize(state: dict[str, Any]) -> dict[str, Any]:
        return {"response": state.get("response") or outcome.response}

    # A declared schema, not bare ``dict``. LangGraph silently discards a write
    # to a channel the schema does not name ("wrote to unknown channel ...,
    # ignoring it"), so a bare-dict double would have shown the epoch never
    # advancing and blamed the resume for it.
    class _State(TypedDict, total=False):
        agent_outcome: Any
        active_agent_id: str
        execution_epoch: int
        execution_budget: dict[str, Any] | None
        execution_phase: str
        carried_messages: list[Any]
        response: Any

    builder = StateGraph(_State)
    builder.add_node("enter", enter)
    builder.add_node("continuation_pause", make_continuation_pause_node())
    builder.add_node("chat_agent", chat_agent)
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "enter")
    builder.add_edge("enter", "continuation_pause")
    builder.add_edge("chat_agent", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=InMemorySaver()), resumed


def _workflow_over(graph) -> MultiAgentWorkflow:
    """A workflow whose graph is the compiled double above."""
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.checkpointer = SimpleNamespace()
    workflow.graph = graph
    workflow.agents = {"chat_agent": object()}
    workflow.routing_service = None
    workflow.routing_context_builder = None
    workflow._build_graph_config = lambda thread_id: {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 25,
    }
    workflow._build_chat_image_loader = lambda user_id: None
    workflow._tool_end_events_from_node_state = lambda *a, **k: []
    return workflow


async def _pause_a_real_turn():
    from app.ai.workflow.continuation import pending_continuation_payload

    graph, resumed = _paused_graph()
    config = {"configurable": {"thread_id": "t-real"}, "recursion_limit": 25}
    await graph.ainvoke({}, config=config)

    snapshot = await graph.aget_state(config)
    assert pending_continuation_payload(snapshot) is not None, "the graph did not pause"
    return graph, resumed


async def test_a_real_graph_pauses_and_the_stream_offers_a_continuation():
    graph, _ = await _pause_a_real_turn()

    from app.ai.graph import StreamProjectionContext

    events = [
        event
        async for event in _workflow_over(graph)._finish_stream(
            StreamProjectionContext(last_emitted_agent="chat_agent", suppress_tokens=False),
            config={"configurable": {"thread_id": "t-real"}},
            thread_id="t-real",
            conversation_id="conv-1",
        )
    ]

    assert [event.type for event in events] == ["continuation_available"]
    assert events[0].data["validated_content"] == "partial answer"


async def test_continue_resumes_the_exact_checkpoint_with_a_fresh_epoch():
    """The whole point of Continue: same turn, next epoch, no re-route."""
    from app.ai.workflow.continuation import ContinuationResume

    graph, resumed = await _pause_a_real_turn()

    events = [
        event
        async for event in _workflow_over(graph).resume_with_continuation_stream(
            "t-real",
            ContinuationResume(action="continue", continuation_id="c-1", expected_epoch=0),
        )
    ]

    assert resumed["epoch"] == 1, "the resumed epoch did not advance"
    assert resumed["budget"] is None, "the spent budget was carried into the new epoch"
    assert "complete" in [event.type for event in events]
    assert all(event.type != "error" for event in events)


async def test_stop_resumes_into_the_finalizer_without_another_epoch():
    from app.ai.workflow.continuation import ContinuationResume

    graph, resumed = await _pause_a_real_turn()

    events = [
        event
        async for event in _workflow_over(graph).resume_with_continuation_stream(
            "t-real",
            ContinuationResume(action="stop", continuation_id="c-1", expected_epoch=0),
        )
    ]

    assert "epoch" not in resumed, "Stop ran another epoch of the specialist"
    assert "complete" in [event.type for event in events]


async def test_a_continue_for_a_stale_epoch_does_not_run_the_specialist():
    """A fence, not a hint. Running it would open an epoch on top of a later one."""
    from app.ai.workflow.continuation import ContinuationResume

    graph, resumed = await _pause_a_real_turn()

    async for _event in _workflow_over(graph).resume_with_continuation_stream(
        "t-real",
        ContinuationResume(action="continue", continuation_id="c-1", expected_epoch=99),
    ):
        pass

    assert "epoch" not in resumed


async def test_continuing_a_turn_that_is_not_paused_is_a_typed_refusal():
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    from app.ai.workflow.continuation import ContinuationResume

    builder = StateGraph(dict)
    builder.add_node("only", lambda state: {"done": True})
    builder.add_edge(START, "only")
    builder.add_edge("only", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    await graph.ainvoke({}, config={"configurable": {"thread_id": "t-done"}})

    events = [
        event
        async for event in _workflow_over(graph).resume_with_continuation_stream(
            "t-done",
            ContinuationResume(action="continue", continuation_id="c-1", expected_epoch=0),
        )
    ]

    assert [event.type for event in events] == ["error"]
    assert "not paused" in events[0].data["error"]


async def test_continuing_a_turn_waiting_on_a_tool_approval_says_so():
    """The two pauses need distinguishable refusals, not one generic one."""
    from app.ai.workflow.continuation import ContinuationResume

    workflow = _workflow_over(None)

    async def aget_state(_config):
        return _Snapshot([_Task([_approval_pause()])])

    workflow.graph = SimpleNamespace(aget_state=aget_state)

    events = [
        event
        async for event in workflow.resume_with_continuation_stream(
            "t-approval",
            ContinuationResume(action="continue", continuation_id="c-1", expected_epoch=0),
        )
    ]

    assert [event.type for event in events] == ["error"]
    assert "human decision" in events[0].data["error"]


async def test_continuing_without_a_checkpointer_is_a_typed_refusal():
    from app.ai.workflow.continuation import ContinuationResume

    workflow = _workflow_over(None)
    workflow.checkpointer = None

    events = [
        event
        async for event in workflow.resume_with_continuation_stream(
            "t-none",
            ContinuationResume(action="continue", continuation_id="c-1", expected_epoch=0),
        )
    ]

    assert [event.type for event in events] == ["error"]
    assert "Checkpointing" in events[0].data["error"]
