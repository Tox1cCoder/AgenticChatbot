"""One epoch number, from the graph through the row and back into the fence.

Three components name the epoch a paused turn is in: the graph state the pause
node fences against, the payload it hands the client, and the ``generations``
row a Continue leases. They used to disagree. ``execution_epoch`` was not a
field on ``WorkflowState``, so LangGraph dropped the pause node's update and the
graph sat at epoch 0 forever, while the row advanced on every human Continue.
The second Continue in a turn was then fenced against an epoch the graph had
never reached, and the turn finalized instead of resuming.

The graph is the source of truth. Its epoch is checkpointed, the pause payload
carries it, and the row records it when the pause is persisted -- so an epoch
the graph rolled on its own, which the row never saw, is still the one the next
Continue is fenced against.

These tests run the real pause and validation nodes on a graph compiled over
the real ``WorkflowState`` (a bare-dict schema would keep every key and hide
the drop), and the real ``GenerationControlService`` over the faithful
repository double.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from app.ai.workflow.continuation import make_continuation_pause_node, pending_continuation_payload
from app.ai.workflow.contracts import OutcomeProvenance, ResponseOutcome, TurnIdentity
from app.ai.workflow.execution_budget import (
    ExecutionBudgetAccountant,
    ExecutionBudgetLimits,
    ExecutionBudgetState,
)
from app.ai.workflow.finalization import make_validate_output_node
from app.ai.workflow.state import WorkflowState
from app.schemas.generation import CreateGeneration, MarkContinuable
from app.services.event_streaming.events import make_event
from app.services.generation_registry import get_generation_registry
from app.services.message_service import MessageService
from tests.generation_control_support import CONVERSATION_ID, USER_ID, build_control_service

TURN_ID = "turn-1"
THREAD_ID = f"routing-v2:{CONVERSATION_ID}:{TURN_ID}"


@pytest.fixture(autouse=True)
def _clean_registry():
    registry = get_generation_registry()
    for key in list(registry._store):  # noqa: SLF001 - test isolation
        registry._store.pop(key, None)  # noqa: SLF001
    yield
    for key in list(registry._store):  # noqa: SLF001
        registry._store.pop(key, None)  # noqa: SLF001


# ----------------------------------------------------------------------
# a turn graph over the real state schema
# ----------------------------------------------------------------------


def _settings(total_epochs: int) -> SimpleNamespace:
    return SimpleNamespace(
        generation_soft_model_calls_per_epoch=3,
        generation_hard_model_calls_per_epoch=5,
        generation_soft_tool_calls_per_epoch=2,
        generation_hard_tool_calls_per_epoch=4,
        generation_total_epochs_per_turn=total_epochs,
    )


class _AcceptingValidator:
    async def validate(self, outcome, state):
        return outcome


class _ExhaustingSpecialist:
    """Spends its whole epoch on every run, the way a long plan does.

    It seeds its accountant from ``state["execution_budget"]`` -- the same read
    Planning makes, and the one ``_specialist_request_for`` hands the
    ``create_agent`` specialists through ``extras``. ``hard_limit_on`` names
    the runs that end on the framework limit, which is what makes the pause
    ask a human even when the turn rolls its own epochs.
    """

    def __init__(self, limits: ExecutionBudgetLimits, *, hard_limit_on: tuple[int, ...] = ()):
        self._limits = limits
        self._hard_limit_on = hard_limit_on
        self.runs: list[dict[str, Any]] = []

    async def __call__(self, state: dict[str, Any]) -> Command:
        carried = state.get("execution_budget")
        accountant = ExecutionBudgetAccountant(
            limits=self._limits,
            state=ExecutionBudgetState.model_validate(carried) if carried else None,
        )
        while accountant.note_tool_call().allowed:
            pass
        run = len(self.runs) + 1
        if run in self._hard_limit_on:
            accountant.note_hard_limit()
        self.runs.append(
            {
                "epoch": state.get("execution_epoch"),
                "epochs_used": accountant.state.epochs_used,
                "turn_tool_calls": accountant.state.turn_tool_calls,
            }
        )
        outcome = ResponseOutcome(
            agent_id="chat_agent",
            response=AgentResponse(
                agent_type=AgentType.CHAT,
                agent_id="chat_agent",
                message=AgentMessage(role=MessageRole.ASSISTANT, content=f"partial {run}"),
                metadata={},
            ),
            provenance=OutcomeProvenance(output_policy_ids=("public_content",)),
        )
        return Command(
            update={
                "agent_outcome": outcome,
                "execution_budget": accountant.state.model_dump(mode="json"),
                "execution_phase": "validating",
            },
            goto="validate_output",
        )


def _turn_graph(
    *, auto_continue: bool, total_epochs: int = 5, hard_limit_on: tuple[int, ...] = ()
):
    settings = _settings(total_epochs)
    specialist = _ExhaustingSpecialist(
        ExecutionBudgetLimits.from_settings(settings), hard_limit_on=hard_limit_on
    )

    async def route(state: dict[str, Any]) -> Command:
        return Command(
            update={"active_agent_id": "chat_agent", "execution_phase": "executing"},
            goto="chat_agent",
        )

    async def finalize(state: dict[str, Any]) -> dict[str, Any]:
        return {"response": state["agent_outcome"].response, "execution_phase": "finalizing"}

    builder = StateGraph(WorkflowState)
    builder.add_node("route", route)
    builder.add_node("chat_agent", specialist)
    builder.add_node(
        "validate_output", make_validate_output_node(_AcceptingValidator(), settings=settings)
    )
    builder.add_node(
        "continuation_pause", make_continuation_pause_node(auto_continue=auto_continue)
    )
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "route")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=InMemorySaver()), specialist


def _config() -> dict[str, Any]:
    return {"configurable": {"thread_id": THREAD_ID}, "recursion_limit": 60}


def _turn_input() -> dict[str, Any]:
    return {
        "messages": [],
        "turn_identity": TurnIdentity(
            request_id=TURN_ID, turn_id=TURN_ID, checkpoint_thread_id=THREAD_ID
        ),
    }


async def _pause_on_graph(graph):
    return pending_continuation_payload(await graph.aget_state(_config()))


async def _resume(graph, *, expected_epoch: int, action: str = "continue") -> None:
    await graph.ainvoke(
        Command(
            resume={"action": action, "continuation_id": "c-1", "expected_epoch": expected_epoch}
        ),
        _config(),
    )


# ----------------------------------------------------------------------
# the graph's own epoch
# ----------------------------------------------------------------------


async def test_the_graph_keeps_the_epoch_it_advanced():
    """The update the pause node writes must survive the state schema."""
    graph, specialist = _turn_graph(auto_continue=False)
    await graph.ainvoke(_turn_input(), _config())
    first = await _pause_on_graph(graph)

    await _resume(graph, expected_epoch=first.execution_epoch)
    second = await _pause_on_graph(graph)

    assert [run["epoch"] for run in specialist.runs] == [None, 1]
    assert (first.execution_epoch, second.execution_epoch) == (0, 1)


async def test_the_pause_names_the_turn_it_belongs_to():
    """The logical turn keys the research accounting a Continue restores."""
    graph, _ = _turn_graph(auto_continue=False)
    await graph.ainvoke(_turn_input(), _config())

    payload = await _pause_on_graph(graph)

    assert payload.logical_turn_id == TURN_ID


async def test_a_stale_continue_is_still_refused_after_the_epoch_moved():
    """The fence exists so an old Continue cannot open an epoch on a moved turn."""
    graph, specialist = _turn_graph(auto_continue=False)
    await graph.ainvoke(_turn_input(), _config())
    await _resume(graph, expected_epoch=0)
    assert (await _pause_on_graph(graph)).execution_epoch == 1

    await _resume(graph, expected_epoch=0)

    assert len(specialist.runs) == 2, "a Continue for epoch 0 ran a third epoch"
    assert await _pause_on_graph(graph) is None
    snapshot = await graph.aget_state(_config())
    assert snapshot.values["execution_phase"] == "finalizing"


# ----------------------------------------------------------------------
# the per-turn epoch cap
# ----------------------------------------------------------------------


async def test_a_turn_that_keeps_needing_more_stops_at_its_epoch_cap():
    """Auto-continue is bounded by the epoch budget, not the recursion limit."""
    graph, specialist = _turn_graph(auto_continue=True, total_epochs=2)

    result = await graph.ainvoke(_turn_input(), _config())

    assert len(specialist.runs) == 2, "the turn ran past generation_total_epochs_per_turn"
    assert await _pause_on_graph(graph) is None
    assert result["response"].message.content == "partial 2"


async def test_the_next_epoch_gets_its_room_back_and_keeps_the_turn_totals():
    graph, specialist = _turn_graph(auto_continue=True, total_epochs=3)

    await graph.ainvoke(_turn_input(), _config())

    assert [run["epochs_used"] for run in specialist.runs] == [1, 2, 3]
    # Two tool calls fit in each epoch; the turn total keeps counting.
    assert [run["turn_tool_calls"] for run in specialist.runs] == [2, 4, 6]


async def test_the_rolled_budget_is_a_fresh_epoch_not_an_empty_quota():
    graph, _ = _turn_graph(auto_continue=False)
    await graph.ainvoke(_turn_input(), _config())

    await _resume(graph, expected_epoch=0)
    budget = (await _pause_on_graph(graph)).budget

    assert budget["execution_epoch"] == 1
    assert budget["epochs_used"] == 2


# ----------------------------------------------------------------------
# the whole chain: graph -> pause persisted on the row -> lease -> fence
# ----------------------------------------------------------------------


class _GraphResumer:
    """The ``ai_service`` seam, resuming the real compiled graph."""

    def __init__(self, graph) -> None:
        self._graph = graph
        self.expected_epochs: list[int] = []

    async def resume_generation_control_stream(self, *, action, expected_epoch, **_kwargs):
        self.expected_epochs.append(expected_epoch)
        await _resume(self._graph, expected_epoch=expected_epoch, action=action)
        async for event in _events_after_run(self._graph):
            yield event


async def _events_after_run(graph):
    payload = await _pause_on_graph(graph)
    if payload is not None:
        yield make_event(
            "continuation_available", sequence=1, data=payload.model_dump(mode="json")
        )
        return
    yield make_event("complete", sequence=1, data={"response": None})


def _service(control) -> MessageService:
    service = MessageService.__new__(MessageService)
    service.generation_control_service = control
    service._turn_coordinator = None

    async def avalidate(user_id, conversation_id):
        return None

    service.conversation_validation_utils = SimpleNamespace(
        avalidate_conversation_access=avalidate,
        validate_conversation_access=lambda user_id, conversation_id: None,
    )

    async def acreate_bot_response_message(*, conversation_id, content, metadata, message_id=None):
        return SimpleNamespace(
            id=message_id or uuid4(),
            model_dump=lambda mode="python": {"content": content, "metadata": dict(metadata)},
        )

    service._acreate_bot_response_message = acreate_bot_response_message
    return service


async def _running_generation(service, control):
    started = await control.start_generation(
        CreateGeneration(
            conversation_id=CONVERSATION_ID,
            user_id=USER_ID,
            logical_turn_id=TURN_ID,
            checkpoint_thread_id=THREAD_ID,
            active_agent_id="chat_agent",
        )
    )
    return await service._amark_generation_running(started, user_id=USER_ID)


async def _first_pause_persisted(graph, service, control):
    """Run the turn to its first pause and persist it the way the stream does."""
    running = await _running_generation(service, control)
    await graph.ainvoke(_turn_input(), _config())
    events = [event async for event in _events_after_run(graph)]
    await _publish(service, events[0], running)


async def _human_continue(service, control) -> list:
    offered = await control.find_by_logical_turn(
        logical_turn_id=TURN_ID, user_id=USER_ID, conversation_id=CONVERSATION_ID
    )
    assert offered.continuation_available, "no Continue was offered for this pause"
    return [
        event
        async for event in service.continue_message_generation_stream(
            generation_id=offered.generation_id,
            continuation_id=offered.continuation_id,
            conversation_id=CONVERSATION_ID,
            user_id=USER_ID,
            idempotency_key=f"continue-{uuid4()}",
            expected_version=offered.version,
        )
    ]


async def test_two_human_continues_in_one_turn_both_resume():
    graph, specialist = _turn_graph(auto_continue=False)
    control = build_control_service()
    service = _service(control)
    service.ai_service = resumer = _GraphResumer(graph)
    await _first_pause_persisted(graph, service, control)

    first = await _human_continue(service, control)
    second = await _human_continue(service, control)

    assert resumer.expected_epochs == [0, 1]
    assert len(specialist.runs) == 3, "the second Continue was refused"
    assert "continuation_available" in [event.type for event in first]
    assert "continuation_available" in [event.type for event in second]


async def test_a_human_continue_after_an_auto_rolled_epoch_resumes():
    """The row never saw the rolled epoch; the pause it records must name it."""
    graph, specialist = _turn_graph(auto_continue=True, hard_limit_on=(2,))
    control = build_control_service()
    service = _service(control)
    service.ai_service = resumer = _GraphResumer(graph)
    await _first_pause_persisted(graph, service, control)
    assert len(specialist.runs) == 2, "epoch 0 should have rolled over without asking"

    await _human_continue(service, control)

    assert resumer.expected_epochs == [1]
    assert len(specialist.runs) >= 3, "the Continue after an auto-rolled epoch was refused"
    assert specialist.runs[2]["epoch"] == 2


# ----------------------------------------------------------------------
# the row records the epoch the graph paused in
# ----------------------------------------------------------------------


async def test_the_row_records_the_epoch_the_graph_paused_in():
    control = build_control_service()
    started = await control.start_generation(
        CreateGeneration(
            conversation_id=CONVERSATION_ID,
            user_id=USER_ID,
            logical_turn_id=TURN_ID,
            checkpoint_thread_id=THREAD_ID,
        )
    )
    running = await control.mark_running(
        generation_id=started.generation_id,
        user_id=USER_ID,
        conversation_id=CONVERSATION_ID,
        expected_version=started.version,
    )
    offered = await control.mark_continuable(
        MarkContinuable(
            generation_id=running.generation_id,
            conversation_id=CONVERSATION_ID,
            user_id=USER_ID,
            expected_version=running.version,
            assistant_message_id=uuid4(),
            execution_epoch=3,
        )
    )

    assert offered.execution_epoch == 3


def _pause_event(*, epoch: int = 0, logical_turn_id: str = TURN_ID):
    return make_event(
        "continuation_available",
        sequence=1,
        data={
            "type": "execution_budget_exhausted",
            "generation_id": "",
            "logical_turn_id": logical_turn_id,
            "execution_epoch": epoch,
            "active_agent_id": "chat_agent",
            "validated_content": "partial",
            "budget": {"exhausted_by": "tool_calls"},
        },
    )


async def _publish(service, event, generation) -> None:
    async for _ in service._apublish_continuation_pause(
        event,
        generation=generation,
        conversation_id=CONVERSATION_ID,
        user_id=USER_ID,
        bot_message_id=uuid4(),
        sanitized_persona=None,
        workflow_request=None,
        inflight=SimpleNamespace(active_agent_id="chat_agent"),
        tool_artifacts=None,
        next_sequence=iter(range(1, 10)).__next__,
    ):
        pass


async def test_the_pause_hands_the_graph_epoch_to_the_row():
    service = _service(build_control_service())
    service._amark_generation_continuable = AsyncMock(return_value=None)
    generation = SimpleNamespace(generation_id=uuid4(), logical_turn_id=TURN_ID)

    await _publish(service, _pause_event(epoch=4), generation)

    assert service._amark_generation_continuable.await_args.kwargs["execution_epoch"] == 4


@pytest.fixture
def _clean_research_budget():
    from app.ai.research_budget import reset_research_budget

    reset_research_budget(logical_turn_id=TURN_ID, conversation_id=str(CONVERSATION_ID))
    yield
    reset_research_budget(logical_turn_id=TURN_ID, conversation_id=str(CONVERSATION_ID))


async def test_the_pause_snapshots_research_under_the_rows_turn(_clean_research_budget):
    """The Continue side restores under the row's turn id; the pause must save there.

    A payload naming no turn (every pause did, before the graph read its turn
    identity) snapshotted the conversation-scoped bucket instead, which is
    empty -- so the next epoch got a fresh research quota.
    """
    from app.ai.research_budget import get_research_budget

    control = build_control_service()
    service = _service(control)
    running = await _running_generation(service, control)
    budget = get_research_budget(logical_turn_id=TURN_ID, conversation_id=str(CONVERSATION_ID))
    budget.record_search("population of vietnam", "…")

    await _publish(service, _pause_event(logical_turn_id=""), running)

    row = control._test_repository.rows[running.generation_id]
    assert row["research_accounting"] is not None, "the turn's accounting was not persisted"
    assert row["research_accounting"]["searched"]


# ----------------------------------------------------------------------
# the budget reaches the specialist that runs the next epoch
# ----------------------------------------------------------------------


def _request_workflow():
    from app.ai.graph import MultiAgentWorkflow

    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow._get_conversation_history = AsyncMock(return_value=[])
    workflow.agents = {"chat_agent": object()}
    workflow.chat_agent = SimpleNamespace(_convert_history_to_langchain_messages=lambda h: [])
    return workflow


def _rolled_budget() -> dict[str, Any]:
    return ExecutionBudgetState(
        execution_epoch=1, epochs_used=2, turn_tool_calls=9
    ).model_dump(mode="json")


async def test_a_specialist_request_carries_the_epoch_budget():
    from langchain_core.messages import HumanMessage

    state = {
        "active_agent_id": "chat_agent",
        "messages": [HumanMessage(content="question")],
        "execution_budget": _rolled_budget(),
    }

    request = await _request_workflow()._specialist_request_for("chat_agent", state)

    assert request.extras["execution_budget"] == _rolled_budget()


async def test_a_first_epoch_request_carries_no_budget():
    from langchain_core.messages import HumanMessage

    state = {"active_agent_id": "chat_agent", "messages": [HumanMessage(content="question")]}

    request = await _request_workflow()._specialist_request_for("chat_agent", state)

    assert "execution_budget" not in request.extras


async def test_the_rag_request_carries_the_epoch_budget():
    from langchain_core.messages import HumanMessage

    from app.ai.workflow.specialists import SpecialistRequest

    captured: dict[str, Any] = {}

    class _Stop(Exception):
        pass

    async def ainvoke(request):
        captured["request"] = request
        raise _Stop

    workflow = _request_workflow()
    workflow.rag_execution_graph = SimpleNamespace(ainvoke=ainvoke)
    state = {"messages": [HumanMessage(content="question")]}
    request = SpecialistRequest(
        agent_id="rag_agent",
        conversation_id=str(CONVERSATION_ID),
        user_id=str(USER_ID),
        device_id=None,
        persona=None,
        model_request=None,
        messages=[],
        extras={"execution_budget": _rolled_budget()},
    )

    with pytest.raises(_Stop):
        await workflow._invoke_rag_specialist(request, state)

    assert captured["request"].execution_budget == _rolled_budget()
