"""The Planning replay regression against the production checkpointer.

``MemorySaver`` and ``AsyncPostgresSaver`` are not interchangeable for this
invariant. The in-memory saver keeps live Python objects, so a "resume" there
can pass while the real one fails on anything that does not round-trip through
the serializer — and the whole fix depends on each worker's result being
*persisted* as its own task write.

So this runs the same scenario as
``tests/test_planning_worker_fanout.py::test_a_paused_worker_does_not_replay_its_completed_sibling``
against PostgreSQL, and additionally discards the graph *and its connection*
between the pause and the resume, because a restarted process is the case the
fan-out exists for.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.ai.checkpoint import _build_checkpoint_serializer
from app.ai.workflow.contracts import WorkerResult
from app.ai.workflow.inventory import AgentDescriptor, RoutingInventory
from app.ai.workflow.planning_execution import (
    PlanningLimits,
    PlanningNodeFactory,
    TodoActionOutcome,
)
from app.ai.workflow.state import WorkflowState, build_checkpoint_thread_id

pytestmark = pytest.mark.selector_event_loop


def _async_url(database_url: str) -> str:
    for prefix in ("postgresql+psycopg2://", "postgresql+psycopg://"):
        if database_url.startswith(prefix):
            return database_url.replace(prefix, "postgresql://", 1)
    return database_url


@pytest.fixture
def database_url() -> str:
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is required for PostgreSQL integration tests")
    return _async_url(url)


@pytest.fixture
def open_saver(database_url: str):
    """Open one production saver, scoped to a single ``async with`` block.

    Each phase of a test opens its own: reusing a connection across the pause
    and the resume would let the resume succeed on in-process state a restarted
    worker would never have. The serializer is the production one, so a
    contract type that fails to round-trip fails here rather than in production.
    """

    @asynccontextmanager
    async def _open():
        async with AsyncPostgresSaver.from_conn_string(
            database_url, serde=_build_checkpoint_serializer()
        ) as saver:
            await saver.setup()
            yield saver

    return _open


def _inventory() -> RoutingInventory:
    return RoutingInventory.from_descriptors(
        AgentDescriptor(
            agent_id=agent_id,
            display_name=agent_id,
            capability_description="",
            enabled=True,
            attached=True,
            kind="base",
        )
        for agent_id in ("chat_agent", "planning_agent")
    )


def _dispatch_call(*task_ids: str, call_id: str = "call-dispatch-1") -> dict:
    return {
        "name": "dispatch_subagents",
        "id": call_id,
        "args": {
            "tasks": [
                {"task_id": task_id, "objective": f"do {task_id}", "agent_id": "chat_agent"}
                for task_id in task_ids
            ]
        },
    }


def _model_response(content: str = "", tool_calls=()) -> Any:
    return SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=[dict(c) for c in tool_calls]),
        metadata={},
    )


class ScriptedPlanningModel:
    """Indexes its script by the state's own AI turns, so restarts replay."""

    def __init__(self, responses):
        self._responses = list(responses)

    async def __call__(self, state):
        turn = sum(
            1 for message in (state.get("messages") or []) if getattr(message, "type", None) == "ai"
        )
        return self._responses[min(turn, len(self._responses) - 1)]


class RecordingWorkerRuntime:
    """Records real executions into a list that outlives the graph."""

    def __init__(self, side_effects: list[str], *, pause_task_ids=()):
        self.side_effects = side_effects
        self._pause = set(pause_task_ids)

    async def run(self, task, state, writer=None) -> WorkerResult:
        if task.task_id in self._pause:
            interrupt({"task_id": task.task_id, "tool_call_id": f"approve-{task.task_id}"})
        self.side_effects.append(task.task_id)
        return WorkerResult(
            dispatch_id=task.dispatch_id,
            task_id=task.task_id,
            position=task.position,
            agent_id=task.agent_id,
            status="completed",
            content=f"result for {task.task_id}",
        )


async def _noop_todo_actions(state, calls) -> TodoActionOutcome:  # pragma: no cover
    return TodoActionOutcome(todos=list(state.get("todos") or []))


def _build_graph(saver, side_effects: list[str], *, pause_task_ids=()):
    factory = PlanningNodeFactory(
        call_model=ScriptedPlanningModel(
            [
                _model_response(tool_calls=(_dispatch_call("w1", "w2"),)),
                _model_response(content="both pieces are done"),
            ]
        ),
        worker_runtime=RecordingWorkerRuntime(side_effects, pause_task_ids=pause_task_ids),
        limits=PlanningLimits(
            max_tasks=8,
            max_concurrency=4,
            max_dispatch_waves=2,
            objective_max_chars=4000,
            parent_context_max_chars=12000,
        ),
        inventory_for=lambda state: _inventory(),
        resolve_allowed_tools=lambda agent_id, state: (),
        apply_todo_actions=_noop_todo_actions,
    )

    graph = StateGraph(WorkflowState)

    async def finalize(state):
        return {"execution_phase": "completed"}

    async def validate_output(state):
        return Command(goto="finalize")

    async def resolve_transition(state):  # pragma: no cover - not reached here
        return Command(goto="finalize")

    for name, node, destinations in factory.descriptors():
        graph.add_node(name, node, destinations=destinations)
    graph.add_node("validate_output", validate_output, destinations=("finalize",))
    graph.add_node("resolve_transition", resolve_transition, destinations=("finalize",))
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "planning_model")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=saver)


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}


def _turn() -> dict:
    return {
        "messages": [],
        "conversation_id": "conversation-1",
        "user_id": "user-1",
        "active_agent_id": "planning_agent",
        "todos": [],
    }


def _thread(suffix: str) -> str:
    """A thread ID unique per process, so reruns never collide."""
    return build_checkpoint_thread_id("conversation-1", f"turn-{os.getpid()}-{suffix}")


async def test_a_discarded_workflow_resumes_without_replaying_a_finished_worker(
    open_saver,
) -> None:
    """``w1, w2`` across a process boundary — never ``w1, w1, w2``."""
    thread_id = _thread("replay")
    side_effects: list[str] = []

    async with open_saver() as saver:
        workflow_a = _build_graph(saver, side_effects, pause_task_ids={"w2"})
        await workflow_a.ainvoke(_turn(), config=_config(thread_id))

        assert side_effects == ["w1"]
        snapshot = await workflow_a.aget_state(_config(thread_id))
        assert len(snapshot.interrupts) == 1

    # Workflow A and its connection are both gone past this point.
    async with open_saver() as saver:
        workflow_b = _build_graph(saver, side_effects)
        state = await workflow_b.ainvoke(
            Command(resume={"decision": "accept"}), config=_config(thread_id)
        )

    assert side_effects == ["w1", "w2"]

    identities = {
        (result.dispatch_id, result.task_id) for result in (state.get("worker_results") or [])
    }
    assert len(identities) == 2
    assert {task_id for _, task_id in identities} == {"w1", "w2"}

    paired = [
        message
        for message in state["messages"]
        if isinstance(message, ToolMessage) and message.tool_call_id == "call-dispatch-1"
    ]
    assert len(paired) == 1


async def test_worker_results_round_trip_through_the_production_serializer(
    open_saver,
) -> None:
    """A ``WorkerResult`` that comes back as a plain dict is not validated.

    The allowlist in ``app/ai/checkpoint.py`` is what keeps it a typed object,
    and only a real saver exercises it.
    """
    thread_id = _thread("serde")
    side_effects: list[str] = []

    async with open_saver() as saver:
        graph = _build_graph(saver, side_effects, pause_task_ids={"w2"})
        await graph.ainvoke(_turn(), config=_config(thread_id))
        snapshot = await graph.aget_state(_config(thread_id))

    restored = snapshot.values.get("worker_results") or []

    assert restored, "the completed worker's result was not checkpointed"
    assert all(isinstance(result, WorkerResult) for result in restored)
    assert restored[0].position == 0


async def test_the_dispatch_itself_survives_the_checkpoint(open_saver) -> None:
    """Collection needs the dispatch after a restart to pair its result."""
    from app.ai.workflow.contracts import PlanningDispatch

    thread_id = _thread("dispatch")
    side_effects: list[str] = []

    async with open_saver() as saver:
        graph = _build_graph(saver, side_effects, pause_task_ids={"w2"})
        await graph.ainvoke(_turn(), config=_config(thread_id))
        snapshot = await graph.aget_state(_config(thread_id))

    dispatch = snapshot.values.get("planning_dispatch")

    assert isinstance(dispatch, PlanningDispatch)
    assert dispatch.tool_call_id == "call-dispatch-1"
    assert [task.task_id for task in dispatch.tasks] == ["w1", "w2"]
