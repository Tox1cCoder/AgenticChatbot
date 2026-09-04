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

The workers here also perform a real durable mutation through
``ToolExecutionReceiptService``. Counting side effects in a Python list proves
the *graph* did not replay a node; only the receipt rows prove the *provider*
was not called twice, and that is the property an operator cares about. The two
can disagree: a node that re-runs and hits a ``completed`` receipt is correct,
and a node that runs once against a lost reservation is not.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Iterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from sqlalchemy import create_engine, delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.ai.checkpoint import _build_checkpoint_serializer
from app.ai.hitl_config import pending_interrupt_payload
from app.ai.workflow.contracts import WorkerResult
from app.ai.workflow.inventory import AgentDescriptor, RoutingInventory
from app.ai.workflow.planning_execution import (
    PlanningLimits,
    PlanningNodeFactory,
    TodoActionOutcome,
)
from app.ai.workflow.state import WorkflowState, build_checkpoint_thread_id
from app.models.base import Base
from app.models.conversation import Conversation
from app.models.tool_execution_receipt import ReceiptStatus, ToolExecutionReceipt
from app.models.user import User
from app.repositories.tool_execution_receipt import ToolExecutionReceiptRepository
from app.services.tool_execution_receipt_service import (
    MutationExecutionScope,
    NormalizedToolResult,
    ToolExecutionReceiptService,
)

pytestmark = pytest.mark.selector_event_loop


def _async_url(database_url: str) -> str:
    """The bare libpq URL ``AsyncPostgresSaver`` expects."""
    for prefix in ("postgresql+psycopg2://", "postgresql+psycopg://"):
        if database_url.startswith(prefix):
            return database_url.replace(prefix, "postgresql://", 1)
    return database_url


def _orm_url(database_url: str) -> str:
    """The async SQLAlchemy URL the repository expects."""
    for prefix in ("postgresql+psycopg2://", "postgresql://"):
        if database_url.startswith(prefix):
            return database_url.replace(prefix, "postgresql+psycopg://", 1)
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


# ----------------------------------------------------------------------
# durable receipts for the workers' mutations
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def engines():
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is required for PostgreSQL integration tests")
    engine = create_engine(database_url)
    Base.metadata.create_all(
        engine,
        tables=[User.__table__, Conversation.__table__, ToolExecutionReceipt.__table__],
    )
    async_engine = create_async_engine(_orm_url(database_url))
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    async_factory = async_sessionmaker(bind=async_engine, expire_on_commit=False)
    try:
        yield factory, async_factory
    finally:
        engine.dispose()
        if sys.platform == "win32":
            asyncio.run(async_engine.dispose(), loop_factory=asyncio.SelectorEventLoop)
        else:
            asyncio.run(async_engine.dispose())


class Owner(SimpleNamespace):
    """One seeded user/conversation plus the repository scoped to it."""

    user_id: UUID
    conversation_id: UUID
    repository: ToolExecutionReceiptRepository
    session_factory: Any


@pytest.fixture()
def owner(engines) -> Iterator[Owner]:
    session_factory, async_session_factory = engines
    user_id = uuid4()
    conversation_id = uuid4()
    with session_factory.begin() as session:
        session.add(
            User(
                id=user_id,
                username=f"owner-{user_id}",
                email=f"{user_id}@example.test",
                password_hash="test",
            )
        )
        session.add(Conversation(id=conversation_id, owner_id=user_id, title="resume test"))

    yield Owner(
        user_id=user_id,
        conversation_id=conversation_id,
        repository=ToolExecutionReceiptRepository(
            session_factory=session_factory, async_session_factory=async_session_factory
        ),
        session_factory=session_factory,
    )

    with session_factory.begin() as session:
        session.execute(delete(ToolExecutionReceipt).where(ToolExecutionReceipt.user_id == user_id))
        session.execute(delete(Conversation).where(Conversation.id == conversation_id))
        session.execute(delete(User).where(User.id == user_id))


def _receipt_rows(owner: Owner) -> list[ToolExecutionReceipt]:
    with owner.session_factory() as session:
        return list(
            session.execute(
                select(ToolExecutionReceipt)
                .where(ToolExecutionReceipt.user_id == owner.user_id)
                .order_by(ToolExecutionReceipt.task_id)
            )
            .scalars()
            .all()
        )


# ----------------------------------------------------------------------
# the scripted planning turn
# ----------------------------------------------------------------------


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
    """Indexes its script by the state's own AI turns, so restarts replay.

    ``calls`` counts what *this instance* was asked to decide. A workflow
    rebuilt after a restart gets a fresh instance, so the counter answers the
    question the restart is really about: was the dispatch decision made again?
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    async def __call__(self, state):
        self.calls += 1
        turn = sum(
            1 for message in (state.get("messages") or []) if getattr(message, "type", None) == "ai"
        )
        return self._responses[min(turn, len(self._responses) - 1)]


def _approval_request(task_id: str) -> dict:
    """The pause shape ``pending_interrupt_payload`` reads.

    Per-worker ``device_id`` provenance is the field that used to be lost: two
    workers paused at once were merged into one metadata dict and the second
    overwrote the first.
    """
    return {
        "action_requests": [
            {
                "action": "send_email",
                "args": {"to": f"{task_id}@example.test"},
                "tool_call_id": f"approve-{task_id}",
            }
        ],
        "metadata": {"device_id": f"device-{task_id}"},
    }


class RecordingWorkerRuntime:
    """Records real executions into lists that outlive the graph.

    With a receipt service attached, each task also performs one durable,
    non-idempotent mutation, so ``provider_calls`` is what the provider
    actually saw rather than what the graph intended.
    """

    def __init__(
        self,
        side_effects: list[str],
        *,
        pause_task_ids=(),
        receipts: ToolExecutionReceiptService | None = None,
        scope_template: dict[str, Any] | None = None,
        provider_calls: list[str] | None = None,
    ):
        self.side_effects = side_effects
        self.provider_calls = provider_calls if provider_calls is not None else []
        self._pause = set(pause_task_ids)
        self._receipts = receipts
        self._scope_template = dict(scope_template or {})

    async def run(self, task, state, writer=None) -> WorkerResult:
        if task.task_id in self._pause:
            interrupt(_approval_request(task.task_id))
        if self._receipts is not None:
            await self._mutate(task)
        self.side_effects.append(task.task_id)
        return WorkerResult(
            dispatch_id=task.dispatch_id,
            task_id=task.task_id,
            position=task.position,
            agent_id=task.agent_id,
            status="completed",
            content=f"result for {task.task_id}",
        )

    async def _mutate(self, task) -> None:
        scope = MutationExecutionScope(
            dispatch_id=task.dispatch_id,
            task_id=task.task_id,
            tool_call_id=f"call-{task.task_id}",
            tool_id="mcp::send_email",
            **self._scope_template,
        )

        async def _invoke() -> NormalizedToolResult:
            self.provider_calls.append(task.task_id)
            return NormalizedToolResult(
                content=f"sent for {task.task_id}",
                provider_receipt_id=f"provider-{task.task_id}",
            )

        await self._receipts.execute_mutation(scope, _invoke)


async def _noop_todo_actions(state, calls) -> TodoActionOutcome:  # pragma: no cover
    return TodoActionOutcome(todos=list(state.get("todos") or []))


def _build_graph(saver, runtime, model=None):
    factory = PlanningNodeFactory(
        call_model=model
        or ScriptedPlanningModel(
            [
                _model_response(tool_calls=(_dispatch_call("w1", "w2"),)),
                _model_response(content="both pieces are done"),
            ]
        ),
        worker_runtime=runtime,
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


def _turn(conversation_id: str = "conversation-1", user_id: str = "user-1") -> dict:
    return {
        "messages": [],
        "conversation_id": conversation_id,
        "user_id": user_id,
        "active_agent_id": "planning_agent",
        "todos": [],
    }


def _thread(suffix: str, conversation_id: str = "conversation-1") -> str:
    """A thread ID unique per process, so reruns never collide."""
    return build_checkpoint_thread_id(conversation_id, f"turn-{os.getpid()}-{suffix}")


def _live_interrupt_ids(snapshot) -> tuple[str, ...]:
    payload = pending_interrupt_payload(snapshot)
    return payload.interrupt_ids if payload else ()


# ----------------------------------------------------------------------
# the restart scenario
# ----------------------------------------------------------------------


async def test_a_discarded_workflow_resumes_without_replaying_a_finished_worker(
    open_saver, owner: Owner
) -> None:
    """``w1, w2`` across a process boundary — never ``w1, w1, w2``.

    Approval names the exact interrupt ID rather than resuming the thread
    blindly, because that is what a resumed HITL decision carries and because a
    blanket resume would pass even if the pending set had been rebuilt wrongly.
    """
    conversation_id = str(owner.conversation_id)
    thread_id = _thread("replay", conversation_id)
    side_effects: list[str] = []
    provider_calls: list[str] = []
    receipts = ToolExecutionReceiptService(repository=owner.repository)
    scope_template = {
        "thread_id": thread_id,
        "user_id": owner.user_id,
        "conversation_id": owner.conversation_id,
        "turn_id": "turn-1",
    }

    def _runtime(*, pause_task_ids=()):
        return RecordingWorkerRuntime(
            side_effects,
            pause_task_ids=pause_task_ids,
            receipts=receipts,
            scope_template=scope_template,
            provider_calls=provider_calls,
        )

    def _model() -> ScriptedPlanningModel:
        return ScriptedPlanningModel(
            [
                _model_response(tool_calls=(_dispatch_call("w1", "w2"),)),
                _model_response(content="both pieces are done"),
            ]
        )

    model_a = _model()
    async with open_saver() as saver:
        workflow_a = _build_graph(saver, _runtime(pause_task_ids={"w2"}), model_a)
        await workflow_a.ainvoke(
            _turn(conversation_id, str(owner.user_id)), config=_config(thread_id)
        )

        assert side_effects == ["w1"]
        snapshot = await workflow_a.aget_state(_config(thread_id))
        interrupt_ids = _live_interrupt_ids(snapshot)
        assert len(interrupt_ids) == 1

    # Workflow A and its connection are both gone past this point.
    model_b = _model()
    async with open_saver() as saver:
        workflow_b = _build_graph(saver, _runtime(), model_b)
        state = await workflow_b.ainvoke(
            Command(resume={interrupt_ids[0]: {"decision": "accept"}}),
            config=_config(thread_id),
        )
        assert _live_interrupt_ids(await workflow_b.aget_state(_config(thread_id))) == ()

    assert side_effects == ["w1", "w2"]
    assert provider_calls == ["w1", "w2"], "the restart re-entered a completed provider call"

    identities = {
        (result.dispatch_id, result.task_id) for result in (state.get("worker_results") or [])
    }
    assert len(identities) == 2
    assert {task_id for _, task_id in identities} == {"w1", "w2"}

    rows = _receipt_rows(owner)
    assert [row.task_id for row in rows] == ["w1", "w2"]
    assert {row.status for row in rows} == {ReceiptStatus.COMPLETED}
    assert len({row.execution_key for row in rows}) == 2

    paired = [
        message
        for message in state["messages"]
        if isinstance(message, ToolMessage) and message.tool_call_id == "call-dispatch-1"
    ]
    assert len(paired) == 1

    # The dispatch decision was made once, by workflow A. Workflow B only
    # synthesized: a resume that re-asked the model would have routed again.
    assert model_a.calls == 1
    assert model_b.calls == 1


async def test_two_paused_workers_are_decided_one_at_a_time_across_a_restart(
    open_saver, owner: Owner
) -> None:
    """Partial approval survives a process boundary with provenance intact.

    The undecided worker must still be waiting after the first decision — not
    replayed as a rejection nobody made, and not carrying the other worker's
    device provenance.
    """
    conversation_id = str(owner.conversation_id)
    thread_id = _thread("partial", conversation_id)
    side_effects: list[str] = []
    provider_calls: list[str] = []
    receipts = ToolExecutionReceiptService(repository=owner.repository)
    scope_template = {
        "thread_id": thread_id,
        "user_id": owner.user_id,
        "conversation_id": owner.conversation_id,
        "turn_id": "turn-1",
    }

    def _runtime(pause_task_ids):
        return RecordingWorkerRuntime(
            side_effects,
            pause_task_ids=pause_task_ids,
            receipts=receipts,
            scope_template=scope_template,
            provider_calls=provider_calls,
        )

    async with open_saver() as saver:
        graph = _build_graph(saver, _runtime({"w1", "w2"}))
        await graph.ainvoke(_turn(conversation_id, str(owner.user_id)), config=_config(thread_id))
        payload = pending_interrupt_payload(await graph.aget_state(_config(thread_id)))

    assert payload is not None
    assert len(payload.interrupt_ids) == 2
    assert side_effects == []
    provenance = {
        call_id: metadata.get("device_id")
        for call_id, metadata in payload.metadata_by_tool_call_id.items()
    }
    assert provenance == {"approve-w1": "device-w1", "approve-w2": "device-w2"}

    decided_first = payload.interrupt_ids[0]
    async with open_saver() as saver:
        graph = _build_graph(saver, _runtime({"w2"}))
        await graph.ainvoke(
            Command(resume={decided_first: {"decision": "accept"}}), config=_config(thread_id)
        )
        remaining = pending_interrupt_payload(await graph.aget_state(_config(thread_id)))

    assert len(side_effects) == 1, "only the answered worker may run"
    assert remaining is not None
    assert len(remaining.interrupt_ids) == 1, "the undecided worker must still be waiting"
    assert remaining.interrupt_ids[0] != decided_first

    async with open_saver() as saver:
        graph = _build_graph(saver, _runtime(()))
        state = await graph.ainvoke(
            Command(resume={remaining.interrupt_ids[0]: {"decision": "accept"}}),
            config=_config(thread_id),
        )

    assert sorted(side_effects) == ["w1", "w2"]
    assert sorted(provider_calls) == ["w1", "w2"]
    results = state.get("worker_results") or []
    assert sorted((result.task_id, result.position) for result in results) == [("w1", 0), ("w2", 1)]

    rows = _receipt_rows(owner)
    assert len({row.execution_key for row in rows}) == 2
    assert {row.status for row in rows} == {ReceiptStatus.COMPLETED}


# ----------------------------------------------------------------------
# what the checkpoint has to carry
# ----------------------------------------------------------------------


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
        graph = _build_graph(saver, RecordingWorkerRuntime(side_effects, pause_task_ids={"w2"}))
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
        graph = _build_graph(saver, RecordingWorkerRuntime(side_effects, pause_task_ids={"w2"}))
        await graph.ainvoke(_turn(), config=_config(thread_id))
        snapshot = await graph.aget_state(_config(thread_id))

    dispatch = snapshot.values.get("planning_dispatch")

    assert isinstance(dispatch, PlanningDispatch)
    assert dispatch.tool_call_id == "call-dispatch-1"
    assert [task.task_id for task in dispatch.tasks] == ["w1", "w2"]
