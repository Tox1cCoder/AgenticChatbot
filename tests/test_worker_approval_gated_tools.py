"""A Planning worker's approval-gated tool call pauses the turn.

This file used to assert the opposite, and said why: workers ran inside
``asyncio.gather`` rather than as framework-managed tasks, so a worker calling
``interrupt()`` raised a ``GraphInterrupt`` that propagated through the gather
and cancelled every sibling. The worker therefore *refused* gated calls with
model-visible feedback and carried on -- failing closed, keeping the work
already done, and leaving approval to the top-level agent.

That was explicitly step 1 of three. Step 2 (fan-out as real parent topology,
so each worker is its own framework-managed task) and step 3 (pending
interrupts as the recovery authority) have both landed, so the constraint is
gone and the refusal with it. A worker now does what the design always wanted:
it pauses, the human decides, and the decision is applied to that worker alone.

The replay property this depends on -- an approved worker runs once, and its
already-completed siblings are not re-run -- is pinned by
``test_a_paused_worker_does_not_replay_its_completed_sibling`` in
``test_planning_worker_fanout.py`` against a real compiled graph.
"""

from __future__ import annotations

from typing import Annotated

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt
from typing_extensions import TypedDict

from app.ai.hitl_config import pending_interrupt_payload
from app.ai.workflow.contracts import WorkerResult, WorkerTask
from app.ai.workflow.planning_execution import PlanningLimits, PlanningWorkerRuntime


def _extend(existing: list | None, update: list | None) -> list:
    return [*(existing or []), *(update or [])]


class _WorkerState(TypedDict):
    done: Annotated[list, _extend]


def _limits() -> PlanningLimits:
    return PlanningLimits(
        max_tasks=8,
        max_concurrency=4,
        max_dispatch_waves=2,
        objective_max_chars=4000,
        parent_context_max_chars=12000,
    )


def _task(task_id: str = "w1", agent_id: str = "search_agent") -> WorkerTask:
    return WorkerTask(
        dispatch_id="d1",
        task_id=task_id,
        position=0,
        objective="send the email",
        agent_id=agent_id,
    )


def _parent_state(policy: dict | None = None) -> dict:
    return {
        "conversation_id": "conversation-1",
        "user_id": "owner",
        "device_id": "device-1",
        "context": {"hitl_policy": policy} if policy else {},
        "messages": [],
    }


_GATED_POLICY = {"master_enabled": True, "global_tools": ["send_email"]}


# ----------------------------------------------------------------------
# the pause reaches the parent
# ----------------------------------------------------------------------


class _PausingFactory:
    """A specialist whose gated tool call raises the framework's pause."""

    def __init__(self) -> None:
        self.calls = 0

    async def invoke_worker(self, request, *, task):
        self.calls += 1
        raise GraphBubbleUp("send_email needs approval")


class _CompletingFactory:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke_worker(self, request, *, task):
        self.calls += 1
        return WorkerResult(
            dispatch_id=task.dispatch_id,
            task_id=task.task_id,
            position=task.position,
            agent_id=task.agent_id,
            status="completed",
            content="sent",
        )


def _runtime(factory):
    return PlanningWorkerRuntime(
        specialist_factory=factory,
        rag_execution_graph=object(),
        limits=_limits(),
    )


async def test_a_gated_worker_call_pauses_instead_of_being_refused():
    """The pause is control flow and must reach the parent untouched."""
    factory = _PausingFactory()

    with pytest.raises(GraphBubbleUp):
        await _runtime(factory).run(_task(), _parent_state(_GATED_POLICY))

    assert factory.calls == 1


async def test_a_pause_is_never_normalized_into_a_failed_result():
    """Reporting it as failed would answer the turn without the human."""
    events: list[dict] = []

    with pytest.raises(GraphBubbleUp):
        await _runtime(_PausingFactory()).run(_task(), _parent_state(_GATED_POLICY), events.append)

    phases = [event["phase"] for event in events]
    assert phases == ["start", "interrupt"], (
        f"a paused worker must report an interrupt, not an end; got {phases}"
    )
    assert not any(event.get("status") == "failed" for event in events)


async def test_an_ungated_worker_call_still_completes_normally():
    factory = _CompletingFactory()

    result = await _runtime(factory).run(_task(), _parent_state())

    assert result.status == "completed"
    assert factory.calls == 1


# ----------------------------------------------------------------------
# the decision applies to that worker alone
# ----------------------------------------------------------------------


def _gated_worker_graph(side_effects: list[str], gated: set[str]):
    """Two workers; only the ones in ``gated`` ask for approval."""

    async def worker(payload: dict) -> dict:
        name = payload["name"]
        if name in gated:
            decision = interrupt({"action_requests": [{"tool_call_id": f"call-{name}"}]})
            if decision != "approve":
                return {"done": [f"{name}:rejected"]}
        side_effects.append(name)
        return {"done": [name]}

    graph = StateGraph(_WorkerState)
    graph.add_node("planning_dispatch", lambda _state: {})
    graph.add_node("planning_worker", worker)
    graph.add_node("planning_collect", lambda _state: {})
    graph.add_conditional_edges(
        "planning_dispatch",
        lambda _state: [Send("planning_worker", {"name": n}) for n in ("w1", "w2")],
        ["planning_worker"],
    )
    graph.add_edge("planning_worker", "planning_collect")
    graph.add_edge(START, "planning_dispatch")
    graph.add_edge("planning_collect", END)
    return graph.compile(checkpointer=InMemorySaver())


async def test_approving_a_gated_worker_runs_it_exactly_once():
    side_effects: list[str] = []
    graph = _gated_worker_graph(side_effects, gated={"w2"})
    config = {"configurable": {"thread_id": "t-approve"}}

    await graph.ainvoke({"done": []}, config=config)
    assert side_effects == ["w1"], "the ungated worker must not wait for the gated one"

    payload = pending_interrupt_payload(await graph.aget_state(config))
    assert payload is not None
    assert len(payload.interrupt_ids) == 1

    await graph.ainvoke(Command(resume={payload.interrupt_ids[0]: "approve"}), config=config)

    assert side_effects == ["w1", "w2"], (
        "approval must run the gated worker once and must not replay its sibling"
    )


async def test_rejecting_a_gated_worker_never_performs_the_side_effect():
    side_effects: list[str] = []
    graph = _gated_worker_graph(side_effects, gated={"w2"})
    config = {"configurable": {"thread_id": "t-reject"}}

    await graph.ainvoke({"done": []}, config=config)
    payload = pending_interrupt_payload(await graph.aget_state(config))

    await graph.ainvoke(Command(resume={payload.interrupt_ids[0]: "reject"}), config=config)

    assert side_effects == ["w1"]
    snapshot = await graph.aget_state(config)
    assert "w2:rejected" in snapshot.values["done"]


async def test_two_gated_workers_can_be_decided_one_at_a_time():
    """Partial approval is the flow the refusal path could not offer at all."""
    side_effects: list[str] = []
    graph = _gated_worker_graph(side_effects, gated={"w1", "w2"})
    config = {"configurable": {"thread_id": "t-partial"}}

    await graph.ainvoke({"done": []}, config=config)
    payload = pending_interrupt_payload(await graph.aget_state(config))
    assert len(payload.interrupt_ids) == 2
    assert side_effects == []

    await graph.ainvoke(Command(resume={payload.interrupt_ids[0]: "approve"}), config=config)
    assert len(side_effects) == 1

    remaining = pending_interrupt_payload(await graph.aget_state(config))
    assert len(remaining.interrupt_ids) == 1, "the undecided worker must still be waiting"

    await graph.ainvoke(Command(resume={remaining.interrupt_ids[0]: "approve"}), config=config)
    assert len(side_effects) == 2


# ----------------------------------------------------------------------
# nothing fabricates an approval status
# ----------------------------------------------------------------------


def test_no_worker_result_can_claim_it_is_awaiting_approval():
    """A worker that needs a human pauses; it does not report a status."""
    assert "awaiting_approval" not in str(WorkerResult.model_fields["status"].annotation)


def test_the_refusal_helper_is_gone():
    """It existed only because a worker in a gather could not interrupt."""
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    runtime = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted((repo_root / "app").rglob("*.py"))
    )
    assert "_refuse_worker_approval_gated_calls" not in runtime
