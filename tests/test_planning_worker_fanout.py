"""How Planning workers are dispatched, and what that buys.

The dispatcher used to be an ``asyncio.gather`` over hand-rolled coroutines.
That made every worker invisible to the framework: nothing bounded how many ran
at once, nothing bounded how many a single plan could start, and a worker could
not pause for a human without taking its siblings down with it.

These pin the fan-out contract. The interrupt behaviour those tasks make
possible lands with the worker loop itself; what is asserted here is that the
dispatch is framework-managed and bounded, which is its prerequisite.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.ai.planning_subagents import (
    DispatchSubagentsInput,
    PlanningSubagentDispatcher,
    PlanningSubagentTask,
)
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from app.core.config import settings


def _ok_response(content: str) -> AgentResponse:
    return AgentResponse(
        agent_type=AgentType.CHAT,
        agent_id="chat_agent",
        message=AgentMessage(role=MessageRole.ASSISTANT, content=content),
        metadata={},
    )


class _StubWorkflow:
    """Minimal MultiAgentWorkflow surface the dispatcher calls into."""

    def __init__(self, runner):
        self._runner = runner

    async def _run_agent_in_isolated_context(self, **kwargs: Any) -> AgentResponse:
        return await self._runner(**kwargs)


def _tasks(count: int) -> list[PlanningSubagentTask]:
    return [
        PlanningSubagentTask(
            id=f"w{index}", agent="chat_agent", task=f"do the {index}th piece of work"
        )
        for index in range(1, count + 1)
    ]


class _ConcurrencyProbe:
    """Records the high-water mark of overlapping worker executions."""

    def __init__(self, hold: float = 0.05):
        self.running = 0
        self.peak = 0
        self.started: list[str] = []
        self._hold = hold

    async def __call__(self, **kwargs: Any) -> AgentResponse:
        self.started.append(kwargs["task_id"])
        self.running += 1
        self.peak = max(self.peak, self.running)
        try:
            await asyncio.sleep(self._hold)
            return _ok_response("done")
        finally:
            self.running -= 1


async def test_worker_fanout_respects_the_configured_concurrency_bound(monkeypatch):
    """``planning_worker_max_concurrency`` is a validated setting with a
    default of 4. Nothing was reading it, so a plan with twelve tasks opened
    twelve provider connections at once."""
    monkeypatch.setattr(settings, "planning_worker_max_concurrency", 3)
    monkeypatch.setattr(settings, "planning_worker_max_tasks", 64)

    probe = _ConcurrencyProbe()
    dispatcher = PlanningSubagentDispatcher(workflow=_StubWorkflow(probe), settings=settings)

    await dispatcher.dispatch(
        DispatchSubagentsInput(tasks=_tasks(9)), parent_state={}
    )

    assert probe.peak <= 3, f"{probe.peak} workers ran at once against a bound of 3"
    assert len(probe.started) == 9, "every dispatched task must still run"


async def test_worker_fanout_bounds_how_many_tasks_one_plan_can_start(monkeypatch):
    """``planning_worker_max_tasks`` bounds one dispatch. Truncating once,
    visibly, beats each worker discovering its own limit."""
    monkeypatch.setattr(settings, "planning_worker_max_tasks", 4)
    monkeypatch.setattr(settings, "planning_worker_max_concurrency", 4)

    probe = _ConcurrencyProbe(hold=0.0)
    dispatcher = PlanningSubagentDispatcher(workflow=_StubWorkflow(probe), settings=settings)

    result = await dispatcher.dispatch(
        DispatchSubagentsInput(tasks=_tasks(7)), parent_state={}
    )

    assert len(probe.started) == 4
    assert [entry.id for entry in result.results] == ["w1", "w2", "w3", "w4"]


async def test_worker_results_keep_their_plan_order(monkeypatch):
    """Synthesis reads the plan in the order it was written. Completion order
    is an accident of latency and must not reorder it."""
    monkeypatch.setattr(settings, "planning_worker_max_tasks", 8)
    monkeypatch.setattr(settings, "planning_worker_max_concurrency", 8)

    delays = {"w1": 0.06, "w2": 0.01, "w3": 0.03}

    async def runner(**kwargs: Any) -> AgentResponse:
        await asyncio.sleep(delays[kwargs["task_id"]])
        return _ok_response(f"answer for {kwargs['task_id']}")

    dispatcher = PlanningSubagentDispatcher(workflow=_StubWorkflow(runner), settings=settings)

    result = await dispatcher.dispatch(
        DispatchSubagentsInput(tasks=_tasks(3)), parent_state={}
    )

    assert [entry.id for entry in result.results] == ["w1", "w2", "w3"]


async def test_a_paused_worker_does_not_become_a_failed_result(monkeypatch):
    """A pause is not a failure, and swallowing it loses the turn.

    ``GraphBubbleUp`` is how LangGraph carries an interrupt or a parent
    command. Catching it as an ordinary exception turns a turn that was
    waiting for a human into a worker that reports it failed — and the plan
    then synthesizes an answer from work nobody approved.
    """
    from langgraph.errors import GraphBubbleUp

    monkeypatch.setattr(settings, "planning_worker_max_tasks", 8)
    monkeypatch.setattr(settings, "planning_worker_max_concurrency", 8)

    async def runner(**kwargs: Any) -> AgentResponse:
        raise GraphBubbleUp("worker is waiting for a human")

    dispatcher = PlanningSubagentDispatcher(workflow=_StubWorkflow(runner), settings=settings)

    with pytest.raises(GraphBubbleUp):
        await dispatcher.dispatch(DispatchSubagentsInput(tasks=_tasks(1)), parent_state={})
