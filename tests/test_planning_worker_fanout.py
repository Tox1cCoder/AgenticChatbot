"""Planning fan-out as real outer-graph topology.

The bug this pins: fan-out used to happen inside a tool. A tool call is one
unit of work to the checkpointer, so when one worker paused for approval the
*whole* tool call was unfinished — and the resume re-ran it from the top,
executing every sibling that had already completed a second time. Side effects
came out as ``w1, w1, w2``.

Making the fan-out ``Send`` from a checkpointed parent node is what fixes it:
each worker is its own task with its own recorded result, so a resume runs only
what never finished. The regression below is the whole point of Task 4 and must
not be weakened into "the right number of results exist" — a duplicated side
effect leaves the result count correct.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.ai.workflow.contracts import WorkerResult, WorkerTask
from app.ai.workflow.inventory import AgentDescriptor, RoutingInventory
from app.ai.workflow.planning_execution import (
    PlanningLimits,
    PlanningNodeFactory,
    TodoActionOutcome,
)
from app.ai.workflow.state import WorkflowState

PLANNING_NODE_NAMES = (
    "planning_model",
    "planning_dispatch",
    "planning_worker",
    "planning_collect",
    "planning_actions",
    "planning_package",
)


def _limits(**overrides) -> PlanningLimits:
    payload = {
        "max_tasks": 8,
        "max_concurrency": 4,
        "max_dispatch_waves": 2,
        "objective_max_chars": 4000,
        "parent_context_max_chars": 12000,
    }
    payload.update(overrides)
    return PlanningLimits(**payload)


#: A custom agent this user once attached and has since detached. It is still
#: a real descriptor, so "unknown agent" would be the wrong rejection.
DETACHED_CUSTOM_AGENT_ID = "custom_agent:detached"


def _inventory() -> RoutingInventory:
    base = [
        AgentDescriptor(
            agent_id=agent_id,
            display_name=agent_id,
            capability_description="",
            enabled=True,
            attached=True,
            kind="base",
        )
        for agent_id in ("chat_agent", "search_agent", "rag_agent", "planning_agent")
    ]
    base.append(
        AgentDescriptor(
            agent_id=DETACHED_CUSTOM_AGENT_ID,
            display_name="Detached Analyst",
            capability_description="",
            enabled=True,
            attached=False,
            kind="custom",
        )
    )
    return RoutingInventory.from_descriptors(base)


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


class ScriptedPlanningModel:
    """Deterministic Planning model. One scripted response per model turn.

    The turn index is read from the state's own message history rather than
    from a call counter, so a *fresh* instance resuming a checkpointed thread
    replays the same turn the real model would. Counting calls would make the
    restart test script turn 1 again and mask the very replay it checks for.
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


def _model_response(content: str = "", tool_calls=()) -> Any:
    return SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=[dict(c) for c in tool_calls]),
        metadata={},
    )


class RecordingWorkerRuntime:
    """Records every worker that actually executes, and can pause one."""

    def __init__(self, *, pause_task_ids=()):
        self.side_effects: list[str] = []
        self._pause = set(pause_task_ids)
        self._resumed: set[str] = set()

    async def run(self, task: WorkerTask, state, writer=None) -> WorkerResult:
        if task.task_id in self._pause and task.task_id not in self._resumed:
            # A real approval-gated worker interrupts here. The decision value
            # is irrelevant; what matters is that the pause reaches the parent.
            interrupt({"task_id": task.task_id, "tool_call_id": f"approve-{task.task_id}"})
            self._resumed.add(task.task_id)
        self.side_effects.append(task.task_id)
        return WorkerResult(
            dispatch_id=task.dispatch_id,
            task_id=task.task_id,
            position=task.position,
            agent_id=task.agent_id,
            status="completed",
            content=f"result for {task.task_id}",
        )


async def _noop_todo_actions(state, calls) -> TodoActionOutcome:
    return TodoActionOutcome(
        todos=list(state.get("todos") or []),
        current_task_index=state.get("current_task_index"),
        tool_messages=tuple(
            ToolMessage(content="ok", tool_call_id=str(call.get("id") or ""), name="write_todos")
            for call in calls
        ),
        actions=("set_todos",),
    )


def _factory(model, worker_runtime, **overrides) -> PlanningNodeFactory:
    payload = {
        "call_model": model,
        "worker_runtime": worker_runtime,
        "limits": _limits(),
        "inventory_for": lambda state: _inventory(),
        "resolve_allowed_tools": lambda agent_id, state: (),
        "apply_todo_actions": _noop_todo_actions,
        "review_rubric": None,
    }
    payload.update(overrides)
    return PlanningNodeFactory(**payload)


def _planning_graph(factory: PlanningNodeFactory, checkpointer=None):
    """The Planning subset of the production topology, wired from descriptors.

    Built from ``factory.descriptors()`` rather than by hand so the test cannot
    pass against a node set the production builder would not register.
    """
    graph = StateGraph(WorkflowState)

    async def finalize(state):
        return {"execution_phase": "completed"}

    async def validate_output(state):
        return Command(goto="finalize")

    async def resolve_transition(state):
        return Command(goto="finalize")

    for name, node, destinations in factory.descriptors():
        graph.add_node(name, node, destinations=destinations)
    graph.add_node("validate_output", validate_output, destinations=("finalize",))
    graph.add_node("resolve_transition", resolve_transition, destinations=("finalize",))
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "planning_model")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer or MemorySaver())


def _config(thread_id: str = "routing-v2:conversation-1:turn-1") -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}


def _turn() -> dict:
    return {
        "messages": [],
        "conversation_id": "conversation-1",
        "user_id": "user-1",
        "active_agent_id": "planning_agent",
        "todos": [],
    }


def _result_keys(snapshot) -> set[tuple[str, str]]:
    return {
        (result.dispatch_id, result.task_id)
        for result in (snapshot.values.get("worker_results") or [])
    }


# ----------------------------------------------------------------------
# the replay regression
# ----------------------------------------------------------------------


async def test_a_paused_worker_does_not_replay_its_completed_sibling():
    """The whole point of the task: ``w1, w2`` and never ``w1, w1, w2``."""
    runtime = RecordingWorkerRuntime(pause_task_ids={"w2"})
    model = ScriptedPlanningModel(
        [
            _model_response(tool_calls=(_dispatch_call("w1", "w2"),)),
            _model_response(content="both pieces are done"),
        ]
    )
    graph = _planning_graph(_factory(model, runtime))

    await graph.ainvoke(_turn(), config=_config())

    assert runtime.side_effects == ["w1"]
    snapshot = await graph.aget_state(_config())
    assert len(snapshot.interrupts) == 1

    await graph.ainvoke(Command(resume={"decision": "accept"}), config=_config())

    assert runtime.side_effects == ["w1", "w2"]
    snapshot = await graph.aget_state(_config())
    assert len(_result_keys(snapshot)) == 2
    assert {task_id for _, task_id in _result_keys(snapshot)} == {"w1", "w2"}


async def test_a_restarted_workflow_resumes_on_the_same_thread():
    """A fresh graph object over the same checkpointer must not redo w1."""
    runtime = RecordingWorkerRuntime(pause_task_ids={"w2"})
    saver = MemorySaver()

    def build():
        model = ScriptedPlanningModel(
            [
                _model_response(tool_calls=(_dispatch_call("w1", "w2"),)),
                _model_response(content="done"),
            ]
        )
        return _planning_graph(_factory(model, runtime), checkpointer=saver)

    await build().ainvoke(_turn(), config=_config())
    assert runtime.side_effects == ["w1"]

    await build().ainvoke(Command(resume={"decision": "accept"}), config=_config())
    assert runtime.side_effects == ["w1", "w2"]


async def test_the_turn_reaches_a_public_outcome_after_collection():
    runtime = RecordingWorkerRuntime()
    model = ScriptedPlanningModel(
        [
            _model_response(tool_calls=(_dispatch_call("w1", "w2"),)),
            _model_response(content="synthesized from both"),
        ]
    )
    graph = _planning_graph(_factory(model, runtime))

    state = await graph.ainvoke(_turn(), config=_config())

    outcome = state.get("agent_outcome")
    assert outcome is not None
    assert outcome.agent_id == "planning_agent"
    assert outcome.response.message.content == "synthesized from both"


# ----------------------------------------------------------------------
# topology
# ----------------------------------------------------------------------


def test_every_required_planning_node_is_registered():
    factory = _factory(ScriptedPlanningModel([]), RecordingWorkerRuntime())
    assert tuple(name for name, _, _ in factory.descriptors()) == PLANNING_NODE_NAMES


def test_no_planning_tool_stage_survives():
    factory = _factory(ScriptedPlanningModel([]), RecordingWorkerRuntime())
    names = {name for name, _, _ in factory.descriptors()}
    assert "planning_tools" not in names


def test_dispatch_only_ever_targets_the_worker():
    factory = _factory(ScriptedPlanningModel([]), RecordingWorkerRuntime())
    destinations = dict((name, dests) for name, _, dests in factory.descriptors())

    assert destinations["planning_dispatch"] == ("planning_worker",)
    assert destinations["planning_worker"] == ("planning_collect",)
    assert "END" not in destinations["planning_package"]
    assert "validate_output" in destinations["planning_package"]


def test_no_planning_node_declares_end_as_a_destination():
    """Only ``finalize`` terminates a turn."""
    factory = _factory(ScriptedPlanningModel([]), RecordingWorkerRuntime())
    for name, _, destinations in factory.descriptors():
        assert END not in destinations, name
        assert "__end__" not in destinations, name


def test_the_compiled_planning_topology_has_no_static_specialist_edges():
    factory = _factory(ScriptedPlanningModel([]), RecordingWorkerRuntime())
    graph = _planning_graph(factory).get_graph()

    end_sources = {edge.source for edge in graph.edges if edge.target == "__end__"}
    assert end_sources == {"finalize"}


# ----------------------------------------------------------------------
# dispatch is server-owned and bounded
# ----------------------------------------------------------------------


async def test_a_dispatch_sends_one_worker_per_validated_task():
    runtime = RecordingWorkerRuntime()
    model = ScriptedPlanningModel(
        [
            _model_response(tool_calls=(_dispatch_call("w1", "w2", "w3"),)),
            _model_response(content="done"),
        ]
    )
    graph = _planning_graph(_factory(model, runtime))

    await graph.ainvoke(_turn(), config=_config())

    assert runtime.side_effects == ["w1", "w2", "w3"]


async def test_collection_pairs_exactly_one_tool_message_to_the_original_call():
    """An unpaired dispatch leaves the model with a call that got no result."""
    runtime = RecordingWorkerRuntime()
    model = ScriptedPlanningModel(
        [
            _model_response(tool_calls=(_dispatch_call("w1", "w2"),)),
            _model_response(content="done"),
        ]
    )
    graph = _planning_graph(_factory(model, runtime))

    state = await graph.ainvoke(_turn(), config=_config())

    paired = [
        message
        for message in state["messages"]
        if isinstance(message, ToolMessage) and message.tool_call_id == "call-dispatch-1"
    ]
    assert len(paired) == 1
    assert "w1" in paired[0].content and "w2" in paired[0].content


async def test_collection_reads_results_in_position_order():
    runtime = RecordingWorkerRuntime()
    model = ScriptedPlanningModel(
        [
            _model_response(tool_calls=(_dispatch_call("w1", "w2", "w3"),)),
            _model_response(content="done"),
        ]
    )
    graph = _planning_graph(_factory(model, runtime))

    state = await graph.ainvoke(_turn(), config=_config())

    paired = next(
        message
        for message in state["messages"]
        if isinstance(message, ToolMessage) and message.tool_call_id == "call-dispatch-1"
    )
    assert paired.content.index("w1") < paired.content.index("w2") < paired.content.index("w3")


def _proposed(task_id: str, *, objective: str | None = None, agent_id: str = "chat_agent") -> dict:
    return {
        "task_id": task_id,
        "objective": objective if objective is not None else f"do {task_id}",
        "agent_id": agent_id,
    }


def _raw_dispatch_call(tasks: list[dict], call_id: str = "call-dispatch-1") -> dict:
    return {"name": "dispatch_subagents", "id": call_id, "args": {"tasks": tasks}}


_HANDOFF_CALL = {
    "name": "hand_off",
    "id": "call-handoff-1",
    "args": {"to_agent_id": "search_agent", "reason": "changed my mind"},
}

#: Every way one dispatch proposal can be refused, and the exact code the model
#: is told. The objective bound is the *deployment* limit rather than the
#: schema's 4000-character ceiling: the ceiling is pydantic's and is already
#: enforced before this code runs, while the limit is what an operator tunes.
_INVALID_DISPATCHES: dict[str, tuple[tuple[dict, ...], str, dict]] = {
    "ninth_task": (
        (_raw_dispatch_call([_proposed(f"w{i}") for i in range(9)]),),
        "dispatch_task_limit",
        {},
    ),
    "duplicate_id": (
        (_raw_dispatch_call([_proposed("w1"), _proposed("w1")]),),
        "duplicate_task_id",
        {},
    ),
    "oversized_objective": (
        (_raw_dispatch_call([_proposed("w1", objective="x" * 200)]),),
        "objective_too_long",
        {"objective_max_chars": 100},
    ),
    "recursive_planning": (
        (_raw_dispatch_call([_proposed("w1", agent_id="planning_agent")]),),
        "recursive_planning",
        {},
    ),
    "detached_custom_agent": (
        (_raw_dispatch_call([_proposed("w1", agent_id=DETACHED_CUSTOM_AGENT_ID)]),),
        "agent_unavailable",
        {},
    ),
    "dispatch_plus_handoff": (
        (_raw_dispatch_call([_proposed("w1")]), _HANDOFF_CALL),
        "dispatch_with_handoff",
        {},
    ),
}


@pytest.mark.parametrize("invalid", sorted(_INVALID_DISPATCHES))
async def test_invalid_dispatch_starts_no_workers(invalid):
    """Validation is all-or-nothing: nothing partially runs and one error pairs.

    An invalid ninth task must not leave eight workers already executing, and
    the refusal must come back on the dispatch call's own ``tool_call_id`` —
    an unpaired tool call is a malformed message history to the next provider
    turn.
    """
    tool_calls, expected_code, limit_overrides = _INVALID_DISPATCHES[invalid]
    runtime = RecordingWorkerRuntime()
    model = ScriptedPlanningModel(
        [
            _model_response(tool_calls=tool_calls),
            _model_response(content="recovered without delegating"),
        ]
    )
    factory = _factory(model, runtime, limits=_limits(**limit_overrides))
    graph = _planning_graph(factory)

    state = await graph.ainvoke(_turn(), config=_config())

    assert runtime.side_effects == []
    assert state.get("worker_results", []) == []
    assert state.get("planning_dispatched_task_count", 0) == 0

    errors = [
        message
        for message in state["messages"]
        if isinstance(message, ToolMessage) and message.tool_call_id == "call-dispatch-1"
    ]
    assert len(errors) == 1
    assert expected_code in errors[0].content


async def test_a_rejected_dispatch_does_not_consume_a_wave():
    """A refusal is not an attempt. Otherwise one malformed proposal would
    silently cost the model half its delegation budget."""
    runtime = RecordingWorkerRuntime()
    model = ScriptedPlanningModel(
        [
            _model_response(tool_calls=(_raw_dispatch_call([_proposed("w1"), _proposed("w1")]),)),
            _model_response(tool_calls=(_dispatch_call("a1", call_id="c2"),)),
            _model_response(tool_calls=(_dispatch_call("b1", call_id="c3"),)),
            _model_response(content="done delegating"),
        ]
    )
    graph = _planning_graph(_factory(model, runtime))

    state = await graph.ainvoke(_turn(), config=_config())

    assert runtime.side_effects == ["a1", "b1"]
    assert state["planning_dispatch_waves"] == 2


async def test_a_third_wave_is_refused_and_reported():
    runtime = RecordingWorkerRuntime()
    model = ScriptedPlanningModel(
        [
            _model_response(tool_calls=(_dispatch_call("w1", call_id="c1"),)),
            _model_response(tool_calls=(_dispatch_call("w2", call_id="c2"),)),
            _model_response(tool_calls=(_dispatch_call("w3", call_id="c3"),)),
            _model_response(content="stopped delegating"),
        ]
    )
    graph = _planning_graph(_factory(model, runtime))

    state = await graph.ainvoke(_turn(), config=_config())

    assert runtime.side_effects == ["w1", "w2"]
    assert state["planning_dispatch_waves"] == 2
    refusals = [
        message
        for message in state["messages"]
        if isinstance(message, ToolMessage) and "dispatch_wave_limit" in str(message.content)
    ]
    assert len(refusals) == 1


async def test_the_cumulative_task_budget_spans_waves():
    runtime = RecordingWorkerRuntime()
    model = ScriptedPlanningModel(
        [
            _model_response(
                tool_calls=(_dispatch_call(*[f"a{i}" for i in range(6)], call_id="c1"),)
            ),
            _model_response(
                tool_calls=(_dispatch_call(*[f"b{i}" for i in range(4)], call_id="c2"),)
            ),
            _model_response(content="stopped"),
        ]
    )
    graph = _planning_graph(_factory(model, runtime))

    state = await graph.ainvoke(_turn(), config=_config())

    assert len(runtime.side_effects) == 6
    assert state["planning_dispatched_task_count"] == 6
    assert any(
        "dispatch_task_limit" in str(message.content)
        for message in state["messages"]
        if isinstance(message, ToolMessage)
    )


# ----------------------------------------------------------------------
# the model's other decisions
# ----------------------------------------------------------------------


async def test_a_final_text_answer_goes_to_packaging():
    runtime = RecordingWorkerRuntime()
    model = ScriptedPlanningModel([_model_response(content="here is the plan")])
    graph = _planning_graph(_factory(model, runtime))

    state = await graph.ainvoke(_turn(), config=_config())

    assert runtime.side_effects == []
    assert state["agent_outcome"].response.message.content == "here is the plan"


async def test_write_todos_goes_to_actions_and_returns_to_the_model():
    seen: list[list[dict]] = []

    async def apply(state, calls):
        seen.append([dict(call) for call in calls])
        return TodoActionOutcome(
            todos=[{"id": "t1", "status": "pending", "content": "step one"}],
            current_task_index=0,
            tool_messages=(
                ToolMessage(
                    content="plan written",
                    tool_call_id=str(calls[0].get("id") or ""),
                    name="write_todos",
                ),
            ),
            actions=("set_todos",),
        )

    runtime = RecordingWorkerRuntime()
    model = ScriptedPlanningModel(
        [
            _model_response(
                tool_calls=(
                    {"name": "write_todos", "id": "call-todos-1", "args": {"action": "set_todos"}},
                )
            ),
            _model_response(content="plan is ready"),
        ]
    )
    graph = _planning_graph(_factory(model, runtime, apply_todo_actions=apply))

    state = await graph.ainvoke(_turn(), config=_config())

    assert [call["name"] for call in seen[0]] == ["write_todos"]
    assert state["todos"] == [{"id": "t1", "status": "pending", "content": "step one"}]
    assert state["agent_outcome"].response.message.content == "plan is ready"


async def test_an_exclusive_handoff_goes_to_the_transition_resolver():
    runtime = RecordingWorkerRuntime()
    model = ScriptedPlanningModel(
        [
            _model_response(
                tool_calls=(
                    {
                        "name": "hand_off",
                        "id": "call-handoff-1",
                        "args": {"to_agent_id": "search_agent", "reason": "needs the web"},
                    },
                )
            )
        ]
    )
    graph = _planning_graph(_factory(model, runtime))

    state = await graph.ainvoke(_turn(), config=_config())

    assert runtime.side_effects == []
    pending = state.get("pending_transition")
    assert pending is not None
    assert pending.to_agent_id == "search_agent"
    assert pending.tool_call_id == "call-handoff-1"


# A dispatch mixed with a handoff is one of the cases in
# ``test_invalid_dispatch_starts_no_workers`` above; it is a rejected proposal
# like any other, not a separate kind of failure.


# ----------------------------------------------------------------------
# concurrency
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bound", [2, 4])
async def test_the_outer_invocation_bounds_live_workers(bound):
    """``max_concurrency`` is the parent's to enforce, not each worker's.

    ``4`` is the production default and ``2`` proves the bound is read rather
    than coincidental — eight tasks against a bound of 8 would pass either way.
    """
    import asyncio

    class ConcurrencyProbe:
        def __init__(self):
            self.running = 0
            self.peak = 0
            self.side_effects: list[str] = []

        async def run(self, task, state, writer=None):
            self.running += 1
            self.peak = max(self.peak, self.running)
            try:
                await asyncio.sleep(0.02)
                self.side_effects.append(task.task_id)
                return WorkerResult(
                    dispatch_id=task.dispatch_id,
                    task_id=task.task_id,
                    position=task.position,
                    agent_id=task.agent_id,
                    status="completed",
                    content="done",
                )
            finally:
                self.running -= 1

    probe = ConcurrencyProbe()
    model = ScriptedPlanningModel(
        [
            _model_response(tool_calls=(_dispatch_call(*[f"w{i}" for i in range(8)]),)),
            _model_response(content="done"),
        ]
    )
    factory = _factory(model, probe, limits=_limits(max_concurrency=bound))
    graph = _planning_graph(factory)

    config = _config()
    config["max_concurrency"] = factory.limits.max_concurrency
    await graph.ainvoke(_turn(), config=config)

    assert len(probe.side_effects) == 8
    assert probe.peak <= bound, f"{probe.peak} workers ran at once against a bound of {bound}"


# ----------------------------------------------------------------------
# the worker node itself
# ----------------------------------------------------------------------


async def test_the_worker_node_returns_only_its_typed_result():
    runtime = RecordingWorkerRuntime()
    factory = _factory(ScriptedPlanningModel([]), runtime)
    task = WorkerTask(
        dispatch_id="d1", task_id="w1", position=0, objective="do w1", agent_id="chat_agent"
    )

    update = await factory.planning_worker(
        {"worker_task": task, "worker_parent_state": {"conversation_id": "conversation-1"}}
    )

    assert set(update) == {"worker_results"}
    assert update["worker_results"][0].task_id == "w1"


async def test_the_worker_node_receives_the_bounded_parent_scope():
    captured: list[dict] = []

    class CapturingRuntime(RecordingWorkerRuntime):
        async def run(self, task, state, writer=None):
            captured.append(dict(state))
            return await super().run(task, state, writer)

    runtime = CapturingRuntime()
    model = ScriptedPlanningModel(
        [
            _model_response(tool_calls=(_dispatch_call("w1"),)),
            _model_response(content="done"),
        ]
    )
    graph = _planning_graph(_factory(model, runtime))

    turn = _turn()
    turn["persona"] = "default"
    await graph.ainvoke(turn, config=_config())

    scope = captured[0]
    assert scope["conversation_id"] == "conversation-1"
    assert scope["user_id"] == "user-1"
    assert "messages" not in scope


@pytest.mark.parametrize("field", ["worker_task"])
async def test_the_worker_node_rejects_a_payload_without_a_task(field):
    factory = _factory(ScriptedPlanningModel([]), RecordingWorkerRuntime())
    with pytest.raises(KeyError):
        await factory.planning_worker({})
