"""Custom agents as Planning workers, after the fan-out cutover.

The invariants are unchanged; what enforces them moved. Target validation is
``validate_dispatch_call`` against the live routing inventory rather than a
Pydantic validator on a task model, and a worker runs through
``PlanningWorkerRuntime`` rather than an isolated-context helper.

One thing here is easy to lose in that move and is not covered elsewhere: a
custom agent's runtime id is ``custom_agent:<uuid>``, which is not a name a
human should be shown. The display name has to survive to the stream, and it
matters most for a *failed* worker -- the case where the bare id tells the
reader least.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.ai.graph import MultiAgentWorkflow
from app.ai.workflow.contracts import WorkerTask
from app.ai.workflow.inventory import AgentDescriptor, RoutingInventory
from app.ai.workflow.planning_execution import (
    InvalidPlanningDispatch,
    PlanningLimits,
    PlanningWorkerRuntime,
    validate_dispatch_call,
)
from app.services.event_streaming.langchain_v3 import V3ProtocolTranslator

_BASE_AGENTS = {
    "chat_agent": object(),
    "rag_agent": object(),
    "search_agent": object(),
    "image_generator_agent": object(),
    "planning_agent": object(),
    "canvas_agent": object(),
}


def _workflow():
    wf = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    wf.agents = dict(_BASE_AGENTS)
    wf._runtime_model_resolver = None
    wf._model_usage_recorder = None
    return wf


def _custom_entry(runtime_id: str, name: str = "Data Analyst") -> dict:
    return {
        "id": runtime_id.split(":", 1)[1],
        "runtime_agent_id": runtime_id,
        "name": name,
        "prompt": "p",
        "model_request": {"provider_type": "openai", "model": "gpt-4.1-mini"},
        "tool_refs": [],
        "skill_refs": [],
    }


def _parent_state(custom_ids, **overrides):
    state = {
        "conversation_id": "c1",
        "user_id": "u1",
        "device_id": None,
        "persona": None,
        "model_request": None,
        "context": {},
        "custom_agents": {cid: _custom_entry(cid) for cid in custom_ids},
    }
    state.update(overrides)
    return state


def _inventory(*custom_ids, attached: bool = True) -> RoutingInventory:
    descriptors = [
        AgentDescriptor(
            agent_id=agent_id,
            display_name=agent_id,
            capability_description="",
            enabled=True,
            attached=True,
            kind="base",
        )
        for agent_id in _BASE_AGENTS
    ]
    descriptors.extend(
        AgentDescriptor(
            agent_id=cid,
            display_name="Data Analyst",
            capability_description="",
            enabled=True,
            attached=attached,
            kind="custom",
        )
        for cid in custom_ids
    )
    return RoutingInventory.from_descriptors(descriptors)


def _tool_call(agent_id: str, task_id: str = "t1") -> dict:
    return {
        "name": "dispatch_subagents",
        "id": "call-1",
        "args": {"tasks": [{"task_id": task_id, "objective": "analyze", "agent_id": agent_id}]},
    }


def _validate(agent_id: str, *, custom_ids=(), attached: bool = True):
    call = _tool_call(agent_id)
    state = _parent_state(custom_ids)
    state["messages"] = [SimpleNamespace(tool_calls=[call])]
    return validate_dispatch_call(
        tool_call=call,
        state=state,
        inventory=_inventory(*custom_ids, attached=attached),
        limits=PlanningLimits(
            max_tasks=8,
            max_concurrency=4,
            max_dispatch_waves=2,
            objective_max_chars=4000,
            parent_context_max_chars=12000,
        ),
        resolve_allowed_tools=lambda _agent_id, _state: (),
    )


# ----------------------------------------------------------------------
# dispatch targets
# ----------------------------------------------------------------------


def test_an_attached_custom_agent_is_a_valid_worker_target():
    rid = f"custom_agent:{uuid4()}"
    dispatch = _validate(rid, custom_ids=[rid])
    assert dispatch.tasks[0].agent_id == rid


def test_a_base_agent_is_a_valid_worker_target():
    dispatch = _validate("search_agent")
    assert dispatch.tasks[0].agent_id == "search_agent"


def test_planning_cannot_dispatch_itself():
    with pytest.raises(InvalidPlanningDispatch) as excinfo:
        _validate("planning_agent")
    assert excinfo.value.code == "recursive_planning"


def test_an_unattached_custom_agent_is_refused():
    rid = f"custom_agent:{uuid4()}"
    with pytest.raises(InvalidPlanningDispatch) as excinfo:
        _validate(rid, custom_ids=[rid], attached=False)
    assert excinfo.value.code == "agent_unavailable"


def test_a_custom_agent_the_inventory_never_saw_is_refused():
    with pytest.raises(InvalidPlanningDispatch) as excinfo:
        _validate(f"custom_agent:{uuid4()}")
    assert excinfo.value.code == "unknown_agent"


# ----------------------------------------------------------------------
# runtime resolution
# ----------------------------------------------------------------------


def test_build_custom_agent_resolves_only_an_attached_worker():
    wf = _workflow()
    rid = f"custom_agent:{uuid4()}"
    state = _parent_state([rid])

    agent = wf._build_custom_agent(state, rid)

    assert agent is not None
    assert agent.agent_id == rid
    assert wf._build_custom_agent(state, f"custom_agent:{uuid4()}") is None


# ----------------------------------------------------------------------
# display identity survives to the stream
# ----------------------------------------------------------------------


def _task(agent_id: str) -> WorkerTask:
    return WorkerTask(
        dispatch_id="d1",
        task_id="w1",
        position=0,
        objective="analyze",
        agent_id=agent_id,
    )


class _Factory:
    def __init__(self, raises=None):
        self._raises = raises

    async def invoke_worker(self, request, *, task):
        if self._raises is not None:
            raise self._raises
        from app.ai.workflow.contracts import WorkerResult

        return WorkerResult(
            dispatch_id=task.dispatch_id,
            task_id=task.task_id,
            position=task.position,
            agent_id=task.agent_id,
            status="completed",
            content="worker answer",
        )


def _runtime(factory):
    return PlanningWorkerRuntime(
        specialist_factory=factory,
        rag_execution_graph=object(),
        limits=PlanningLimits(
            max_tasks=8,
            max_concurrency=4,
            max_dispatch_waves=2,
            objective_max_chars=4000,
            parent_context_max_chars=12000,
        ),
    )


def _projected(events: list[dict]):
    translator = V3ProtocolTranslator()
    produced = []
    for event in events:
        produced.extend(
            translator.translate(
                {
                    "type": "event",
                    "method": "custom",
                    "params": {"namespace": [], "timestamp": 0, "data": event},
                }
            )
        )
    return produced


async def test_a_custom_worker_is_shown_by_name_not_by_runtime_id():
    rid = f"custom_agent:{uuid4()}"
    events: list[dict] = []

    result = await _runtime(_Factory()).run(_task(rid), _parent_state([rid]), events.append)

    assert result.status == "completed"
    assert result.agent_id == rid
    assert [event.subagent.name for event in _projected(events)] == [
        "Data Analyst",
        "Data Analyst",
    ]


async def test_a_failed_custom_worker_keeps_its_display_name():
    """The bare runtime id is least useful exactly when the worker failed."""
    rid = f"custom_agent:{uuid4()}"
    events: list[dict] = []

    result = await _runtime(_Factory(raises=RuntimeError("boom"))).run(
        _task(rid), _parent_state([rid]), events.append
    )

    assert result.status == "failed"
    assert result.error_code == "tool_execution_failed"
    projected = _projected(events)
    assert projected[-1].subagent.name == "Data Analyst"
    assert projected[-1].subagent.status == "failed"


async def test_a_base_agent_worker_is_shown_by_its_agent_id():
    events: list[dict] = []

    await _runtime(_Factory()).run(_task("search_agent"), _parent_state([]), events.append)

    assert {event.subagent.name for event in _projected(events)} == {"search_agent"}
