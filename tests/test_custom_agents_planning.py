"""Planning-dispatch tests for custom-agent workers (Task 11)."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.ai.graph import MultiAgentWorkflow
from app.ai.planning_subagents import PlanningSubagentTask

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
    return wf


def _parent_state(custom_ids):
    return {
        "conversation_id": "c1",
        "user_id": "u1",
        "device_id": None,
        "persona": None,
        "model_request": None,
        "context": {},
        "custom_agents": {
            cid: {
                "id": cid.split(":", 1)[1],
                "runtime_agent_id": cid,
                "name": "Worker",
                "prompt": "p",
                "model_request": {"provider_type": "openai", "model": "gpt-4.1-mini"},
                "tool_refs": [],
                "skill_refs": [],
            }
            for cid in custom_ids
        },
    }


def test_planning_task_accepts_custom_and_base_targets():
    rid = f"custom_agent:{uuid4()}"
    assert PlanningSubagentTask(id="t1", agent=rid, task="do work").agent == rid
    assert (
        PlanningSubagentTask(id="t2", agent="search_agent", task="search").agent == "search_agent"
    )


def test_planning_task_rejects_planning_agent_target():
    with pytest.raises(ValidationError):
        PlanningSubagentTask(id="t1", agent="planning_agent", task="recurse")
    with pytest.raises(ValidationError):
        PlanningSubagentTask(id="t2", agent="   ", task="empty")


def test_build_custom_agent_resolves_attached_worker():
    wf = _workflow()
    rid = f"custom_agent:{uuid4()}"
    state = _parent_state([rid])
    agent = wf._build_custom_agent(state, rid)
    assert agent is not None
    assert agent.agent_id == rid
    # Unattached id resolves to None.
    assert wf._build_custom_agent(state, f"custom_agent:{uuid4()}") is None


@pytest.mark.asyncio
async def test_isolated_context_rejects_planning_agent_worker():
    wf = _workflow()
    with pytest.raises(ValueError, match="planning_agent"):
        await wf._run_agent_in_isolated_context(
            agent_name="planning_agent",
            task_prompt="x",
            parent_state=_parent_state([]),
        )


@pytest.mark.asyncio
async def test_isolated_context_rejects_unattached_custom_worker():
    wf = _workflow()
    bogus = f"custom_agent:{uuid4()}"
    with pytest.raises(ValueError, match="Unknown subagent target"):
        await wf._run_agent_in_isolated_context(
            agent_name=bogus,
            task_prompt="x",
            parent_state=_parent_state([]),
        )


@pytest.mark.asyncio
async def test_custom_subagent_result_includes_display_identity():
    from app.ai.planning_subagents import (
        DispatchSubagentsInput,
        PlanningSubagentDispatcher,
        PlanningSubagentTask,
    )
    from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole

    rid = f"custom_agent:{uuid4()}"

    class _Workflow:
        async def _run_agent_in_isolated_context(self, **kwargs):
            return AgentResponse(
                agent_type=AgentType.CHAT,
                agent_id=rid,
                message=AgentMessage(role=MessageRole.ASSISTANT, content="worker answer"),
                metadata={
                    "runtime_agent_id": rid,
                    "custom_agent_id": rid.split(":", 1)[1],
                    "custom_agent_name": "Data Analyst",
                },
            )

    dispatcher = PlanningSubagentDispatcher(workflow=_Workflow())
    result = await dispatcher.dispatch(
        DispatchSubagentsInput(tasks=[PlanningSubagentTask(id="w1", agent=rid, task="analyze")]),
        parent_state={
            "custom_agents": {
                rid: {
                    "id": rid.split(":", 1)[1],
                    "runtime_agent_id": rid,
                    "name": "Data Analyst",
                }
            }
        },
    )

    entry = result.results[0]
    assert entry.agent == rid
    assert entry.agent_name == "Data Analyst"
    assert entry.custom_agent_id == rid.split(":", 1)[1]


@pytest.mark.asyncio
async def test_failed_custom_subagent_result_keeps_display_identity():
    from app.ai.planning_subagents import (
        DispatchSubagentsInput,
        PlanningSubagentDispatcher,
        PlanningSubagentTask,
    )

    rid = f"custom_agent:{uuid4()}"

    class _Workflow:
        async def _run_agent_in_isolated_context(self, **kwargs):
            raise RuntimeError("boom")

    dispatcher = PlanningSubagentDispatcher(workflow=_Workflow())
    result = await dispatcher.dispatch(
        DispatchSubagentsInput(tasks=[PlanningSubagentTask(id="w1", agent=rid, task="analyze")]),
        parent_state={
            "custom_agents": {
                rid: {
                    "id": rid.split(":", 1)[1],
                    "runtime_agent_id": rid,
                    "name": "Data Analyst",
                }
            }
        },
    )

    entry = result.results[0]
    assert entry.status == "failed"
    assert entry.agent == rid
    assert entry.agent_name == "Data Analyst"
    assert entry.custom_agent_id == rid.split(":", 1)[1]
