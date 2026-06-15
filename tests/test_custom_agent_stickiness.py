"""Custom-agent stickiness across turns.

A custom agent invoked on one turn must keep handling natural follow-ups on the
next turn, instead of the router silently re-routing a terse follow-up to a base
agent (which loses the custom agent's deferred tools/skills and triggers a
handoff storm). Stickiness is scoped to custom agents only and is released when
planning supervises or the user explicitly names a different attached custom
agent.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage

from app.ai.agents.router import Router
from app.ai.graph import MultiAgentWorkflow
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole


class _RecordingRouter:
    """Stub router that records whether the LLM route path was reached."""

    def __init__(self):
        self.calls = 0
        # Reuse the real deterministic explicit-name matcher.
        self._match_explicit_custom_agent = Router._match_explicit_custom_agent

    async def route_message(self, *args, **kwargs):
        self.calls += 1
        return "chat_agent"


def _workflow(router):
    wf = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    wf.agents = {
        "chat_agent": object(),
        "rag_agent": object(),
        "search_agent": object(),
        "image_generator_agent": object(),
        "planning_agent": object(),
        "canvas_agent": object(),
    }
    wf._runtime_model_resolver = None
    wf.router = router
    wf.document_repository = None
    return wf


def _custom_entry(rid, *, name="Analyst", description="Answers analytics questions."):
    return {
        "id": rid.split(":", 1)[1],
        "runtime_agent_id": rid,
        "name": name,
        "description": description,
        "prompt": "p",
        "model_request": {"provider_type": "openai", "model": "gpt-4.1-mini"},
        "tool_refs": [],
        "skill_refs": [],
    }


def _followup_state(*, last_agent, content, custom_agents):
    return {
        "selected_agent": None,
        "last_agent": last_agent,
        "messages": [HumanMessage(content=content)],
        "custom_agents": custom_agents,
        "context": {},
        "planning_mode_enabled": False,
        "has_existing_plan": False,
    }


@pytest.mark.asyncio
async def test_followup_sticks_to_previous_custom_agent_without_router():
    router = _RecordingRouter()
    wf = _workflow(router)
    rid = f"custom_agent:{uuid4()}"
    state = _followup_state(
        last_agent=rid,
        content="now do the same for this other file",
        custom_agents={rid: _custom_entry(rid)},
    )

    out = await wf._route_node(state)

    assert out["selected_agent"] == rid
    # Sticky path short-circuits before the (LLM) router runs.
    assert router.calls == 0
    trail = out["context"]["agents_invoked"]
    assert trail[-1]["id"] == rid
    assert trail[-1]["via"] == "sticky"


@pytest.mark.asyncio
async def test_followup_does_not_stick_to_base_agent():
    router = _RecordingRouter()
    wf = _workflow(router)
    rid = f"custom_agent:{uuid4()}"
    # Previous turn ended on a base agent → no stickiness; the router runs.
    state = _followup_state(
        last_agent="chat_agent",
        content="now do the same for this other file",
        custom_agents={rid: _custom_entry(rid)},
    )

    out = await wf._route_node(state)

    assert router.calls == 1
    assert out["selected_agent"] == "chat_agent"


@pytest.mark.asyncio
async def test_explicit_naming_a_different_custom_agent_breaks_stickiness():
    # Use the real router (no Gemini) so its deterministic explicit-name
    # override selects the named agent.
    router = Router.__new__(Router)
    router.gemini_client = None
    router.model_name = "x"
    wf = _workflow(router)

    rid1 = f"custom_agent:{uuid4()}"
    rid2 = f"custom_agent:{uuid4()}"
    state = _followup_state(
        last_agent=rid1,
        content="Ask the Legal Reviewer to check this clause.",
        custom_agents={
            rid1: _custom_entry(rid1, name="Analyst"),
            rid2: _custom_entry(rid2, name="Legal Reviewer", description="Reviews contracts."),
        },
    )

    out = await wf._route_node(state)

    # Stickiness yields to the explicitly named different custom agent.
    assert out["selected_agent"] == rid2


@pytest.mark.asyncio
async def test_planning_mode_with_existing_plan_bypasses_stickiness():
    router = _RecordingRouter()
    wf = _workflow(router)
    rid = f"custom_agent:{uuid4()}"
    state = _followup_state(
        last_agent=rid,
        content="continue the plan",
        custom_agents={rid: _custom_entry(rid)},
    )
    state["planning_mode_enabled"] = True
    state["has_existing_plan"] = True

    out = await wf._route_node(state)

    # Planning supervises: the sticky short-circuit must not fire.
    assert router.calls == 1
    assert out["selected_agent"] != rid


@pytest.mark.asyncio
async def test_stickiness_released_when_previous_custom_agent_detached():
    router = _RecordingRouter()
    wf = _workflow(router)
    rid = f"custom_agent:{uuid4()}"
    # last_agent points at a custom agent no longer attached to the conversation.
    state = _followup_state(
        last_agent=rid,
        content="keep going",
        custom_agents={},
    )

    out = await wf._route_node(state)

    assert router.calls == 1
    assert out["selected_agent"] == "chat_agent"


def test_finalize_persists_last_agent():
    wf = _workflow(_RecordingRouter())
    rid = f"custom_agent:{uuid4()}"
    state = {
        "selected_agent": rid,
        "custom_agents": {rid: _custom_entry(rid)},
        "context": {},
        "messages": [],
    }
    response = AgentResponse(
        agent_type=AgentType.CHAT,
        agent_id=rid,
        message=AgentMessage(role=MessageRole.ASSISTANT, content="answer"),
        metadata={"runtime_agent_id": rid, "custom_agent_name": "Analyst"},
    )

    wf._finalize_agent_response(state, response)

    assert state["last_agent"] == rid
