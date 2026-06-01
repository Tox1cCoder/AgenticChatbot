"""Graph-level / custom-agent runtime adapter tests."""

from __future__ import annotations

from uuid import uuid4

from app.ai.agents.custom_agent import CustomAgent
from app.ai.custom_agent_runtime import build_custom_agent_runtime_spec
from app.ai.schemas import WorkflowExecutionRequest


def _spec(name="Data Analyst", skill_refs=None):
    cid = uuid4()
    return build_custom_agent_runtime_spec(
        {
            "id": str(cid),
            "runtime_agent_id": f"custom_agent:{cid}",
            "model_agent_key": "custom",
            "name": name,
            "description": "Answers analytics questions.",
            "prompt": "You are a precise data analyst.",
            "model_request": {
                "provider_type": "openai",
                "model": "gpt-4.1-mini",
                "temperature": 0.2,
                "reasoning_effort": None,
            },
            "tool_refs": [],
            "skill_refs": skill_refs or [],
        }
    )


def test_custom_agent_response_metadata_contains_custom_identity():
    spec = _spec()
    agent = CustomAgent(spec)

    metadata: dict = {"conversation_id": "c1"}
    agent.set_runtime_warnings(["Selected client tool 'x' is unavailable and was skipped."])
    agent._augment_response_metadata(metadata)

    assert metadata["runtime_agent_id"] == spec.runtime_agent_id
    assert metadata["custom_agent_id"] == str(spec.custom_agent_id)
    assert metadata["custom_agent_name"] == "Data Analyst"
    assert metadata["custom_agent_warnings"] == [
        "Selected client tool 'x' is unavailable and was skipped."
    ]
    # The response agent_id is the runtime id; agent_type stays CHAT-compatible.
    assert agent.agent_id == spec.runtime_agent_id
    assert agent.agent_type.value == "chat"


def test_custom_agent_resolves_its_own_model_request():
    spec = _spec()
    agent = CustomAgent(spec)
    # Incoming per-base-agent map is ignored; the custom agent uses its own.
    resolved = agent._resolve_model_request({"chat": {"provider_type": "gemini", "model": "g"}})
    assert resolved == spec.model_request
    assert resolved["provider_type"] == "openai"


# --------------------------------------------------------------------------- #
# Static graph multiplexer routing (Task 9)
# --------------------------------------------------------------------------- #


from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from app.ai.graph import MultiAgentWorkflow  # noqa: E402

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
    return wf


def _custom_state(runtime_id, *, messages=None, iteration_count=0):
    return {
        "selected_agent": runtime_id,
        "custom_agents": {
            runtime_id: {
                "id": runtime_id.split(":", 1)[1],
                "runtime_agent_id": runtime_id,
                "name": "Analyst",
                "prompt": "p",
                "model_request": {"provider_type": "openai", "model": "gpt-4.1-mini"},
                "tool_refs": [],
                "skill_refs": [],
            }
        },
        "messages": messages or [],
        "iteration_count": iteration_count,
        "context": {},
    }


def test_no_custom_agents_keeps_existing_chat_path():
    wf = _workflow()
    # Base routing unchanged.
    assert wf._should_continue({"selected_agent": "chat_agent"}) == "chat_agent"
    assert wf._should_continue({"selected_agent": "rag_agent"}) == "rag_agent"
    # A custom id with NO attached custom agents falls through to end (no change
    # to behavior for conversations without custom agents).
    assert wf._should_continue({"selected_agent": "custom_agent:abc", "custom_agents": {}}) == "end"
    assert wf._should_continue({"selected_agent": "unknown_agent"}) == "end"


def test_initial_state_injects_attached_custom_agents():
    wf = _workflow()
    rid = f"custom_agent:{uuid4()}"

    state = wf._build_initial_state_from_request(
        WorkflowExecutionRequest(
            message="ask Analyst for help",
            custom_agents={
                rid: {
                    "id": rid.split(":", 1)[1],
                    "runtime_agent_id": rid,
                    "name": "Analyst",
                    "prompt": "p",
                    "model_request": {"provider_type": "openai", "model": "gpt-4.1-mini"},
                    "tool_refs": [],
                    "skill_refs": [],
                }
            },
        )
    )

    assert state["custom_agents"][rid]["name"] == "Analyst"


def test_should_continue_routes_custom_runtime_id_to_static_custom_node():
    wf = _workflow()
    rid = f"custom_agent:{uuid4()}"
    state = _custom_state(rid)
    assert wf._should_continue(state) == "custom_agent"
    # Unattached custom id is rejected.
    assert (
        wf._should_continue({"selected_agent": f"custom_agent:{uuid4()}", "custom_agents": {}})
        == "end"
    )


def test_custom_agent_react_loop_routes_back_to_custom_node_until_done():
    wf = _workflow()
    rid = f"custom_agent:{uuid4()}"

    # After a tool executed (last message is a ToolMessage), the loop routes
    # back to the static custom node while under the iteration budget.
    state = _custom_state(
        rid,
        messages=[
            HumanMessage(content="hi"),
            AIMessage(content="", tool_calls=[{"id": "t1", "name": "tool_search", "args": {}}]),
            ToolMessage(content="result", tool_call_id="t1", name="tool_search"),
        ],
        iteration_count=0,
    )
    assert wf._route_tool_output(state) == "custom_agent"

    # The loop terminates when the custom agent emits no further tool calls.
    done_state = {
        "selected_agent": rid,
        "messages": [
            HumanMessage(content="hi"),
            AIMessage(content="Final answer."),
        ],
    }
    assert wf._should_call_tools(done_state) == "end"
    # And a tool-calling response routes into the tools node.
    tool_state = {
        "selected_agent": rid,
        "messages": [
            AIMessage(content="", tool_calls=[{"id": "t2", "name": "tool_search", "args": {}}]),
        ],
    }
    assert wf._should_call_tools(tool_state) == "tools"


# --------------------------------------------------------------------------- #
# Dynamic router + dynamic handoff (Task 10)
# --------------------------------------------------------------------------- #

import json  # noqa: E402

import pytest  # noqa: E402

from app.ai.agents.router import Router  # noqa: E402
from app.ai.schemas import AgentMessage, MessageRole  # noqa: E402


def _handoff_output(target, tool_call_id="h1"):
    return [
        {
            "name": "hand_off",
            "content": json.dumps({"hand_off": target, "reason": "delegating"}),
            "tool_call_id": tool_call_id,
        }
    ]


def _multi_custom_state(selected, ids):
    return {
        "selected_agent": selected,
        "delegation_count": 0,
        "messages": [HumanMessage(content="hi")],
        "context": {},
        "custom_agents": {
            cid: {"id": cid.split(":", 1)[1], "runtime_agent_id": cid, "name": f"A{i}"}
            for i, cid in enumerate(ids)
        },
    }


@pytest.mark.asyncio
async def test_router_can_select_attached_custom_agent_by_name():
    router = Router.__new__(Router)
    router.gemini_client = None  # force deterministic-only path
    router.model_name = "x"

    rid = f"custom_agent:{uuid4()}"
    descriptors = [
        {
            "runtime_agent_id": rid,
            "name": "Data Analyst",
            "description": "analytics",
            "agent_order": 0,
        }
    ]
    msg = AgentMessage(
        role=MessageRole.USER,
        content="Please ask the Data Analyst to summarize this.",
        metadata={},
    )
    result = await router.route_message(
        msg, ["chat_agent", "rag_agent", rid], custom_agent_descriptors=descriptors
    )
    assert result == rid


def test_base_agent_can_handoff_to_custom_agent():
    wf = _workflow()
    rid = f"custom_agent:{uuid4()}"
    state = _multi_custom_state("chat_agent", [rid])
    new_state = wf._apply_hand_off_if_present(state, _handoff_output(rid))
    assert new_state["selected_agent"] == rid


def test_custom_agent_can_handoff_to_base_agent():
    wf = _workflow()
    rid = f"custom_agent:{uuid4()}"
    state = _multi_custom_state(rid, [rid])
    new_state = wf._apply_hand_off_if_present(state, _handoff_output("search_agent"))
    assert new_state["selected_agent"] == "search_agent"


def test_custom_agent_can_handoff_to_another_attached_custom_agent():
    wf = _workflow()
    rid1 = f"custom_agent:{uuid4()}"
    rid2 = f"custom_agent:{uuid4()}"
    state = _multi_custom_state(rid1, [rid1, rid2])
    new_state = wf._apply_hand_off_if_present(state, _handoff_output(rid2))
    assert new_state["selected_agent"] == rid2


def test_no_custom_agents_regression_across_paths():
    """With no custom agents, routing/handoff/request/display behave as before."""
    from app.ai.schemas import WorkflowExecutionRequest as AIWorkflowExecutionRequest
    from app.schemas.workflow import WorkflowExecutionRequest

    wf = _workflow()
    # Routing: base agents unchanged, no custom node reachable.
    for base in ("chat_agent", "rag_agent", "search_agent", "canvas_agent", "planning_agent"):
        assert wf._should_continue({"selected_agent": base, "custom_agents": {}}) == base
    # Handoff to a base target still works without any custom agents.
    state = {
        "selected_agent": "chat_agent",
        "delegation_count": 0,
        "messages": [HumanMessage(content="hi")],
        "context": {},
        "custom_agents": {},
    }
    out = wf._apply_hand_off_if_present(state, _handoff_output("search_agent"))
    assert out["selected_agent"] == "search_agent"
    # Workflow requests default custom_agents to empty for both schemas.
    assert WorkflowExecutionRequest(message="x").custom_agents == {}
    assert AIWorkflowExecutionRequest(message="x").custom_agents == {}


def test_handoff_to_unattached_custom_agent_is_refused_with_tool_error():
    wf = _workflow()
    rid = f"custom_agent:{uuid4()}"
    bogus = f"custom_agent:{uuid4()}"
    state = _multi_custom_state("chat_agent", [rid])
    new_state = wf._apply_hand_off_if_present(state, _handoff_output(bogus))
    # Selection unchanged; a structured tool error was appended.
    assert new_state["selected_agent"] == "chat_agent"
    last = new_state["messages"][-1]
    assert isinstance(last, ToolMessage)
    assert "not a valid target" in last.content
