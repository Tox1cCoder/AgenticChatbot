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


def test_custom_agent_prompt_mentions_missing_device_capabilities():
    agent = CustomAgent(_spec())
    agent.set_runtime_warnings(["Selected skill 'kobo-library' is not available on this device."])

    prompt = agent._build_system_prompt(None, False)

    assert "DEVICE CAPABILITY NOTICE" in prompt
    assert "kobo-library" in prompt
    # Implementation emits the sentence capitalized (see T006 progress note).
    assert "Continue with available capabilities" in prompt


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
    wf._runtime_model_resolver = None
    wf._model_usage_recorder = None
    return wf


def _custom_state(runtime_id, *, messages=None, iteration_count=0):
    return {
        "active_agent_id": runtime_id,
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
    assert wf._should_continue({"active_agent_id": "chat_agent"}) == "chat_agent"
    assert wf._should_continue({"active_agent_id": "rag_agent"}) == "rag_agent"
    # A custom id with NO attached custom agents falls through to end (no change
    # to behavior for conversations without custom agents).
    assert (
        wf._should_continue({"active_agent_id": "custom_agent:abc", "custom_agents": {}}) == "end"
    )
    assert wf._should_continue({"active_agent_id": "unknown_agent"}) == "end"


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
        wf._should_continue({"active_agent_id": f"custom_agent:{uuid4()}", "custom_agents": {}})
        == "end"
    )


async def test_custom_agent_react_loop_routes_back_to_custom_node_until_done():
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
        "active_agent_id": rid,
        "messages": [
            HumanMessage(content="hi"),
            AIMessage(content="Final answer."),
        ],
    }
    assert await wf._should_call_tools(done_state) == "end"
    # And a tool-calling response routes into the tools node.
    tool_state = {
        "active_agent_id": rid,
        "messages": [
            AIMessage(content="", tool_calls=[{"id": "t2", "name": "tool_search", "args": {}}]),
        ],
    }
    assert await wf._should_call_tools(tool_state) == "tools"


# --------------------------------------------------------------------------- #
# Dynamic router + dynamic handoff (Task 10)
# --------------------------------------------------------------------------- #

import json  # noqa: E402

import pytest  # noqa: E402

from app.ai.agents.router import Router  # noqa: E402
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole  # noqa: E402
from app.core.config import settings  # noqa: E402


def _handoff_output(target, tool_call_id="h1"):
    return [
        {
            "name": "hand_off",
            "content": json.dumps({"hand_off": target}),
            "tool_call_id": tool_call_id,
        }
    ]


def _multi_custom_state(selected, ids):
    return {
        "active_agent_id": selected,
        "delegation_count": 0,
        "messages": [HumanMessage(content="hi")],
        "context": {},
        "custom_agents": {
            cid: {"id": cid.split(":", 1)[1], "runtime_agent_id": cid, "name": f"A{i}"}
            for i, cid in enumerate(ids)
        },
    }


@pytest.mark.asyncio
async def test_attached_custom_agent_is_a_routable_target_chosen_by_the_model():
    """The model selects a custom agent from the inventory.

    There is no explicit-name matcher: naming an agent in the message must not
    select it in Python. It only makes the model's choice more likely.
    """
    from app.ai.workflow.contracts import RoutingDecision

    rid = f"custom_agent:{uuid4()}"
    descriptors = [
        {
            "runtime_agent_id": rid,
            "name": "Data Analyst",
            "description": "analytics",
            "agent_order": 0,
        }
    ]

    class _Service:
        def __init__(self):
            self.last_inventory = None

        async def route(self, context, inventory, *, user_id, model_request, request_id):
            self.last_inventory = inventory
            return RoutingDecision(agent_id=rid, confidence=0.9, reason="analytics specialist")

    service = _Service()
    router = Router(recorder=None, routing_service=service)
    msg = AgentMessage(
        role=MessageRole.USER,
        content="Please ask the Data Analyst to summarize this.",
        metadata={},
    )

    result = await router.route_message(
        msg, ["chat_agent", "rag_agent", rid], custom_agent_descriptors=descriptors
    )

    assert result == rid
    assert service.last_inventory.is_routable(rid) is True
    assert service.last_inventory.resolve_node(rid) == "custom_agent"


@pytest.mark.asyncio
async def test_naming_an_agent_does_not_select_it_without_the_model():
    """A message naming an agent still routes wherever the model decides."""
    from app.ai.workflow.contracts import RoutingDecision

    rid = f"custom_agent:{uuid4()}"

    class _Service:
        async def route(self, context, inventory, *, user_id, model_request, request_id):
            return RoutingDecision(agent_id="chat_agent", confidence=0.6, reason="general help")

    router = Router(recorder=None, routing_service=_Service())
    result = await router.route_message(
        AgentMessage(role=MessageRole.USER, content=f"use {rid} now", metadata={}),
        ["chat_agent", rid],
        custom_agent_descriptors=[{"runtime_agent_id": rid, "name": "Data Analyst"}],
    )
    assert result == "chat_agent"


def test_base_agent_can_handoff_to_custom_agent():
    wf = _workflow()
    rid = f"custom_agent:{uuid4()}"
    state = _multi_custom_state("chat_agent", [rid])
    new_state = wf._apply_hand_off_if_present(state, _handoff_output(rid))
    assert new_state["active_agent_id"] == rid
    assert new_state["context"]["handoff"]["source_agent"] == "chat_agent"
    assert new_state["context"]["handoff"]["target_agent"] == rid


def test_custom_agent_can_handoff_to_base_agent():
    wf = _workflow()
    rid = f"custom_agent:{uuid4()}"
    state = _multi_custom_state(rid, [rid])
    new_state = wf._apply_hand_off_if_present(state, _handoff_output("search_agent"))
    assert new_state["active_agent_id"] == "search_agent"
    assert new_state["context"]["handoff"]["source_agent"] == rid
    assert new_state["context"]["handoff"]["target_agent"] == "search_agent"


def test_custom_agent_can_handoff_to_another_attached_custom_agent():
    wf = _workflow()
    rid1 = f"custom_agent:{uuid4()}"
    rid2 = f"custom_agent:{uuid4()}"
    state = _multi_custom_state(rid1, [rid1, rid2])
    new_state = wf._apply_hand_off_if_present(state, _handoff_output(rid2))
    assert new_state["active_agent_id"] == rid2


def test_custom_agent_handoff_targets_are_dynamic_graph_targets():
    wf = _workflow()
    rid = f"custom_agent:{uuid4()}"
    other = f"custom_agent:{uuid4()}"
    state = _multi_custom_state(rid, [rid, other])

    assert wf._custom_handoff_targets(state, rid) == [*wf.agents.keys(), other]


def test_custom_agent_handoff_tool_describes_attached_custom_targets():
    wf = _workflow()
    rid1 = f"custom_agent:{uuid4()}"
    rid2 = f"custom_agent:{uuid4()}"
    state = _multi_custom_state(rid1, [rid1, rid2])
    state["custom_agents"][rid2]["name"] = "Legal Reviewer"
    state["custom_agents"][rid2]["description"] = "Reviews contract and policy questions."

    agent = wf._build_custom_agent(state, rid1)

    assert agent is not None
    handoff_tool = next(
        tool
        for tool in agent.restricted_internal_tools(user_id=None, device_id=None)
        if getattr(tool, "name", None) == "hand_off"
    )
    description = handoff_tool.description
    assert rid2 in description
    assert "Legal Reviewer" in description
    assert "Reviews contract and policy questions." in description
    assert "hand off to planning_agent" not in description
    assert "coordinate the work" not in description


def test_custom_handoff_target_descriptions_include_base_agent_capabilities():
    """A custom agent with a limited toolset cannot pick a suitable delegate
    unless it knows what each base agent is good at. Base targets must carry
    capability blurbs, not appear as bare ids."""
    wf = _workflow()
    rid = f"custom_agent:{uuid4()}"
    state = _multi_custom_state(rid, [rid])

    descriptions = wf._custom_handoff_target_descriptions(state, rid)

    # Every base agent (except the delegating agent itself) is described.
    for base in wf.agents:
        assert base in descriptions
        assert descriptions[base].strip()
    # The blurb is capability-bearing (not just the bare id): search_agent
    # mentions current/news/web info so the LLM can recognise the fit.
    search_blurb = descriptions["search_agent"].lower()
    assert search_blurb != "search_agent"
    assert "current" in search_blurb or "news" in search_blurb or "web" in search_blurb
    # The delegating custom agent never describes itself.
    assert rid not in descriptions


def test_custom_agent_handoff_tool_describes_base_agent_capabilities():
    """The rendered hand_off tool description must surface base-agent
    capabilities so the LLM can route stuck work to the right specialist."""
    wf = _workflow()
    rid = f"custom_agent:{uuid4()}"
    state = _multi_custom_state(rid, [rid])

    agent = wf._build_custom_agent(state, rid)
    assert agent is not None
    handoff_tool = next(
        tool
        for tool in agent.restricted_internal_tools(user_id=None, device_id=None)
        if getattr(tool, "name", None) == "hand_off"
    )
    description = handoff_tool.description
    assert "search_agent" in description
    # A distinctive capability word (absent from the agent id and the static
    # tool doc) proves the base-agent blurb actually rendered.
    low = description.lower()
    assert "news" in low or "fact-check" in low


def test_custom_agent_delegation_prompt_encourages_delegation_on_capability_gap():
    """The delegation suffix must tell a limited-toolset agent to hand off when
    it lacks a tool/capability or is not making progress — instead of looping
    on tool calls it cannot complete."""
    wf = _workflow()
    rid = f"custom_agent:{uuid4()}"
    state = _multi_custom_state(rid, [rid])

    agent = wf._build_custom_agent(state, rid)
    assert agent is not None
    suffix = agent._build_delegation_suffix()
    low = suffix.lower()

    assert "hand_off" in suffix
    # Encourages delegating on a capability/tool gap rather than grinding.
    assert "tool" in low
    assert "retr" in low or "progress" in low or "stuck" in low
    # Still surfaces capability-bearing base targets.
    assert "search_agent" in suffix


def test_custom_agent_system_prompt_describes_dynamic_handoff_targets():
    wf = _workflow()
    rid1 = f"custom_agent:{uuid4()}"
    rid2 = f"custom_agent:{uuid4()}"
    state = _multi_custom_state(rid1, [rid1, rid2])
    state["custom_agents"][rid2]["name"] = "Legal Reviewer"
    state["custom_agents"][rid2]["description"] = "Reviews contract and policy questions."

    agent = wf._build_custom_agent(state, rid1)

    assert agent is not None
    prompt = agent._build_system_prompt(None, False, include_hand_off=True)
    assert rid2 in prompt
    assert "Legal Reviewer" in prompt
    assert "Reviews contract and policy questions." in prompt
    assert "hand off to `planning_agent`" not in prompt
    assert "coordinate the work" not in prompt


def test_no_custom_agents_regression_across_paths():
    """With no custom agents, routing/handoff/request/display behave as before."""
    from app.ai.schemas import WorkflowExecutionRequest as AIWorkflowExecutionRequest
    from app.schemas.workflow import WorkflowExecutionRequest

    wf = _workflow()
    # Routing: base agents unchanged, no custom node reachable.
    for base in ("chat_agent", "rag_agent", "search_agent", "canvas_agent", "planning_agent"):
        assert wf._should_continue({"active_agent_id": base, "custom_agents": {}}) == base
    # Handoff to a base target still works without any custom agents.
    state = {
        "active_agent_id": "chat_agent",
        "delegation_count": 0,
        "messages": [HumanMessage(content="hi")],
        "context": {},
        "custom_agents": {},
    }
    out = wf._apply_hand_off_if_present(state, _handoff_output("search_agent"))
    assert out["active_agent_id"] == "search_agent"
    # Workflow requests default custom_agents to empty for both schemas.
    assert WorkflowExecutionRequest(message="x").custom_agents == {}
    assert AIWorkflowExecutionRequest(message="x").custom_agents == {}


def test_handoff_to_unattached_custom_agent_rewrites_its_single_tool_result():
    wf = _workflow()
    rid = f"custom_agent:{uuid4()}"
    bogus = f"custom_agent:{uuid4()}"
    state = _multi_custom_state("chat_agent", [rid])
    outputs = _handoff_output(bogus)

    new_state = wf._apply_hand_off_if_present(state, outputs)

    # Selection remains unchanged and the interpreter has not appended a
    # duplicate message. The normal tool-output writer creates the sole pair.
    assert new_state["active_agent_id"] == "chat_agent"
    assert len(new_state["messages"]) == 1
    wf._apply_tool_outputs_to_state(new_state, tool_outputs=outputs)
    handoff_messages = [message for message in new_state["messages"] if message.name == "hand_off"]
    assert len(handoff_messages) == 1
    last = handoff_messages[0]
    assert isinstance(last, ToolMessage)
    assert "not a reachable target" in last.content


@pytest.mark.parametrize(
    ("outputs", "expected_error"),
    [
        (
            [
                *_handoff_output("search_agent", tool_call_id="h1"),
                *_handoff_output("planning_agent", tool_call_id="h2"),
            ],
            "exactly one hand_off call",
        ),
        (
            [
                {
                    "name": "hand_off",
                    "content": '{"hand_off": "search_agent", "reason": "legacy"}',
                    "tool_call_id": "h1",
                }
            ],
            "exactly one target agent",
        ),
    ],
)
def test_handoff_rewrites_noncanonical_or_multiple_outputs(outputs, expected_error):
    wf = _workflow()
    state = _multi_custom_state("chat_agent", [])

    wf._apply_hand_off_if_present(state, outputs)

    assert state["active_agent_id"] == "chat_agent"
    assert all(expected_error in output["content"] for output in outputs)


def test_handoff_rejects_self_and_revisited_targets():
    wf = _workflow()
    state = _multi_custom_state("chat_agent", [])
    self_output = _handoff_output("chat_agent")

    wf._apply_hand_off_if_present(state, self_output)

    assert "not a reachable target" in self_output[0]["content"]
    state["context"]["agents_invoked"] = [{"id": "search_agent"}]
    repeated_output = _handoff_output("search_agent")
    wf._apply_hand_off_if_present(state, repeated_output)

    assert "already handled this turn" in repeated_output[0]["content"]


def test_handoff_rejects_when_configured_depth_is_exhausted(monkeypatch):
    wf = _workflow()
    state = _multi_custom_state("chat_agent", [])
    # Delegation depth is turn-scoped context, not a top-level state field.
    state["context"] = {**(state.get("context") or {}), "delegation_count": 1}
    outputs = _handoff_output("search_agent")
    monkeypatch.setattr(settings, "max_handoff_delegation_depth", 1)

    wf._apply_hand_off_if_present(state, outputs)

    assert state["active_agent_id"] == "chat_agent"
    assert "maximum delegation depth of 1" in outputs[0]["content"]


# --------------------------------------------------------------------------- #
# Canonical agent metadata on final responses (Task 2)
# --------------------------------------------------------------------------- #


def test_finalize_response_adds_canonical_custom_agent_metadata():
    wf = _workflow()
    rid = f"custom_agent:{uuid4()}"
    state = _multi_custom_state(rid, [rid])
    response = AgentResponse(
        agent_type=AgentType.CHAT,
        agent_id=rid,
        message=AgentMessage(role=MessageRole.ASSISTANT, content="answer"),
        metadata={"runtime_agent_id": rid, "custom_agent_name": "A0"},
    )

    wf._attach_final_agent_metadata(state, response)

    assert response.metadata["agent"] == {
        "id": rid,
        "kind": "custom",
        "name": "A0",
        "custom_agent_id": rid.split(":", 1)[1],
        "source": "response",
    }


def test_finalize_response_adds_handoff_metadata():
    wf = _workflow()
    rid = f"custom_agent:{uuid4()}"
    state = _multi_custom_state("chat_agent", [rid])
    wf._apply_hand_off_if_present(state, _handoff_output(rid, tool_call_id="h1"))

    response = AgentResponse(
        agent_type=AgentType.CHAT,
        agent_id=rid,
        message=AgentMessage(role=MessageRole.ASSISTANT, content="answer"),
        metadata={},
    )

    wf._attach_final_agent_metadata(state, response)

    assert response.metadata["handoff"]["from_agent_id"] == "chat_agent"
    assert response.metadata["handoff"]["to_agent_id"] == rid
    assert "reason" not in response.metadata["handoff"]


def test_recover_terminal_response_adds_canonical_agent_metadata():
    wf = _workflow()
    rid = f"custom_agent:{uuid4()}"
    state = _multi_custom_state(rid, [rid])
    state["messages"].append(AIMessage(content="answer"))

    response = wf._recover_terminal_response(state)

    assert response is not None
    assert response.metadata["agent"]["id"] == rid
    assert response.metadata["agent"]["name"] == "A0"


# --------------------------------------------------------------------------- #
# Issue 1: base agents must be able to hand off to attached custom agents
# --------------------------------------------------------------------------- #


def test_base_agent_multi_agent_kwargs_inject_custom_handoff_tool():
    """When custom agents are attached, a base agent's invocation must receive a
    dynamic hand_off tool that lists the attached custom agent as a valid
    target — otherwise it can only ever delegate to other base agents."""
    wf = _workflow()
    rid = f"custom_agent:{uuid4()}"
    state = _multi_custom_state("chat_agent", [rid])
    state["custom_agents"][rid]["name"] = "Legal Reviewer"
    state["custom_agents"][rid]["description"] = "Reviews contract and policy questions."

    kwargs = wf._multi_agent_kwargs(state, "chat_agent")

    internal = kwargs.get("internal_tools") or []
    handoff = next((t for t in internal if getattr(t, "name", None) == "hand_off"), None)
    assert handoff is not None, "base agent should get a dynamic hand_off tool"
    assert rid in handoff.description
    assert "Legal Reviewer" in handoff.description
    # The prompt-side target descriptions must also carry the custom target.
    assert rid in kwargs.get("handoff_target_descriptions", {})
    # A base agent never offers itself as a target.
    assert "chat_agent" not in handoff.description.split("Valid targets:", 1)[-1]


def test_base_agent_multi_agent_kwargs_are_dynamic_without_custom_agents():
    """Base agents always receive a live roster, not a static handoff tool."""
    wf = _workflow()
    state = {"active_agent_id": "chat_agent", "custom_agents": {}, "context": {}}

    kwargs = wf._multi_agent_kwargs(state, "chat_agent")

    handoff = next(
        tool for tool in kwargs["internal_tools"] if getattr(tool, "name", None) == "hand_off"
    )
    assert "chat_agent" not in handoff.description.split("Valid targets:", 1)[-1]
    assert "search_agent" in kwargs["handoff_target_descriptions"]


def test_custom_node_multi_agent_kwargs_skip_tool_injection():
    """The custom agent builds its own dynamic hand_off from its spec, so the
    graph must not also inject one (only the awareness block)."""
    wf = _workflow()
    rid = f"custom_agent:{uuid4()}"
    state = _multi_custom_state(rid, [rid])

    kwargs = wf._multi_agent_kwargs(state, rid)

    assert "internal_tools" not in kwargs
    assert "multi_agent_activity" in kwargs


# --------------------------------------------------------------------------- #
# Issue 2: agent awareness — identity, roster, and per-turn invocation trail
# --------------------------------------------------------------------------- #


def test_build_multi_agent_activity_block_has_identity_and_roster():
    wf = _workflow()
    rid = f"custom_agent:{uuid4()}"
    state = _multi_custom_state("chat_agent", [rid])
    state["custom_agents"][rid]["name"] = "Legal Reviewer"

    block = wf._build_multi_agent_activity_block(state, "chat_agent")

    assert block is not None
    # Identity of the active agent.
    assert "Chat Agent" in block
    # Roster includes the attached custom agent and at least one base specialist.
    assert "Legal Reviewer" in block
    assert "search_agent" in block


def test_agent_trail_records_router_selection_and_handoff():
    wf = _workflow()
    rid = f"custom_agent:{uuid4()}"
    state = _multi_custom_state("chat_agent", [rid])
    state["custom_agents"][rid]["name"] = "Legal Reviewer"

    # Simulate the router having selected chat_agent for this turn.
    wf._record_agent_invocation(state, "chat_agent", via="router")
    # Then chat_agent hands off to the custom agent.
    wf._apply_hand_off_if_present(state, _handoff_output(rid))

    trail = state["context"]["agents_invoked"]
    assert [e["id"] for e in trail] == ["chat_agent", rid]
    assert trail[0]["via"] == "router"
    assert trail[1]["via"] == "handoff"
    assert trail[1]["name"] == "Legal Reviewer"


def test_activity_block_reflects_handoff_trail():
    wf = _workflow()
    rid = f"custom_agent:{uuid4()}"
    state = _multi_custom_state(rid, [rid])
    state["custom_agents"][rid]["name"] = "Legal Reviewer"
    wf._record_agent_invocation(state, "chat_agent", via="router")
    wf._record_agent_invocation(state, rid, via="handoff")

    block = wf._build_multi_agent_activity_block(state, rid)

    assert block is not None
    # The active custom agent can now see who was involved this turn.
    assert "Chat Agent" in block
    assert "Legal Reviewer" in block
