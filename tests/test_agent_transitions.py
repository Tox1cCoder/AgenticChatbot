"""Handoffs as LangGraph state transitions.

A handoff is a control-plane event, not a JSON string the orchestrator parses
back out of a tool result. The tool records a pending transition and its paired
``ToolMessage``; one resolver — and only that resolver — decides whether the
transition is accepted, and it never rewrites the turn's routing decision.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from app.ai.hand_off_tool import HAND_OFF_TOOL_NAME, create_hand_off_tool, handoff_message_id
from app.ai.workflow.contracts import AgentTransition, PendingTransition, RoutingDecision
from app.ai.workflow.inventory import build_routing_inventory
from app.ai.workflow.transitions import TransitionResolver

BASE_AGENTS = ["chat_agent", "search_agent", "rag_agent", "canvas_agent"]


def _inventory(custom_agents=None, disabled=None):
    return build_routing_inventory(
        base_agent_ids=BASE_AGENTS,
        custom_agents=custom_agents or {},
        disabled_agent_ids=disabled or set(),
    )


def _routed_state(*, initial="chat_agent", active="chat_agent", history=None, **overrides):
    state = {
        "routing_decision": RoutingDecision(
            agent_id=initial, confidence=0.9, reason="initial route"
        ),
        "active_agent_id": active,
        "agent_history": history
        if history is not None
        else [AgentTransition(from_agent_id=None, to_agent_id=initial, source="router")],
        "messages": [],
        "context": {},
    }
    state.update(overrides)
    return state


def _pending(source="chat_agent", target="search_agent", call_id="call-1"):
    return PendingTransition(
        from_agent_id=source,
        to_agent_id=target,
        tool_call_id=call_id,
        tool_message_id=handoff_message_id(call_id),
        reason="needs current sources",
    )


def _resolver(inventory=None, max_depth=5):
    return TransitionResolver(inventory=inventory or _inventory(), max_delegation_depth=max_depth)


def invoke_handoff(*, source: str, target: str, call_id: str, targets=None) -> Command:
    tool = create_hand_off_tool(
        source_agent_id=source, allowed_targets=targets or [target, "rag_agent"]
    )
    return tool.invoke(
        {
            "name": HAND_OFF_TOOL_NAME,
            "args": {"target_agent": target, "reason": "better suited"},
            "id": call_id,
            "type": "tool_call",
        }
    )


# ----------------------------------------------------------------------
# the tool
# ----------------------------------------------------------------------


def test_handoff_tool_returns_parent_command_with_paired_tool_message():
    command = invoke_handoff(source="chat_agent", target="search_agent", call_id="call-1")

    assert isinstance(command, Command)
    assert command.graph == Command.PARENT
    assert command.goto == "resolve_transition"

    message = command.update["messages"][0]
    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == "call-1"
    assert message.id == "handoff:call-1"
    assert message.name == HAND_OFF_TOOL_NAME


def test_handoff_tool_records_a_pending_transition_not_a_json_payload():
    command = invoke_handoff(source="chat_agent", target="search_agent", call_id="call-1")
    pending = command.update["pending_transition"]

    assert isinstance(pending, PendingTransition)
    assert pending.from_agent_id == "chat_agent"
    assert pending.to_agent_id == "search_agent"
    assert pending.tool_call_id == "call-1"
    assert pending.tool_message_id == "handoff:call-1"
    # The control decision is state, never text the orchestrator re-parses.
    assert "hand_off" not in command.update["messages"][0].content


def test_handoff_message_id_is_deterministic():
    assert handoff_message_id("call-9") == "handoff:call-9"


def test_pending_transition_cannot_be_built_without_a_tool_call_id():
    """Pairing is a contract invariant, so the resolver never sees a broken one."""
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        PendingTransition(
            from_agent_id="chat_agent",
            to_agent_id="search_agent",
            tool_call_id="",
            tool_message_id="handoff:",
            reason="no call id",
        )


def test_handoff_tool_description_lists_only_reachable_targets():
    tool = create_hand_off_tool(
        source_agent_id="chat_agent",
        allowed_targets=["search_agent"],
        target_descriptions={"search_agent": "current information"},
    )
    assert "search_agent" in tool.description
    assert "current information" in tool.description
    assert "chat_agent" not in tool.description.split("Valid targets:")[-1]


# ----------------------------------------------------------------------
# the resolver
# ----------------------------------------------------------------------


async def test_accepted_handoff_preserves_initial_decision():
    resolver = _resolver()
    state = _routed_state(initial="chat_agent", active="chat_agent")
    state["pending_transition"] = _pending(target="search_agent")

    command = await resolver(state)

    assert command.update["active_agent_id"] == "search_agent"
    assert "routing_decision" not in command.update
    assert command.goto == "search_agent"


async def test_accepted_handoff_appends_exactly_one_transition():
    resolver = _resolver()
    state = _routed_state()
    state["pending_transition"] = _pending(target="search_agent")

    command = await resolver(state)

    appended = command.update["agent_history"]
    assert len(appended) == 1
    assert appended[0] == AgentTransition(
        from_agent_id="chat_agent",
        to_agent_id="search_agent",
        source="handoff",
        tool_call_id="call-1",
    )


async def test_accepted_handoff_clears_the_pending_transition():
    resolver = _resolver()
    state = _routed_state()
    state["pending_transition"] = _pending()

    command = await resolver(state)
    assert command.update["pending_transition"] is None


async def test_custom_target_routes_to_resolved_node_name():
    inventory = _inventory(
        {"custom_agent:123": {"runtime_agent_id": "custom_agent:123", "name": "Alpha"}}
    )
    resolver = _resolver(inventory)
    state = _routed_state()
    state["pending_transition"] = _pending(target="custom_agent:123")

    command = await resolver(state)

    assert command.update["active_agent_id"] == "custom_agent:123"
    assert command.goto == inventory.resolve_node("custom_agent:123")
    assert command.goto == "custom_agent"


@pytest.mark.parametrize(
    "case", ["self", "cycle", "detached", "unknown", "over_depth", "source_mismatch"]
)
async def test_rejected_handoff_returns_to_source_with_paired_feedback(case):
    inventory = _inventory({"custom_agent:detached": {"runtime_agent_id": "custom_agent:detached"}})
    state = _routed_state()

    if case == "self":
        state["pending_transition"] = _pending(target="chat_agent")
        resolver = _resolver(inventory)
    elif case == "cycle":
        state = _routed_state(
            active="search_agent",
            history=[
                AgentTransition(from_agent_id=None, to_agent_id="chat_agent", source="router"),
                AgentTransition(
                    from_agent_id="chat_agent",
                    to_agent_id="search_agent",
                    source="handoff",
                    tool_call_id="call-0",
                ),
            ],
        )
        state["pending_transition"] = _pending(source="search_agent", target="chat_agent")
        resolver = _resolver(inventory)
    elif case == "detached":
        state["pending_transition"] = _pending(target="custom_agent:missing")
        resolver = _resolver(inventory)
    elif case == "unknown":
        state["pending_transition"] = _pending(target="nope_agent")
        resolver = _resolver(inventory)
    elif case == "over_depth":
        state = _routed_state(
            history=[
                AgentTransition(from_agent_id=None, to_agent_id="chat_agent", source="router"),
                AgentTransition(
                    from_agent_id="chat_agent",
                    to_agent_id="rag_agent",
                    source="handoff",
                    tool_call_id="call-0",
                ),
            ]
        )
        state["pending_transition"] = _pending(target="search_agent")
        resolver = _resolver(inventory, max_depth=1)
    else:  # source_mismatch
        state["pending_transition"] = _pending(source="rag_agent", target="search_agent")
        resolver = _resolver(inventory)

    command = await resolver(state)

    # A refusal returns control to whoever asked, not to the requested target.
    assert command.goto == state["active_agent_id"]
    assert command.update["pending_transition"] is None
    assert "active_agent_id" not in command.update
    message = command.update["messages"][0]
    assert isinstance(message, ToolMessage)
    assert message.content.startswith("Hand-off refused")


async def test_rejection_reuses_the_request_marker_id_instead_of_duplicating():
    """A rejection replaces the request marker; it does not append a second one.

    The reducer keys on message id, so reusing the deterministic id is what
    keeps exactly one tool message paired with the originating call.
    """
    resolver = _resolver()
    state = _routed_state()
    state["pending_transition"] = _pending(target="chat_agent")

    command = await resolver(state)

    assert command.update["messages"][0].id == "handoff:call-1"
    assert len([m for m in command.update["messages"] if m.tool_call_id == "call-1"]) == 1


async def test_rejection_feedback_names_the_reason():
    resolver = _resolver()
    state = _routed_state()
    state["pending_transition"] = _pending(target="nope_agent")

    command = await resolver(state)
    assert "not a reachable target" in command.update["messages"][0].content


async def test_over_depth_rejection_names_the_configured_limit():
    state = _routed_state(
        history=[
            AgentTransition(from_agent_id=None, to_agent_id="chat_agent", source="router"),
            AgentTransition(
                from_agent_id="chat_agent",
                to_agent_id="rag_agent",
                source="handoff",
                tool_call_id="call-0",
            ),
        ]
    )
    state["pending_transition"] = _pending(target="search_agent")

    command = await _resolver(max_depth=1)(state)
    assert "maximum delegation depth of 1" in command.update["messages"][0].content


async def test_only_accepted_handoffs_consume_transition_depth():
    """A resume transition is audit history, not a delegation."""
    state = _routed_state(
        history=[
            AgentTransition(from_agent_id=None, to_agent_id="chat_agent", source="router"),
            AgentTransition(from_agent_id="chat_agent", to_agent_id="chat_agent", source="resume"),
        ]
    )
    state["pending_transition"] = _pending(target="search_agent")

    command = await _resolver(max_depth=1)(state)
    assert command.update["active_agent_id"] == "search_agent"


async def test_missing_pending_transition_fails_closed():
    command = await _resolver()(_routed_state())
    assert command.goto == "finalize"
    assert command.update["workflow_error"].code == "response_validation_failed"


async def test_resolver_compares_only_control_plane_identifiers():
    """No message text is interpreted; identity comparison decides everything."""
    import ast
    import pathlib

    source = pathlib.Path("app/ai/workflow/transitions.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "lower" not in called
    assert "loads" not in called
    assert "json" not in {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
