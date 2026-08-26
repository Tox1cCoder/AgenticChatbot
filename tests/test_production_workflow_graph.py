"""Routing-v2 parent graph contract.

The parent graph owns exactly three things: it routes once per new turn, it
moves execution between specialists, and it terminates through one validated
finalizer. Nothing else may reach ``END``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langgraph.types import Command

from app.ai.workflow.contracts import (
    AgentTransition,
    RoutingDecision,
    TurnIdentity,
    WorkflowError,
    WorkflowRoutingException,
)
from app.ai.workflow.graph_builder import (
    SPECIALIST_NODE_NAMES,
    build_workflow_graph,
    make_route_node,
)
from app.ai.workflow.inventory import build_routing_inventory
from app.ai.workflow.state import build_checkpoint_thread_id

BASE_AGENT_IDS = [
    "chat_agent",
    "rag_agent",
    "search_agent",
    "image_generator_agent",
    "planning_agent",
    "canvas_agent",
]


def _inventory(custom_agents=None):
    return build_routing_inventory(
        base_agent_ids=BASE_AGENT_IDS, custom_agents=custom_agents or {}
    )


def _turn_identity(conversation_id="conversation-1", turn_id="message-1") -> TurnIdentity:
    return TurnIdentity(
        request_id="request-1",
        turn_id=turn_id,
        checkpoint_thread_id=build_checkpoint_thread_id(conversation_id, turn_id),
    )


class FakeWorkflow:
    """Minimal stand-in exposing only what the graph builder consumes."""

    def __init__(self):
        self.agents = dict.fromkeys(BASE_AGENT_IDS)
        self.calls: list[str] = []

        async def _noop(state):
            return state

        for attribute in (
            "_chat_node",
            "_rag_node",
            "_search_node",
            "_image_generator_node",
            "_planning_node",
            "_canvas_node",
            "_custom_agent_node",
            "_tool_node",
            "_approval_node",
            "_rag_tools_node",
            "_planning_tools_node",
        ):
            setattr(self, attribute, _noop)

        async def _end(state):
            return "end"

        self._should_call_tools = _end
        self._should_call_rag_tools = lambda state: "end"
        self._should_call_planning_tools = lambda state: "end"
        self._should_continue_rag = lambda state: "end"
        self._should_continue_planning = lambda state: "end"
        self._route_tool_output = lambda state: "end"


@pytest.fixture
def compiled_graph():
    return build_workflow_graph(FakeWorkflow(), checkpointer=None)


def test_only_finalizer_reaches_end(compiled_graph):
    graph = compiled_graph.get_graph()
    end_sources = {edge.source for edge in graph.edges if edge.target == "__end__"}
    assert end_sources == {"finalize"}


def test_compatibility_specialists_still_end_through_finalizer(compiled_graph):
    graph = compiled_graph.get_graph()
    assert not any(
        edge.target == "__end__" and edge.source != "finalize" for edge in graph.edges
    )


def test_specialist_nodes_have_no_static_outgoing_edges(compiled_graph):
    """A node returning a dynamic Command must not also carry a static edge.

    Its declared destinations render as conditional edges, which is what makes
    the dynamic topology inspectable; a *static* edge alongside a Command is
    what would let one node execute two paths.
    """
    graph = compiled_graph.get_graph()
    for node_name in SPECIALIST_NODE_NAMES:
        outgoing = [edge for edge in graph.edges if edge.source == node_name]
        assert outgoing, node_name
        assert all(edge.conditional for edge in outgoing), node_name


def test_specialist_destinations_are_validation_or_execution_only(compiled_graph):
    graph = compiled_graph.get_graph()
    for node_name in SPECIALIST_NODE_NAMES:
        targets = {edge.target for edge in graph.edges if edge.source == node_name}
        assert "__end__" not in targets
        assert targets <= {
            "validate_output",
            "finalize",
            "tools",
            "approval",
            "rag_tools",
            "planning_tools",
        }, (node_name, targets)


def test_finalize_is_the_only_static_terminal_edge(compiled_graph):
    graph = compiled_graph.get_graph()
    terminal = [edge for edge in graph.edges if edge.target == "__end__"]
    assert len(terminal) == 1
    assert terminal[0].source == "finalize"
    assert terminal[0].conditional is False


def test_graph_starts_at_the_route_node(compiled_graph):
    graph = compiled_graph.get_graph()
    assert {edge.target for edge in graph.edges if edge.source == "__start__"} == {"route"}


def test_graph_registers_validation_and_finalization_nodes(compiled_graph):
    nodes = set(compiled_graph.get_graph().nodes)
    assert {"route", "validate_output", "finalize"} <= nodes
    assert nodes >= SPECIALIST_NODE_NAMES


# ----------------------------------------------------------------------
# route node
# ----------------------------------------------------------------------


async def _build_context(state, inventory):
    return SimpleNamespace(serialized_json="{}", inventory_version=inventory.version)


def _runtime(routing_service, inventory=None):
    return SimpleNamespace(
        context=SimpleNamespace(
            routing_service=routing_service,
            inventory=inventory or _inventory(),
            build_routing_context=_build_context,
        )
    )


class FakeRoutingService:
    def __init__(self, decision=None, error=None):
        self.decision = decision
        self.error = error
        self.route = AsyncMock(side_effect=self._route)

    async def _route(self, *args, **kwargs):
        if self.error is not None:
            raise self.error
        return self.decision

    async def build_context(self, **kwargs):
        return None


async def test_route_node_returns_command_with_immutable_decision():
    decision = RoutingDecision(agent_id="search_agent", confidence=0.9, reason="current info")
    service = FakeRoutingService(decision=decision)
    node = make_route_node()

    command = await node(
        {"turn_identity": _turn_identity(), "messages": []}, _runtime(service)
    )

    assert isinstance(command, Command)
    assert command.goto == "search_agent"
    assert command.update["routing_decision"] == decision
    assert command.update["active_agent_id"] == "search_agent"
    assert command.update["execution_phase"] == "executing"
    assert command.update["agent_history"] == [
        AgentTransition(from_agent_id=None, to_agent_id="search_agent", source="router")
    ]


async def test_route_node_resolves_custom_targets_to_the_shared_node():
    inventory = _inventory(
        {"custom_agent:alpha": {"runtime_agent_id": "custom_agent:alpha", "name": "Alpha"}}
    )
    service = FakeRoutingService(
        decision=RoutingDecision(agent_id="custom_agent:alpha", confidence=0.8, reason="attached")
    )
    node = make_route_node()

    command = await node(
        {"turn_identity": _turn_identity(), "messages": []}, _runtime(service, inventory)
    )

    assert command.goto == "custom_agent"
    assert command.update["active_agent_id"] == "custom_agent:alpha"


async def test_route_node_records_the_inventory_version():
    inventory = _inventory()
    service = FakeRoutingService(
        decision=RoutingDecision(agent_id="chat_agent", confidence=0.5, reason="general")
    )
    command = await make_route_node()(
        {"turn_identity": _turn_identity(), "messages": []}, _runtime(service, inventory)
    )
    assert command.update["routing_inventory_version"] == inventory.version


async def test_route_node_failure_goes_to_finalize_without_selecting_an_agent():
    error = WorkflowRoutingException(
        WorkflowError(code="routing_timeout", retriable=True, request_id="request-1")
    )
    service = FakeRoutingService(error=error)

    command = await make_route_node()(
        {"turn_identity": _turn_identity(), "messages": []}, _runtime(service)
    )

    assert command.goto == "finalize"
    assert command.update["execution_phase"] == "failed"
    assert command.update["workflow_error"].code == "routing_timeout"
    assert "active_agent_id" not in command.update
    assert "routing_decision" not in command.update


async def test_route_node_failure_never_substitutes_chat_agent():
    service = FakeRoutingService(
        error=WorkflowRoutingException(
            WorkflowError(
                code="routing_provider_unavailable", retriable=True, request_id="request-1"
            )
        )
    )
    command = await make_route_node()(
        {"turn_identity": _turn_identity(), "messages": []}, _runtime(service)
    )
    assert command.update.get("active_agent_id") is None
    assert command.goto != "chat_agent"


# ----------------------------------------------------------------------
# turn isolation
# ----------------------------------------------------------------------


def test_every_turn_gets_its_own_checkpoint_thread():
    first = build_checkpoint_thread_id("conversation-1", "message-1")
    second = build_checkpoint_thread_id("conversation-1", "message-2")
    assert first == "routing-v2:conversation-1:message-1"
    assert second == "routing-v2:conversation-1:message-2"
    assert first != second


def test_state_has_no_legacy_routing_vocabulary():
    from app.ai.workflow.state import WorkflowState

    for forbidden in ("selected_agent", "last_agent", "delegation_count"):
        assert forbidden not in WorkflowState.__annotations__
