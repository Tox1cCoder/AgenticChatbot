"""Parent graph topology for the routing-v2 production workflow.

This module owns topology only. Every new turn enters one ``route`` node, and
every path out of a specialist ends at ``validate_output`` and then
``finalize``. ``finalize -> END`` is the graph's sole static terminal edge, so
no specialist can publish an answer or end a turn on its own.

Specialists and tool stages return dynamic ``Command`` values and therefore
carry no static outgoing edges: a node cannot both command and be routed by an
edge, which is what made the pre-v2 graph able to execute two paths.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.ai.workflow.contracts import AgentTransition, WorkflowRoutingException
from app.ai.workflow.finalization import make_finalize_node, make_validate_output_node
from app.ai.workflow.specialists import (
    make_specialist_wrapper,
    make_subgraph_specialist_wrapper,
    make_tool_stage_wrapper,
)
from app.ai.workflow.state import WorkflowState

logger = logging.getLogger(__name__)

__all__ = [
    "BASE_SPECIALIST_NODES",
    "TRANSITION_RESOLVER_NODE",
    "SUBGRAPH_SPECIALIST_NODES",
    "SPECIALIST_NODE_NAMES",
    "TOOL_STAGE_NODES",
    "build_workflow_graph",
    "make_route_node",
]

BASE_SPECIALIST_NODES: tuple[str, ...] = (
    "chat_agent",
    "rag_agent",
    "search_agent",
    "image_generator_agent",
    "planning_agent",
    "canvas_agent",
)

SPECIALIST_NODE_NAMES: frozenset[str] = frozenset({*BASE_SPECIALIST_NODES, "custom_agent"})

TRANSITION_RESOLVER_NODE = "resolve_transition"

# Specialists whose model/tool loop already runs inside a compiled
# ``create_agent`` subgraph. RAG and Planning join them in Tasks 7 and 8.
SUBGRAPH_SPECIALIST_NODES: tuple[str, ...] = (
    "chat_agent",
    "search_agent",
    "image_generator_agent",
    "canvas_agent",
    "custom_agent",
)

# Pre-v2 execution stages that still run as top-level nodes. RAG and Planning
# move theirs inside subgraphs in Tasks 7 and 8; the standard specialists no
# longer have one at all.
TOOL_STAGE_NODES: tuple[str, ...] = ("rag_tools", "planning_tools")

# Every pre-v2 stage decision maps onto a v2 node. ``"end"`` means "this
# specialist produced a candidate answer", which is validation, never END.
_STAGE_TARGETS = {"end": "validate_output"}


def make_route_node():
    """Build the single new-turn routing node.

    The node calls ``RoutingService.route`` exactly once and returns a
    ``Command``. On failure it records a typed error and goes to ``finalize``:
    it never selects an agent, and never falls back to chat.
    """

    async def route(state: dict[str, Any], runtime: Any) -> Command:
        context = getattr(runtime, "context", None)
        routing_service = getattr(context, "routing_service", None)
        inventory = getattr(context, "inventory", None)
        identity = state.get("turn_identity")
        request_id = getattr(identity, "request_id", None) or "unknown"

        try:
            if routing_service is None or inventory is None:
                raise WorkflowRoutingException(
                    _routing_error("routing_provider_unavailable", request_id, "runtime_missing")
                )
            routing_context = await _build_routing_context(context, state, inventory, request_id)
            decision = await routing_service.route(
                routing_context,
                inventory,
                user_id=state.get("user_id"),
                model_request=state.get("model_request"),
                request_id=request_id,
            )
            target_node = inventory.resolve_node(decision.agent_id)
        except WorkflowRoutingException as exc:
            return Command(
                update={"workflow_error": exc.error, "execution_phase": "failed"},
                goto="finalize",
            )
        except KeyError:
            return Command(
                update={
                    "workflow_error": _routing_error(
                        "routing_target_unavailable", request_id, "unresolvable_node"
                    ),
                    "execution_phase": "failed",
                },
                goto="finalize",
            )

        return Command(
            update={
                "routing_decision": decision,
                "routing_inventory_version": inventory.version,
                "active_agent_id": decision.agent_id,
                "agent_history": [
                    AgentTransition(
                        from_agent_id=None, to_agent_id=decision.agent_id, source="router"
                    )
                ],
                "execution_phase": "executing",
            },
            goto=target_node,
        )

    return route


async def _build_routing_context(
    runtime_context: Any, state: dict[str, Any], inventory: Any, request_id: str
):
    """Assemble the bounded routing context for this turn.

    The runtime context owns context construction so the node stays a thin
    adapter: it must not learn how to read documents, canvas, or history.
    """
    build = getattr(runtime_context, "build_routing_context", None)
    if not callable(build):
        raise WorkflowRoutingException(
            _routing_error(
                "routing_provider_unavailable", request_id, "routing_context_builder_missing"
            )
        )
    return await build(state, inventory)


def _routing_error(code: str, request_id: str, reason: str):
    from app.ai.workflow.contracts import WorkflowError

    return WorkflowError(
        code=code, retriable=True, request_id=request_id, details={"reason": reason}
    )


def build_workflow_graph(
    workflow: Any,
    *,
    checkpointer: Any | None,
    context_schema: Any | None = None,
) -> Any:
    """Compile the routing-v2 parent graph.

    ``workflow`` supplies the specialist and stage callables. The topology
    below is the whole contract: one entry, dynamic transitions, one exit.
    """
    graph = (
        StateGraph(WorkflowState, context_schema=context_schema)
        if context_schema is not None
        else StateGraph(WorkflowState)
    )

    graph.add_node(
        "route",
        make_route_node(),
        destinations=tuple(sorted({*SPECIALIST_NODE_NAMES, "finalize"})),
    )

    # Standard specialists run their whole model/tool loop inside a compiled
    # ``create_agent`` subgraph, so they have no parent-level tool stage. RAG
    # and Planning keep theirs until Tasks 7 and 8 move them into subgraphs.
    subgraph_specialists = {
        node_name: make_subgraph_specialist_wrapper(
            node_name, _specialist_invoker(workflow, node_name)
        )
        for node_name in SUBGRAPH_SPECIALIST_NODES
    }

    specialist_callables = {
        "rag_agent": workflow._rag_node,
        "planning_agent": workflow._planning_node,
    }
    stage_routers = {
        "rag_agent": workflow._should_call_rag_tools,
        "planning_agent": workflow._should_call_planning_tools,
    }
    stage_targets_by_node = {
        "rag_agent": {"end": "validate_output", "rag_tools": "rag_tools"},
        "planning_agent": {"end": "validate_output", "planning_tools": "planning_tools"},
    }

    for node_name, wrapper in subgraph_specialists.items():
        graph.add_node(
            node_name,
            wrapper,
            destinations=("finalize", "resolve_transition", "validate_output"),
        )

    # The one component allowed to choose the next top-level specialist. A
    # specialist's handoff tool records a pending transition and comes here;
    # nothing else may move execution between agents.
    graph.add_node(
        "resolve_transition",
        workflow.build_transition_resolver(),
        destinations=tuple(sorted({*SPECIALIST_NODE_NAMES, "finalize"})),
    )

    # Declared destinations make the dynamic topology inspectable: the
    # "only finalize reaches END" invariant is checkable on the compiled graph
    # instead of living only in prose.
    stage_destinations = tuple(sorted({*SPECIALIST_NODE_NAMES, "validate_output", "finalize"}))

    for node_name, specialist in specialist_callables.items():
        targets = stage_targets_by_node.get(node_name, _STAGE_TARGETS)
        graph.add_node(
            node_name,
            make_specialist_wrapper(
                node_name,
                specialist,
                stage_router=stage_routers.get(node_name, workflow._should_call_tools),
                stage_targets=targets,
            ),
            destinations=tuple(sorted({*targets.values(), "finalize"})),
        )

    graph.add_node(
        "rag_tools",
        make_tool_stage_wrapper(
            "rag_tools", workflow._rag_tools_node, stage_router=workflow._should_continue_rag
        ),
        destinations=stage_destinations,
    )
    graph.add_node(
        "planning_tools",
        make_tool_stage_wrapper(
            "planning_tools",
            workflow._planning_tools_node,
            stage_router=workflow._should_continue_planning,
        ),
        destinations=stage_destinations,
    )
    graph.add_node("validate_output", make_validate_output_node(), destinations=("finalize",))
    graph.add_node("finalize", make_finalize_node())

    graph.add_edge(START, "route")
    graph.add_edge("finalize", END)

    if checkpointer:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()


def _specialist_invoker(workflow: Any, node_name: str):
    """Bind the workflow's per-node subgraph entry point."""

    async def invoke(state: dict[str, Any]):
        return await workflow.invoke_specialist_subgraph(node_name, state)

    return invoke


