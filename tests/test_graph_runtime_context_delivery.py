"""The runtime context has to actually arrive at the node that needs it.

The `route` node reads its collaborators from `runtime.context`. If that is
`None` it raises `routing_provider_unavailable` with reason `runtime_missing`
*before* the router is consulted, so every turn fails and no provider is ever
called.

That is exactly what shipped: the context was merged into `config["context"]`,
which LangGraph ignores — it populates `Runtime.context` only from the
`context=` keyword argument of `ainvoke`/`astream`/`astream_events`. Nothing
raised, nothing warned; `getattr(runtime, "context", None)` simply returned
`None` on every call.

No existing test caught it because the end-to-end suites script the workflow
object and never drive the real compiled graph, and the unit tests for `route`
construct a runtime double by hand. Both stub out the one thing that was
broken: the delivery. These tests therefore assert the *plumbing* against a
real compiled `StateGraph`, and the first one is deliberately a
characterisation of LangGraph's own contract rather than of our code — if a
future version starts honouring `config["context"]`, this is where that shows
up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class _State(TypedDict, total=False):
    seen: Any


@dataclass
class _Ctx:
    marker: str = "present"


def _probe_graph():
    async def node(state, runtime=None):
        return {"seen": getattr(runtime, "context", None)}

    graph = StateGraph(_State)
    graph.add_node("n", node)
    graph.add_edge(START, "n")
    graph.add_edge("n", END)
    return graph.compile()


# ----------------------------------------------------------------------
# LangGraph's contract, pinned
# ----------------------------------------------------------------------


async def test_langgraph_ignores_a_context_placed_in_config():
    """`config["context"]` is silently dropped. This is the bug's mechanism."""
    result = await _probe_graph().ainvoke({}, config={"context": _Ctx()})
    assert result["seen"] is None


async def test_langgraph_delivers_a_context_passed_as_a_keyword():
    result = await _probe_graph().ainvoke({}, context=_Ctx())
    assert result["seen"] == _Ctx()


async def test_the_context_keyword_also_reaches_a_streamed_run():
    graph = _probe_graph()
    seen = []
    async for chunk in graph.astream({}, context=_Ctx(), stream_mode="updates"):
        seen.append(chunk)

    contexts = [
        update["seen"] for chunk in seen for update in chunk.values() if isinstance(update, dict)
    ]
    assert _Ctx() in contexts


# ----------------------------------------------------------------------
# our own wiring
# ----------------------------------------------------------------------


def test_every_parent_graph_run_passes_the_context_keyword():
    """Structural, because the failure is silent by construction.

    Smuggling the context through `config` raises nothing and logs nothing --
    it just makes `route` fail on every turn, so there is no signal to notice
    at runtime. Assert on the call shape instead: every run of the parent graph
    must name `context=`.
    """
    import ast
    import pathlib

    source = (pathlib.Path(__file__).resolve().parent.parent / "app" / "ai" / "graph.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)

    def _is_parent_graph_run(node: ast.Call) -> bool:
        func = node.func
        if isinstance(func, ast.Name) and func.id == "iter_v3_events_from_graph":
            first = node.args[0] if node.args else None
            return isinstance(first, ast.Attribute) and first.attr == "graph"
        if not isinstance(func, ast.Attribute) or func.attr not in {"ainvoke", "astream"}:
            return False
        # `self.graph.<call>` only -- the RAG subgraph takes its collaborators
        # through its own invocation scope, not through Runtime.context.
        return isinstance(func.value, ast.Attribute) and func.value.attr == "graph"

    missing = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _is_parent_graph_run(node)
        and not any(keyword.arg == "context" for keyword in node.keywords)
    ]

    assert missing == [], (
        "these parent-graph runs in app/ai/graph.py do not pass `context=`, so their `route` "
        f"node will see runtime.context=None and fail every turn: lines {missing}"
    )


async def test_the_route_node_reports_runtime_missing_without_a_context():
    """The failure mode itself, so the error stays honest rather than silent.

    A missing context must produce a typed, named failure. Guessing a default
    router or an empty inventory here would route every turn to whatever agent
    sorted first.
    """
    from app.ai.workflow.contracts import WorkflowError
    from app.ai.workflow.graph_builder import make_route_node

    route = make_route_node()
    command = await route({"turn_identity": None}, runtime=None)

    error = command.update["workflow_error"]
    assert isinstance(error, WorkflowError)
    assert error.code == "routing_provider_unavailable"
    assert error.details["reason"] == "runtime_missing"


@pytest.mark.parametrize("missing", ["routing_service", "inventory"])
async def test_a_half_built_context_is_refused_too(missing):
    """Half a context is not better than none; both halves are required."""
    from types import SimpleNamespace

    from app.ai.workflow.graph_builder import make_route_node
    from app.ai.workflow.inventory import build_routing_inventory

    context = SimpleNamespace(
        routing_service=object(),
        inventory=build_routing_inventory(base_agent_ids=["chat_agent"], custom_agents={}),
    )
    setattr(context, missing, None)

    route = make_route_node()
    command = await route({"turn_identity": None}, runtime=SimpleNamespace(context=context))

    assert command.update["workflow_error"].details["reason"] == "runtime_missing"


async def test_the_real_route_node_reaches_the_router_when_context_is_delivered():
    """End to end over the seam that was broken: node + delivery together.

    The two halves were each fine in isolation -- the node reads
    `runtime.context`, and LangGraph delivers `context=` -- and the bug lived
    only in how they were connected. So assert them connected, in a compiled
    graph, with the production node.
    """
    from types import SimpleNamespace

    from app.ai.workflow.contracts import RoutingDecision
    from app.ai.workflow.graph_builder import make_route_node
    from app.ai.workflow.inventory import build_routing_inventory
    from app.ai.workflow.runtime_context import WorkflowRuntimeContext

    inventory = build_routing_inventory(
        base_agent_ids=["chat_agent", "search_agent"], custom_agents={}
    )
    routed = []

    class _Router:
        async def route(self, context, inv, **kwargs):
            routed.append(kwargs.get("request_id"))
            return RoutingDecision(agent_id="search_agent", confidence=0.9, reason="needs the web")

    class _Builder:
        async def build(self, request):
            return SimpleNamespace(message=request.message, inventory_version=inv_version)

    inv_version = inventory.version

    class _State(TypedDict, total=False):
        turn_identity: Any
        messages: list
        custom_agents: dict
        routing_decision: Any
        active_agent_id: str
        workflow_error: Any

    graph = StateGraph(_State)
    graph.add_node(
        "route",
        make_route_node(),
        destinations=("chat_agent", "search_agent", "finalize"),
    )
    graph.add_node("chat_agent", lambda state: {})
    graph.add_node("search_agent", lambda state: {})
    graph.add_node("finalize", lambda state: {})
    graph.add_edge(START, "route")
    for terminal in ("chat_agent", "search_agent", "finalize"):
        graph.add_edge(terminal, END)
    compiled = graph.compile()

    context = WorkflowRuntimeContext(
        routing_service=_Router(),
        inventory=inventory,
        routing_context_builder=_Builder(),
    )
    result = await compiled.ainvoke(
        {"turn_identity": SimpleNamespace(request_id="request-1"), "messages": []},
        context=context,
    )

    assert result.get("workflow_error") is None, result.get("workflow_error")
    assert routed == ["request-1"], "the router was never reached"
    assert result["active_agent_id"] == "search_agent"
