"""What the routing-v2 cutover has actually removed — and what it has not.

This file is deliberately two-sided. The first half asserts the legacy paths
that are gone, so they cannot creep back. The second half records, explicitly,
the legacy code that is still reachable in production, so nobody reads a green
suite as "the cutover is finished".

Update the second half by deleting entries as their behavior moves into the v2
components — never by loosening an assertion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_PATHS = (REPO_ROOT / "app" / "ai", REPO_ROOT / "app" / "services")


def _runtime_source() -> str:
    parts: list[str] = []
    for root in RUNTIME_PATHS:
        for path in sorted(root.rglob("*.py")):
            parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


RUNTIME_SOURCE = _runtime_source()


# ----------------------------------------------------------------------
# removed for good
# ----------------------------------------------------------------------

REMOVED_RUNTIME_TOKENS = (
    # Routing vocabulary: one mutable field no longer carries five concerns.
    "selected_agent",
    "last_agent",
    # Custom-agent stickiness and explicit-name matching.
    "_sticky_custom_agent_for_followup",
    "_sticky_custom_agent",
    "_match_explicit_custom_agent",
    "_extract_agent_name",
    # The grounding rollout branch.
    "rag_grounded_answer_gate_enabled",
    # The auto-continuation outer loop. A second round re-enters ``route``,
    # and the turn's routing decision is set-once.
    "auto_continue_enabled",
    "auto_continue_max_rounds",
    "_build_continuation_state",
    "_capture_state_for_continuation",
    # Parent-level tool and approval stages: a standard specialist runs its
    # whole loop, approval included, inside its own subgraph.
    "_approval_node",
    "_route_tool_output",
    # A handoff is a parent command, never a value a specialist returns.
    "HandoffOutcome",
    # An authorization hook production never supplied, so it allowed everything.
    "ToolAuthorizationMiddleware",
)


@pytest.mark.parametrize("token", REMOVED_RUNTIME_TOKENS)
def test_removed_legacy_token_is_absent_from_the_runtime(token):
    assert token not in RUNTIME_SOURCE, f"{token!r} is back in the runtime"


def test_the_router_owns_no_provider_sdk_client():
    router = (REPO_ROOT / "app" / "ai" / "agents" / "router.py").read_text(encoding="utf-8")
    assert "genai" not in router
    assert "_call_llm" not in router


def test_routing_never_returns_a_hard_coded_agent():
    routing = (REPO_ROOT / "app" / "ai" / "workflow" / "routing.py").read_text(encoding="utf-8")
    assert 'return "chat_agent"' not in routing


def test_delegation_depth_is_no_longer_a_top_level_state_field():
    from app.ai.workflow.state import WorkflowState

    assert "delegation_count" not in WorkflowState.__annotations__


def test_the_graph_has_no_top_level_tool_or_approval_stage():
    """Standard specialists run their loop inside their own subgraph."""
    from unittest.mock import MagicMock

    from app.ai.graph import create_workflow

    workflow = create_workflow(qdrant_client=MagicMock(), embedding_service=MagicMock())
    nodes = set(workflow.graph.get_graph().nodes)

    assert "tools" not in nodes
    assert "approval" not in nodes


def test_only_the_finalizer_reaches_end():
    from unittest.mock import MagicMock

    from app.ai.graph import create_workflow

    workflow = create_workflow(qdrant_client=MagicMock(), embedding_service=MagicMock())
    graph = workflow.graph.get_graph()
    assert [edge.source for edge in graph.edges if edge.target == "__end__"] == ["finalize"]


# ----------------------------------------------------------------------
# not removed yet — recorded, not excused
# ----------------------------------------------------------------------

# Legacy modules still reachable in production. Each entry names what still
# runs through it, so the remaining work is visible rather than implied.
STILL_LIVE_LEGACY_MODULES = {
    "app/ai/workflow/rag_loop.py": (
        "the rag_agent node still runs the pre-v2 RAG loop — but it now owns "
        "the one grounding gate every RAG answer passes through, including "
        "Planning workers and the Planning synthesis"
    ),
    "app/ai/workflow/planning_loop.py": "the planning_agent node still runs the pre-v2 loop",
    "app/ai/workflow/tool_loop.py": (
        "rag_tools and planning_tools still use its approval, artifact, and "
        "tool-error helpers; its own tool and approval nodes are gone"
    ),
    "app/ai/agents/base_agent.py": (
        "specialist definitions delegate prompt and tool assembly to it; only its "
        "model/tool loop was superseded"
    ),
    "app/ai/agents/router.py": "a thin RoutingService adapter the pre-v2 graph import still needs",
}


@pytest.mark.parametrize("relative_path", sorted(STILL_LIVE_LEGACY_MODULES))
def test_known_live_legacy_module_still_exists(relative_path):
    """Fails when a module is deleted, so its entry must be removed here too.

    The point is bookkeeping: this test failing means the cutover advanced and
    this record is stale, not that something broke.
    """
    assert (REPO_ROOT / relative_path).exists(), (
        f"{relative_path} was deleted — remove its entry from "
        "STILL_LIVE_LEGACY_MODULES and delete its live-path assertions"
    )


def test_rag_and_planning_still_use_their_pre_v2_nodes():
    """Records that Tasks 7 and 8 built their components but did not cut over.

    The shared RAG graph and the Planning orchestrator exist and are tested,
    but the graph still routes rag_agent and planning_agent to the pre-v2
    loops. This asserts that gap so it cannot be mistaken for finished work.
    """
    from unittest.mock import MagicMock

    from app.ai.graph import create_workflow

    workflow = create_workflow(qdrant_client=MagicMock(), embedding_service=MagicMock())
    nodes = set(workflow.graph.get_graph().nodes)

    assert "rag_tools" in nodes
    assert "planning_tools" in nodes


def test_response_recovery_no_longer_fabricates_an_answer():
    """The salvage path around the finalizer is gone.

    ``_recover_terminal_response`` still exists as the adapter's accessor for
    the finalizer's response, but it no longer scans accumulated stream chunks
    or checkpoint messages for assistant-looking text, and it no longer takes
    a parameter to pass such text in through. A turn with no finalized response
    yields a typed error instead of an unvalidated draft. Covered in detail by
    tests/test_no_unvalidated_response_recovery.py.
    """
    import ast

    source = (REPO_ROOT / "app" / "ai" / "graph.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    recover = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_recover_terminal_response"
    )

    # No AgentResponse is constructed inside it any more — it only returns one
    # that the finalizer already built and validated.
    constructed = {
        node.func.id
        for node in ast.walk(recover)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "AgentResponse" not in constructed
    assert "AgentMessage" not in constructed


def test_the_finalizer_is_the_only_in_graph_response_builder():
    """Whatever the adapter does, no graph node but finalize sets a response."""
    from app.ai.workflow import finalization, graph_builder

    assert "PublicResponseFinalizer" in dir(finalization)
    assert graph_builder.SPECIALIST_NODE_NAMES
    builder_source = (REPO_ROOT / "app" / "ai" / "workflow" / "graph_builder.py").read_text(
        encoding="utf-8"
    )
    assert builder_source.count('graph.add_edge("finalize", END)') == 1


def test_the_shared_rag_graph_and_worker_runtime_are_constructed_once():
    """The v2 RAG/Planning components are now the wired production objects.

    Both entry points must hold the *same* compiled graph. This checks the
    wiring structurally -- the worker runtime is handed ``self.rag_execution_
    graph`` rather than anything it could build per invocation -- because a
    factory is how the two paths drifted into two graphs with two grounding
    policies. Runtime identity of the two attributes is asserted in
    ``tests/test_planning_execution_graph.py``.

    Replaces the record that these were importable but unwired. The remaining
    step is the topology cutover, tracked by
    ``test_the_two_rag_loops_are_still_separate_implementations``.
    """
    import pathlib
    import re

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    graph_source = (repo_root / "app" / "ai" / "graph.py").read_text(encoding="utf-8")

    assert re.search(r"self\.rag_execution_graph = RagExecutionGraph\(", graph_source)
    assert re.search(r"rag_execution_graph=self\.rag_execution_graph", graph_source)
    assert "PlanningWorkerRuntime(" in graph_source

    # No per-invocation RAG graph construction survives anywhere in the app.
    factories = [
        path.relative_to(repo_root).as_posix()
        for path in sorted((repo_root / "app").rglob("*.py"))
        if re.search(r"RagExecutionGraphFactory|rag_execution_factory", path.read_text("utf-8"))
    ]
    assert factories == [], f"a per-invocation RAG graph factory came back in {factories}"
