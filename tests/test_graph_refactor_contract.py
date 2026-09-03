"""Graph refactor contract guards (Tasks 6 & 8).

Task 6 removed the legacy no-op ``summarize`` node. The plan's original
pre-deletion gate — ``SELECT COUNT(*) FROM checkpoints WHERE checkpoint::text
ILIKE '%summarize%'`` == 0 — is unusable as a permanent guard: that substring
also matches historical ``versions_seen`` / ``channel_versions`` bookkeeping
(retained forever on any thread that ever ran the node) and ordinary user
message content ("please summarize ..."), so it is non-zero on any real DB and
does not indicate danger.

The real, verified invariant is that no live checkpoint *schedules* the removed
node: a pending write whose channel is exactly ``summarize`` would cause a
"node not found" error on resume. That count is asserted here. (The exhaustive
one-time pre-deletion gate loaded every summarize-referencing thread's state
through the compiled graph and confirmed 0 of 247 had ``summarize`` in
``.next``.)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, text

from app.ai.graph import create_workflow
from app.ai.workflow.graph_builder import BASE_SPECIALIST_NODES, PLANNING_ENTRY_NODE
from app.ai.workflow.planning_execution import PLANNING_NODE_NAMES
from app.core.config import settings

# Node names, not public agent IDs: Planning's node is ``planning_model``.
# Read from the builder so this cannot drift away from the graph it describes.
BASE_AGENTS = frozenset(BASE_SPECIALIST_NODES)
STANDARD_TOOL_CALLING_AGENTS = frozenset(
    {"chat_agent", "search_agent", "image_generator_agent", "canvas_agent", "custom_agent"}
)


@pytest.fixture(scope="module")
def topology():
    """Introspectable compiled topology built from the extracted builder.

    Uses ``MagicMock`` collaborators so no live DB or Qdrant is touched — the
    graph wiring is independent of those runtime dependencies.
    """
    wf = create_workflow(qdrant_client=MagicMock(), embedding_service=MagicMock())
    return wf.graph.get_graph()


def _targets(graph, source: str) -> set[str]:
    return {edge.target for edge in graph.edges if edge.source == source}


def test_start_routes_directly_to_route(topology):
    assert _targets(topology, "__start__") == {"route"}


def test_no_summarize_node(topology):
    assert "summarize" not in topology.nodes


def test_route_targets_all_base_agents_plus_custom(topology):
    """The router reaches every routable specialist, and failure reaches finalize."""
    targets = _targets(topology, "route")
    assert targets >= BASE_AGENTS
    assert "custom_agent" in targets
    assert "finalize" in targets
    # Routing failure terminates through the finalizer, never straight to END.
    assert "__end__" not in targets


def test_standard_specialists_have_no_parent_level_tool_stage(topology):
    """Their model/tool loop runs inside a compiled create_agent subgraph.

    A parent-level ``tools``/``approval`` hop would mean the loop leaked back
    out of the subgraph, which is exactly what routing-v2 removed.
    """
    for agent in STANDARD_TOOL_CALLING_AGENTS:
        targets = _targets(topology, agent)
        assert targets <= {"validate_output", "finalize", "resolve_transition"}, (agent, targets)
        assert "tools" not in targets
        assert "approval" not in targets


def test_planning_has_no_parent_level_tool_stage(topology):
    """Fan-out is topology now. A ``planning_tools`` node is the replay bug.

    A tool call is one unit of work to the checkpointer, so a worker pausing
    inside one left the whole call unfinished and the resume re-ran every
    sibling that had already completed.
    """
    assert "planning_tools" not in topology.nodes
    assert set(PLANNING_NODE_NAMES) <= set(topology.nodes)


def test_planning_reaches_only_its_own_nodes_and_the_shared_exits(topology):
    """Planning never targets another specialist directly.

    Moving execution between agents is the transition resolver's job, so a
    Planning handoff goes there rather than jumping to the target.
    """
    targets = _targets(topology, PLANNING_ENTRY_NODE)

    assert targets <= {
        *PLANNING_NODE_NAMES,
        "resolve_transition",
        "validate_output",
        "finalize",
    }, targets
    assert not (targets & (BASE_AGENTS - {PLANNING_ENTRY_NODE})), targets
    assert "__end__" not in targets


def test_planning_packaging_revalidates_before_publication(topology):
    """Planning writes the public answer itself, so it is validated again."""
    assert "validate_output" in _targets(topology, "planning_package")


def test_only_the_finalizer_reaches_end(topology):
    terminal = [edge for edge in topology.edges if edge.target == "__end__"]
    assert [edge.source for edge in terminal] == ["finalize"]


def _engine():
    if not settings.database_url.startswith("postgresql"):
        pytest.skip("checkpoint contract requires PostgreSQL")
    return create_engine(settings.database_url)


def test_no_live_checkpoint_write_schedules_removed_summarize_node():
    with _engine().connect() as conn:
        pending = conn.execute(
            text("SELECT COUNT(*) FROM checkpoint_writes WHERE channel = 'summarize'")
        ).scalar_one()
    assert pending == 0
