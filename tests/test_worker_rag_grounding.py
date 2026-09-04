"""One grounding path, for a worker and for a public turn alike.

This file used to pin an invariant across three separate call sites: the
top-level RAG loop, the Planning RAG worker, and Planning synthesis each
reached ``_apply_grounded_answer_gate`` on their own, and a fourth path that
forgot to would have published unchecked citations. The three call sites are
gone. ``RagExecutionGraph`` is compiled once and both entry points hold the
same object, so "every path is grounded" is now a property of there being one
path rather than of three call sites agreeing.

What moved where:

* citation neutralization, zero-evidence answers, ambiguous ids, and the
  server source list -> ``test_rag_execution_graph.py``, which exercises them
  mode-agnostically because public and worker runs are the same graph;
* a worker being graded only against the evidence *it* retrieved ->
  ``test_rag_worker_is_graded_only_against_its_own_scope`` in
  ``test_planning_execution_graph.py``;
* synthesis declaring ``rag_grounding`` when it carries worker evidence ->
  ``test_synthesis_propagates_worker_evidence_and_declares_grounding``.

What is left here is what none of those cover: that the two entry points are
literally the same object, and that grounding is not applied to things that
are not answers.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.ai.workflow.contracts import WorkerResult
from app.ai.workflow.planning_execution import build_planning_outcome


def _result(agent_id: str, content: str, evidence=()) -> WorkerResult:
    return WorkerResult(
        dispatch_id="d1",
        task_id="w1",
        position=0,
        agent_id=agent_id,
        status="completed",
        content=content,
        evidence=tuple(evidence),
    )


# ----------------------------------------------------------------------
# one graph, two entry points
# ----------------------------------------------------------------------


def test_the_public_turn_and_the_worker_hold_the_same_graph_object():
    """Not "two graphs configured alike" -- the same instance.

    Two instances were how one of them could be given a different gate.
    """
    from app.ai.graph import create_workflow

    workflow = create_workflow(qdrant_client=MagicMock(), embedding_service=MagicMock())

    assert workflow.planning_worker_runtime.rag_execution_graph is workflow.rag_execution_graph
    assert workflow.rag_execution_graph.compile_count == 1


def test_there_is_no_second_grounding_call_site():
    """The helper the three old paths each called is gone.

    A new answer path cannot forget to ground, because grounding is a node in
    the only graph that produces a RAG answer.
    """
    import pathlib
    import re

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    callers = sorted(
        path.relative_to(repo_root).as_posix()
        for path in (repo_root / "app").rglob("*.py")
        if re.search(r"_apply_grounded_answer_gate", path.read_text(encoding="utf-8"))
    )
    assert callers == [], f"a second grounding call site reappeared in {callers}"


def test_validation_is_a_node_the_graph_cannot_route_around():
    from types import SimpleNamespace as _NS

    from app.ai.workflow.rag_execution import RagExecutionGraph
    from app.services.rag_grounding import GroundedAnswerGate

    graph = RagExecutionGraph(
        runtime=_NS(),
        grounded_answer_gate=GroundedAnswerGate(min_coverage=0.5),
        settings=_NS(rag_max_tool_iterations=4),
    )
    compiled = graph._compiled.get_graph()
    edges = {(edge.source, edge.target) for edge in compiled.edges}

    assert ("validate_grounding", "package_rag_result") in edges
    incoming = {source for source, target in edges if target == "package_rag_result"}
    assert incoming == {"validate_grounding"}, (
        f"packaging is reachable without validation, from {incoming - {'validate_grounding'}}"
    )


# ----------------------------------------------------------------------
# grounding applies to answers, and only to answers
# ----------------------------------------------------------------------


@pytest.mark.parametrize("agent_id", ["search_agent", "chat_agent"])
def test_a_non_rag_worker_result_declares_no_grounding(agent_id):
    """Grounding is about retrieved evidence; a search worker retrieves none."""
    outcome = build_planning_outcome(content="a plain answer", results=[_result(agent_id, "x")])

    assert "rag_grounding" not in outcome.provenance.output_policy_ids


def test_a_synthesis_over_a_rag_worker_declares_grounding():
    outcome = build_planning_outcome(
        content="synthesized [E1]",
        results=[_result("rag_agent", "grounded", evidence=({"evidence_id": "E1"},))],
    )

    assert "rag_grounding" in outcome.provenance.output_policy_ids
    assert outcome.provenance.evidence == ({"evidence_id": "E1"},)


def test_a_failed_worker_contributes_no_evidence_to_the_synthesis():
    """A failure has nothing to ground, and must not look like it does."""
    failed = WorkerResult(
        dispatch_id="d1",
        task_id="w1",
        position=0,
        agent_id="rag_agent",
        status="failed",
        content="",
        error_code="worker_timeout",
    )

    outcome = build_planning_outcome(content="could not complete", results=[failed])

    assert outcome.provenance.evidence == ()
    assert "rag_grounding" not in outcome.provenance.output_policy_ids


def test_only_the_packaging_node_produces_a_planning_outcome():
    """A tool-calling Planning step publishes nothing, so it is never grounded."""
    import inspect

    from app.ai.workflow.planning_execution import PLANNING_NODE_NAMES, PlanningNodeFactory

    assert "planning_package" in PLANNING_NODE_NAMES

    source = inspect.getsource(PlanningNodeFactory)
    assert source.count("build_planning_outcome(") == 1, (
        "a second Planning node builds a public outcome; only packaging may"
    )
