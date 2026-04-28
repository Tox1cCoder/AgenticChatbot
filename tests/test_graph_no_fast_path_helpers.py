"""Phase 12 guard: graph.py has no leftover traditional-RAG fast-path helpers.

Phase 8 deleted the traditional RAG streaming branch from
``execute_request_stream``. Two helpers (``_run_fast_path_summarization``
and ``_persist_fast_path_turn``) and the divider comment were left behind
unreferenced. This test pins their removal.
"""

from __future__ import annotations

import inspect


def test_workflow_class_has_no_fast_path_methods():
    from app.ai.graph import MultiAgentWorkflow

    assert not hasattr(MultiAgentWorkflow, "_run_fast_path_summarization"), (
        "_run_fast_path_summarization is dead code from Phase 8 — remove it"
    )
    assert not hasattr(MultiAgentWorkflow, "_persist_fast_path_turn"), (
        "_persist_fast_path_turn is dead code from Phase 8 — remove it"
    )


def test_graph_source_has_no_fast_path_residue():
    import app.ai.graph as graph_module

    source = inspect.getsource(graph_module)

    forbidden_phrases = (
        "_run_fast_path_summarization",
        "_persist_fast_path_turn",
        "Fast-path helpers",
        "traditional-RAG streaming",
        "traditional RAG streaming",
    )
    for phrase in forbidden_phrases:
        assert phrase not in source, (
            f"Phase 12: residual '{phrase}' must be deleted from app/ai/graph.py"
        )
