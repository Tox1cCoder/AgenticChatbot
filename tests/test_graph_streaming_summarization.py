"""Memory refactor Task 6 guards: long-term summarization is off the hot path.

The graph no longer contains the legacy ``summarize`` node at all — it was a
no-op orphan (no incoming/outgoing edges) kept only for old checkpoints, and
was verified to be scheduled by zero live checkpoints before removal. Durable
summaries refresh after assistant persistence (in ``MessageService``) —
verified separately in ``test_message_history_pipeline.py``.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("qdrant_client", MagicMock())
sys.modules.setdefault("qdrant_client.models", MagicMock())
sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("langchain_openai", MagicMock())
sys.modules.setdefault("langchain", MagicMock())
sys.modules.setdefault("langchain.agents", MagicMock())

from app.ai.graph import MultiAgentWorkflow  # noqa: E402 (heavy deps stubbed above first)


def test_workflow_has_no_summarize_node_or_method():
    """The legacy compatibility node and its method must be fully removed, and
    ``START`` must connect directly to ``route`` so streaming never waits on a
    summary node before the first user-visible token."""
    import inspect

    from app.ai.workflow.graph_builder import build_workflow_graph

    assert not hasattr(MultiAgentWorkflow, "_summarization_node")
    # Topology now lives in the extracted builder (Task 8); introspect it so the
    # START -> route invariant is still guarded at its real definition site.
    source = inspect.getsource(build_workflow_graph)
    assert '"summarize"' not in source
    assert 'add_edge(START, "route")' in source


def test_message_service_refresh_summary_method_exists():
    """MessageService must expose the off-hot-path summary refresh entry
    point used after assistant persistence."""
    from app.services.message_service import MessageService

    assert hasattr(MessageService, "refresh_summary_after_turn")
    assert callable(MessageService.refresh_summary_after_turn)
