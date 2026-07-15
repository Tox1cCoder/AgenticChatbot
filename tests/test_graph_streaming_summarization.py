"""Memory refactor Task 6 guards: long-term summarization is off the hot path.

The graph no longer contains the legacy ``summarize`` node at all — it was a
no-op orphan (no incoming/outgoing edges) kept only for old checkpoints, and
was verified to be scheduled by zero live checkpoints before removal. Assistant
persistence atomically advances durable compaction work; a content-free Celery
hint is published after commit and PostgreSQL reconciliation recovers losses.
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


def test_message_service_has_no_in_process_summary_refresh():
    """Assistant persistence owns durable work; the service has no runner."""
    from app.services.message_service import MessageService

    assert not hasattr(MessageService, "refresh_summary_after_turn")
    assert not hasattr(MessageService, "_schedule_summary_refresh")
