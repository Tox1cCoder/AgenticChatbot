"""Memory refactor Task 6 guards: long-term summarization is off the hot path.

The graph no longer routes through ``summarize`` before ``route``. The
``_summarization_node`` is kept as a no-op so legacy checkpoints don't
crash. Durable summaries refresh after assistant persistence (in
``MessageService``) — verified separately in
``test_message_history_pipeline.py``.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("qdrant_client", MagicMock())
sys.modules.setdefault("qdrant_client.models", MagicMock())
sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("langchain_openai", MagicMock())
sys.modules.setdefault("langchain", MagicMock())
sys.modules.setdefault("langchain.agents", MagicMock())

from app.ai.graph import MultiAgentWorkflow
from app.ai.schemas import GraphState


@pytest.mark.asyncio
async def test_summarization_node_is_a_noop_passthrough():
    """The kept-for-compat node returns state unchanged and never invokes
    the model."""
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    state: GraphState = {  # type: ignore[assignment]
        "conversation_id": "conv-1",
        "context": {},
    }

    result = await workflow._summarization_node(state)

    assert result is state
    assert "history_summary" not in result


def test_graph_starts_at_route_not_summarize():
    """``START`` must connect directly to ``route`` so streaming never waits
    on the summary model before the first user-visible token."""
    import inspect

    source = inspect.getsource(MultiAgentWorkflow._build_graph)

    # The legacy edge "START -> summarize -> route" must be gone.
    assert 'add_edge(START, "summarize")' not in source
    assert 'add_edge("summarize", "route")' not in source

    # Route is the new entrypoint.
    assert 'add_edge(START, "route")' in source


def test_message_service_refresh_summary_method_exists():
    """MessageService must expose the off-hot-path summary refresh entry
    point used after assistant persistence."""
    from app.services.message_service import MessageService

    assert hasattr(MessageService, "refresh_summary_after_turn")
    assert callable(MessageService.refresh_summary_after_turn)
