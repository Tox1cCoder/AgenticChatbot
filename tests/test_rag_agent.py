"""Phase 8 + Phase 12 guards: RAGAgent has a single agentic execution path.

Prompt-built RAG, the ``agentic_rag_enabled`` toggle, and the traditional
streaming branch are all gone. Every code path funnels through the
search_documents tool. Phase 12 adds: READ_DOCUMENT / GREP_DOCUMENT /
LIST_DOCUMENTS hydrate from SQL with server-context auth filters.
"""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

from app.ai.agents import rag_agent as rag_agent_module
from app.ai.agents.rag_agent import RAGAgent


def test_rag_agent_module_does_not_import_build_rag_prompt():
    source = inspect.getsource(rag_agent_module)
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert "build_rag_prompt" not in stripped, (
            f"build_rag_prompt must not be referenced in rag_agent.py: {stripped!r}"
        )


def test_rag_agent_does_not_read_agentic_rag_enabled_flag():
    source = inspect.getsource(rag_agent_module)
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert "agentic_rag_enabled" not in stripped, (
            "agentic mode is the only mode — settings.agentic_rag_enabled must go"
        )


def test_process_message_has_no_traditional_rag_branch():
    source = inspect.getsource(RAGAgent.process_message)
    assert "Traditional RAG" not in source
    assert "build_rag_prompt" not in source
    # Single agentic path means no branch on agentic_mode.
    assert "self.agentic_mode" not in source, (
        "process_message must not branch on agentic_mode — it's the only path"
    )


def test_graph_document_aware_chat_does_not_split_on_traditional_rag():
    import app.ai.graph as graph_module

    source = inspect.getsource(graph_module)
    assert "Traditional RAG streaming" not in source, (
        "Traditional RAG streaming branch must be removed from graph.py"
    )
    assert "build_rag_prompt" not in source, (
        "graph.py must not reference build_rag_prompt after Phase 8"
    )


def test_search_tool_schema_does_not_carry_server_context():
    """server-owned context (user_id, conversation_id) must live outside the model-facing schema."""
    from app.ai.schemas import SearchDocumentsInput

    forbidden = {"user_id", "conversation_id", "device_id"}
    leaked = forbidden & set(SearchDocumentsInput.model_fields)
    assert not leaked, f"Server context must not be model-facing: {leaked}"


# ---------------------------------------------------------------------------
# Phase 12: SQL hydration with server-context auth filters.
# ---------------------------------------------------------------------------


def _build_minimal_agent() -> RAGAgent:
    """Construct a RAGAgent without invoking BaseAgent.__init__.

    Tests of the public document-helper methods only need ``settings`` and
    a stable agentic_preview_chars; everything else stays as MagicMock.
    """
    agent = object.__new__(RAGAgent)
    agent.settings = MagicMock()
    agent.agentic_preview_chars = 500
    return agent


def test_read_document_helper_accepts_server_context_filters():
    """get_document_full_content must accept user_id/conversation_id filters."""
    sig = inspect.signature(RAGAgent.get_document_full_content)
    params = sig.parameters
    assert "user_id" in params, "get_document_full_content must accept server-context user_id"
    assert "conversation_id" in params, (
        "get_document_full_content must accept server-context conversation_id"
    )


def test_grep_document_helper_accepts_server_context_filters():
    sig = inspect.signature(RAGAgent.grep_document)
    params = sig.parameters
    assert "user_id" in params, "grep_document must accept user_id"
    assert "conversation_id" in params, "grep_document must accept conversation_id"


def test_list_conversation_documents_helper_accepts_user_id_filter():
    sig = inspect.signature(RAGAgent.list_conversation_documents)
    params = sig.parameters
    assert "user_id" in params, "list_conversation_documents must accept user_id"


def test_get_document_full_content_returns_none_when_user_or_conversation_does_not_match():
    """Auth filters must short-circuit before any chunk content is hydrated."""
    agent = _build_minimal_agent()

    chunk_repo = MagicMock()
    # Repository returns no rows when the auth filters don't match.
    chunk_repo.get_by_document_for_scope.return_value = []

    document_id = uuid4()

    with patch("app.ai.agents.rag_agent.DocumentChunkRepository") as repo_cls:
        repo_cls.return_value = chunk_repo
        result = asyncio.run(
            agent.get_document_full_content(
                str(document_id),
                user_id="other-user",
                conversation_id="other-conv",
            )
        )

    assert result is None
    # Auth filters must reach the SQL layer, not be applied after the fact.
    chunk_repo.get_by_document_for_scope.assert_called_once()
    call_kwargs = chunk_repo.get_by_document_for_scope.call_args.kwargs
    assert call_kwargs.get("user_id") == "other-user"
    assert call_kwargs.get("conversation_id") == "other-conv"


def test_get_document_full_content_hydrates_when_filters_match():
    agent = _build_minimal_agent()

    chunk_repo = MagicMock()
    chunk_repo.get_by_document_for_scope.return_value = [
        SimpleNamespace(content="alpha"),
        SimpleNamespace(content="beta"),
    ]

    with patch("app.ai.agents.rag_agent.DocumentChunkRepository") as repo_cls:
        repo_cls.return_value = chunk_repo
        result = asyncio.run(
            agent.get_document_full_content(str(uuid4()), user_id="u", conversation_id="c")
        )

    assert result == "alpha\n\nbeta"


def test_list_conversation_documents_filters_by_user_when_provided():
    agent = _build_minimal_agent()

    fake_db = MagicMock()
    fake_query = MagicMock()
    fake_query.outerjoin.return_value = fake_query
    fake_query.join.return_value = fake_query
    fake_query.filter.return_value = fake_query
    fake_query.group_by.return_value = fake_query
    fake_query.order_by.return_value = fake_query
    fake_query.all.return_value = []
    fake_db.__enter__ = MagicMock(return_value=fake_db)
    fake_db.__exit__ = MagicMock(return_value=False)
    fake_db.query.return_value = fake_query

    with patch("app.ai.agents.rag_agent.SessionLocal", return_value=fake_db):
        asyncio.run(agent.list_conversation_documents(str(uuid4()), user_id=str(uuid4())))

    # The filter must be applied at the SQL layer — that means at least one
    # ``filter(...)`` call applies a user-id condition. We don't introspect the
    # SQLAlchemy expression — we just assert the filter happened more than the
    # baseline conversation_id-only path does, and that the user-id auth path
    # additionally joined the Conversation table.
    assert fake_query.filter.call_count >= 2, (
        "user_id auth filter must be added to the SQL filter chain"
    )
    assert fake_query.join.called, (
        "user_id auth filter must reach Conversation.owner_id via a SQL join"
    )


def test_rag_tool_actions_threads_server_context_into_helpers():
    """rag_tool_actions must forward user_id / conversation_id to RAGAgent helpers."""
    import app.ai.rag_tool_actions as actions_module

    source = inspect.getsource(actions_module)
    # Each helper invocation in the action handlers must carry user_id and
    # conversation_id forward — guard against silent regressions.
    assert "user_id=user_id" in source, (
        "rag_tool_actions must forward server-owned user_id to RAGAgent helpers"
    )
    assert "conversation_id=conversation_id" in source, (
        "rag_tool_actions must forward server-owned conversation_id to RAGAgent helpers"
    )
