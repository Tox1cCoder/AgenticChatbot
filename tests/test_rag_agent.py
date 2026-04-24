"""Phase 8 guards: RAGAgent has a single agentic execution path.

Prompt-built RAG, the ``agentic_rag_enabled`` toggle, and the traditional
streaming branch are all gone. Every code path funnels through the
search_documents tool.
"""

from __future__ import annotations

import inspect

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
