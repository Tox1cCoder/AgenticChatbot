"""RAG agent and document-action behaviour that survived the loop cutover.

The graph-level half of this file tested ``_should_continue_rag`` and the
``rag_tools`` stage, which are gone: the tool loop is inside the shared
``RagExecutionGraph`` now, and its budget backstop is pinned by
``test_the_tool_loop_has_a_backstop_ceiling`` in ``test_rag_execution_graph.py``.

What remains is the agent-and-actions half, which the cutover did not touch:
``RAGAgent._process_message_agentic`` honouring the forced-final-response flag,
``execute_search_documents_action`` error shapes, ``list_documents`` pagination,
and the server-scope checks that refuse to query without a conversation and a
user.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.ai.agents.rag_agent import RAGAgent
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole


def test_process_message_agentic_disables_tools_when_force_final_response_flag_set():
    """RAG must disable tool binding when graph forwards rag_force_final_response."""
    agent = object.__new__(RAGAgent)
    agent.settings = type("S", (), {"agentic_preview_chars": 500})()
    agent.agentic_max_iterations = 5
    agent.tools = []
    agent.mcp_manager = None
    agent._tools_generation_seen = -1

    captured = {}

    async def fake_invoke(**kwargs):
        captured.update(kwargs)
        from app.ai.schemas import AgentResponse, AgentType

        return AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="final answer"),
            metadata={"agentic_mode": True},
        )

    from app.core.runtime_modeling import ResolvedRuntimeModelConfig

    runtime_config = ResolvedRuntimeModelConfig(
        agent_key="rag",
        provider="gemini",
        model="gemini-2.5-flash",
        temperature=0.7,
        api_key=None,
        key_source="settings",
        source="agent_default",
        capabilities={"supports_vision": False},
        fallback_config=None,
        warnings=[],
        provider_fallback=None,
        is_custom_model=False,
    )

    agent._invoke_agentic_rag_model = fake_invoke
    agent._resolve_runtime_model_config = lambda *a, **kw: runtime_config
    agent._create_fallback_runtime_config = lambda *a, **kw: None
    agent._build_skills_suffix = lambda **kw: ""
    agent._get_tools_for_binding = lambda **kw: []

    msg = AgentMessage(
        role=MessageRole.USER,
        content="Summarise the document.",
        metadata={
            "original_query": "Summarise the document.",
            "rag_force_final_response": True,
            "rag_tool_budget_notice": "Tool budget reached. Synthesise now.",
        },
    )

    response = asyncio.run(agent._process_message_agentic(msg, "conv-1"))

    assert captured.get("disable_tools") is True, (
        f"_invoke_agentic_rag_model must be called with disable_tools=True; "
        f"got {captured.get('disable_tools')}"
    )

    # The system prompt must include the budget notice.
    system_msg = captured["messages"][0]
    rendered = system_msg.content if hasattr(system_msg, "content") else str(system_msg)
    assert "TOOL BUDGET NOTICE" in rendered
    assert "Tool budget reached. Synthesise now." in rendered
    assert "Use the search_documents tool" not in captured["messages"][-1].content

    assert response.metadata.get("rag_force_final_response") is True
    assert response.message.content == "final answer"


@pytest.mark.asyncio
async def test_rag_agent_preserves_current_assistant_tool_group_without_synthetic_human_text():
    agent = object.__new__(RAGAgent)
    agent.settings = type("S", (), {"agentic_preview_chars": 500})()
    agent.agentic_max_iterations = 5
    agent.tools = [SimpleNamespace(name="search_documents")]
    agent.mcp_manager = None
    agent._tools_generation_seen = -1
    captured = {}

    async def fake_invoke(**kwargs):
        captured.update(kwargs)
        return AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="answer"),
            metadata={},
        )

    from app.core.runtime_modeling import ResolvedRuntimeModelConfig

    runtime_config = ResolvedRuntimeModelConfig(
        agent_key="rag",
        provider="gemini",
        model="gemini-2.5-flash",
        temperature=0.7,
        api_key=None,
        key_source="settings",
        source="agent_default",
        capabilities={"supports_vision": False},
        fallback_config=None,
        warnings=[],
        provider_fallback=None,
        is_custom_model=False,
    )
    agent._invoke_agentic_rag_model = fake_invoke
    agent._resolve_runtime_model_config = lambda *a, **kw: runtime_config
    agent._create_fallback_runtime_config = lambda *a, **kw: None
    agent._build_skills_suffix = lambda **kw: ""
    agent._get_tools_for_binding = lambda **kw: []

    tool_call = AIMessage(
        content="",
        tool_calls=[{"id": "search-1", "name": "search_documents", "args": {}}],
    )
    evidence = ToolMessage(
        content="BEGIN UNTRUSTED EVIDENCE E1\nbounded\nEND UNTRUSTED EVIDENCE E1",
        tool_call_id="search-1",
        name="search_documents",
    )
    msg = AgentMessage(
        role=MessageRole.USER,
        content="What is revenue?",
        metadata={
            "original_query": "What is revenue?",
            "rag_tool_messages": [tool_call, evidence],
        },
    )

    response = await agent._process_message_agentic(msg, "conv-1")

    assert response.message.content == "answer"
    emitted = captured["messages"]
    assert isinstance(emitted[-3], HumanMessage)
    assert isinstance(emitted[-2], AIMessage)
    assert isinstance(emitted[-1], ToolMessage)
    assert all("Previous Tool Results" not in str(item.content) for item in emitted)


@pytest.mark.asyncio
async def test_a_scoped_rag_worker_binds_only_its_allowed_tools():
    """A dispatch's ``allowed_tool_ids`` narrows what the model is offered."""
    from app.core.runtime_modeling import ResolvedRuntimeModelConfig

    agent = object.__new__(RAGAgent)
    agent.settings = type("S", (), {"agentic_preview_chars": 500})()
    agent.agentic_max_iterations = 5
    agent.tools = [SimpleNamespace(name="search_documents")]
    agent.mcp_manager = None
    agent._tools_generation_seen = -1
    captured = {}

    async def fake_invoke(**kwargs):
        captured.update(kwargs)
        return AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="answer"),
            metadata={},
        )

    agent._invoke_agentic_rag_model = fake_invoke
    agent._resolve_runtime_model_config = lambda *a, **kw: ResolvedRuntimeModelConfig(
        agent_key="rag",
        provider="gemini",
        model="gemini-2.5-flash",
        temperature=0.7,
        api_key=None,
        key_source="settings",
        source="agent_default",
        capabilities={"supports_vision": False},
    )
    agent._create_fallback_runtime_config = lambda *a, **kw: None
    agent._build_skills_suffix = lambda **kw: ""
    agent._get_tools_for_binding = lambda **kw: [
        SimpleNamespace(name="search_documents", metadata={}),
        SimpleNamespace(name="delete_file", metadata={}),
    ]

    await agent._process_message_agentic(
        AgentMessage(
            role=MessageRole.USER,
            content="What is revenue?",
            metadata={
                "original_query": "What is revenue?",
                "allowed_tool_ids": ("search_documents",),
            },
        ),
        "conv-1",
    )

    assert [tool.name for tool in captured["tools"]] == ["search_documents"]


@pytest.mark.asyncio
async def test_a_failed_rag_model_call_does_not_publish_the_exception_text():
    """The error content becomes the RAG answer, which is streamed and persisted."""
    from app.core.runtime_modeling import ResolvedRuntimeModelConfig

    agent = object.__new__(RAGAgent)
    agent.settings = type("S", (), {"agentic_preview_chars": 500})()
    agent.agentic_max_iterations = 5
    agent.tools = [SimpleNamespace(name="search_documents")]
    agent.mcp_manager = None
    agent._tools_generation_seen = -1

    async def failing_invoke(**_kwargs):
        raise RuntimeError("401 https://api.example.test/v1?key=sk-live-SECRET123")

    agent._invoke_agentic_rag_model = failing_invoke
    agent._resolve_runtime_model_config = lambda *a, **kw: ResolvedRuntimeModelConfig(
        agent_key="rag",
        provider="gemini",
        model="gemini-2.5-flash",
        temperature=0.7,
        api_key=None,
        key_source="settings",
        source="agent_default",
        capabilities={"supports_vision": False},
    )
    agent._create_fallback_runtime_config = lambda *a, **kw: None
    agent._build_skills_suffix = lambda **kw: ""
    agent._get_tools_for_binding = lambda **kw: []

    msg = AgentMessage(
        role=MessageRole.USER,
        content="What is revenue?",
        metadata={"original_query": "What is revenue?"},
    )

    response = await agent._process_message_agentic(msg, "conv-1")

    assert "SECRET123" not in response.message.content
    assert "RuntimeError" in response.message.content
    # ``error`` and the metadata copy are persisted too, so they carry the
    # type rather than the provider's text.
    assert response.error == "RuntimeError"
    assert response.metadata["error"] == "RuntimeError"
    assert "SECRET123" not in response.model_dump_json()


@pytest.mark.asyncio
async def test_execute_search_documents_action_returns_compact_error_for_unknown_action():
    import json
    from types import SimpleNamespace

    from app.ai.rag_tool_actions import execute_search_documents_action

    result, action, evidence = await execute_search_documents_action(
        rag_agent=SimpleNamespace(),
        conversation_id="conv-1",
        user_id="user-1",
        tool_args={"action": "not_a_real_action"},
        context={},
        max_agentic_images=3,
    )

    payload = json.loads(result)
    assert action == "not_a_real_action"
    assert evidence == {}
    assert payload == {
        "status": "error",
        "error_type": "validation",
        "retryable": False,
        "message": "search_documents rejected the requested action.",
        "hint": "Use one of the supported document exploration actions from the tool schema.",
    }


@pytest.mark.asyncio
async def test_an_unexpected_action_failure_is_compact_and_hides_the_raw_message():
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from app.ai.rag_tool_actions import execute_search_documents_action

    rag_agent = SimpleNamespace(
        scan_all_documents=AsyncMock(
            side_effect=RuntimeError("connection to postgresql://svc:hunter2@10.0.0.5 failed")
        )
    )

    result, action, _ = await execute_search_documents_action(
        rag_agent=rag_agent,
        conversation_id="conv-1",
        user_id="user-1",
        tool_args={"action": "scan_all"},
        context={},
        max_agentic_images=3,
    )

    payload = json.loads(result)
    assert action == "scan_all"
    assert payload["status"] == "error"
    assert payload["error_type"] == "unknown"
    assert "RuntimeError" in payload["message"]
    assert "hunter2" not in result and "10.0.0.5" not in result


@pytest.mark.asyncio
async def test_list_documents_returns_one_server_bounded_page_with_total_count():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from app.ai.rag_tool_actions import execute_search_documents_action

    documents_page = {
        "documents": [
            {"document_id": "doc-1", "filename": "one.pdf", "chunk_count": 2},
        ],
        "total": 76,
        "page": 3,
        "page_size": 25,
    }
    rag_agent = SimpleNamespace(list_conversation_documents=AsyncMock(return_value=documents_page))

    result, action, evidence = await execute_search_documents_action(
        rag_agent=rag_agent,
        conversation_id="conv-1",
        user_id="user-1",
        tool_args={"action": "list_documents", "page": 3, "page_size": 999},
        context={},
        max_agentic_images=3,
    )

    assert action == "list_documents"
    assert "Page 3" in result
    assert evidence["pagination"] == {
        "page": 3,
        "page_size": 25,
        "total": 76,
        "next_page": 4,
    }
    rag_agent.list_conversation_documents.assert_awaited_once_with(
        "conv-1",
        user_id="user-1",
        page=3,
        page_size=25,
    )


@pytest.mark.asyncio
async def test_list_documents_empty_page_still_returns_pagination():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from app.ai.rag_tool_actions import execute_search_documents_action

    rag_agent = SimpleNamespace(
        list_conversation_documents=AsyncMock(
            return_value={
                "documents": [],
                "total": 7,
                "page": 2,
                "page_size": 5,
            }
        )
    )

    result, _, evidence = await execute_search_documents_action(
        rag_agent=rag_agent,
        conversation_id="conv-1",
        user_id="user-1",
        tool_args={"action": "list_documents", "page": 2, "page_size": 5},
        context={},
        max_agentic_images=3,
    )

    assert "Page 2" in result
    assert "0 of 7 documents" in result
    assert evidence["pagination"] == {
        "page": 2,
        "page_size": 5,
        "total": 7,
        "next_page": None,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["search_chunks", "view_images"])
@pytest.mark.parametrize(
    ("conversation_id", "user_id"),
    [(None, None), ("conv-1", None), (None, "user-1")],
)
async def test_retrieval_actions_reject_missing_server_scope_without_querying(
    action, conversation_id, user_id
):
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from app.ai.rag_tool_actions import execute_search_documents_action

    rag_agent = SimpleNamespace(
        _search=AsyncMock(),
        get_document_images=AsyncMock(),
    )
    tool_args = {"action": action}
    if action == "search_chunks":
        tool_args["query"] = "revenue"
    else:
        tool_args["document_id"] = "doc-1"

    result, normalized_action, evidence = await execute_search_documents_action(
        rag_agent=rag_agent,
        conversation_id=conversation_id,
        user_id=user_id,
        tool_args=tool_args,
        context={},
        max_agentic_images=3,
    )

    payload = json.loads(result)
    assert normalized_action == action
    assert evidence == {}
    assert payload["error_type"] == "validation"
    assert "authenticated user and conversation context" in payload["message"]
    rag_agent._search.assert_not_awaited()
    rag_agent.get_document_images.assert_not_awaited()
