"""Phase 4 (Task 4.1-4.3): RAG must produce a final no-tools synthesis pass
when the agentic tool-loop budget is exhausted.

Behaviour under test:
* ``_should_continue_rag`` routes back to ``rag_agent`` once more after the
  budget is reached, marking the context so RAG knows to disable tool binding.
* If the final pass also returns tool calls, the graph routes to ``end`` to
  avoid an infinite loop.
* ``RAGAgent._process_message_agentic`` honours the
  ``rag_force_final_response`` flag by invoking ``_invoke_agentic_rag_model``
  with ``disable_tools=True``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.ai.agents.rag_agent import RAGAgent
from app.ai.graph import MultiAgentWorkflow
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from app.core.config import settings


def _make_workflow():
    return MultiAgentWorkflow.__new__(MultiAgentWorkflow)


@pytest.mark.asyncio
async def test_rag_handoff_routes_to_search_agent_in_same_turn(monkeypatch):
    """RAG must transfer immediately instead of consuming its local loop budget."""
    import app.ai.workflow.rag_loop as rag_loop

    workflow = _make_workflow()
    workflow.rag_agent = object()
    workflow.agents = {
        "rag_agent": SimpleNamespace(agent_config_key="rag"),
        "search_agent": object(),
    }

    async def no_approval(*args, **kwargs):
        return False

    async def tool_map(*args, **kwargs):
        return {"hand_off": SimpleNamespace(name="hand_off")}

    async def execute_tools(**kwargs):
        return (
            [
                {
                    "tool_call_id": "handoff-1",
                    "name": "hand_off",
                    "content": '{"hand_off":"search_agent"}',
                }
            ],
            [],
            [],
        )

    workflow._needs_approval = no_approval
    workflow._update_tool_error_streak = lambda *args, **kwargs: None
    monkeypatch.setattr(rag_loop, "ensure_agent_tool_map", tool_map)
    monkeypatch.setattr(rag_loop, "execute_tool_calls", execute_tools)

    state = {
        "selected_agent": "rag_agent",
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"id": "handoff-1", "name": "hand_off", "args": {}}],
            )
        ],
        "context": {},
        "custom_agents": {},
    }

    await workflow._rag_tools_node(state)

    assert state["selected_agent"] == "search_agent"
    assert workflow._should_continue_rag(state) == "search_agent"


def test_rag_budget_routes_to_final_no_tool_pass(monkeypatch):
    monkeypatch.setattr(
        "app.ai.graph.settings.agentic_max_iterations",
        2,
        raising=False,
    )

    workflow = _make_workflow()
    state = {
        "messages": [ToolMessage(content="chunk text", tool_call_id="call-1")],
        "context": {"agentic_rag_iteration": 2},
    }

    route = workflow._should_continue_rag(state)

    assert route == "rag_agent", (
        "Budget exhausted with a tool result available — RAG should run "
        "one final no-tools synthesis pass instead of ending immediately."
    )
    assert state["context"].get("rag_force_final_response") is True
    assert state["context"].get("rag_tool_budget_notice"), (
        "A budget notice must be set so the RAG agent can append it to the prompt"
    )


def test_rag_budget_already_forced_routes_to_end_to_avoid_loop(monkeypatch):
    """If the final pass still emits tool calls, the second visit must end."""
    monkeypatch.setattr(
        "app.ai.graph.settings.agentic_max_iterations",
        2,
        raising=False,
    )

    workflow = _make_workflow()
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"id": "call-2", "name": "search_documents", "args": {}}],
            )
        ],
        "context": {
            "agentic_rag_iteration": 3,
            "rag_force_final_response": True,
            "rag_tool_budget_notice": "budget reached",
        },
    }

    route = workflow._should_continue_rag(state)
    assert route == "end", (
        "rag_force_final_response=True means we already gave the model the "
        "final pass; if it still asks for tools, the graph must end."
    )


def test_forced_final_assistant_tool_calls_do_not_route_to_rag_tools():
    workflow = _make_workflow()
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"id": "call-2", "name": "search_documents", "args": {}}],
            )
        ],
        "context": {"rag_force_final_response": True},
    }

    route = workflow._should_call_rag_tools(state)

    assert route == "end"


def test_rag_budget_not_yet_reached_continues_normally(monkeypatch):
    monkeypatch.setattr(
        "app.ai.graph.settings.agentic_max_iterations",
        5,
        raising=False,
    )

    workflow = _make_workflow()
    state = {
        "messages": [ToolMessage(content="x", tool_call_id="call-1")],
        "context": {"agentic_rag_iteration": 2},
    }

    route = workflow._should_continue_rag(state)
    assert route == "rag_agent"
    assert "rag_force_final_response" not in state["context"]


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


def test_rag_document_tool_results_are_recorded_as_response_artifacts(monkeypatch):
    workflow = _make_workflow()
    workflow.rag_agent = object()
    workflow.agents = {}

    async def fake_execute_search_documents_action(**kwargs):
        action = kwargs["tool_args"]["action"]
        if action == "search_chunks":
            return "SEARCH RESULTS:\n\n[1] report.pdf\nchunk evidence", action, {}
        if action == "read_document":
            return "DOCUMENT CONTENT (doc-1):\n\nfull document text", action, {}
        raise AssertionError(f"Unexpected action: {action}")

    monkeypatch.setattr(
        "app.ai.workflow.rag_loop.execute_search_documents_action",
        fake_execute_search_documents_action,
    )

    state = {
        "conversation_id": "conv-1",
        "user_id": "user-1",
        "context": {},
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "chunk-call",
                        "name": "search_documents",
                        "args": {"action": "search_chunks", "query": "revenue"},
                    },
                    {
                        "id": "document-call",
                        "name": "search_documents",
                        "args": {"action": "read_document", "document_id": "doc-1"},
                    },
                ],
            )
        ],
        "response": AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="final"),
            metadata={},
        ),
    }

    asyncio.run(workflow._rag_tools_node(state))

    artifacts = state["context"].get("tool_artifacts")
    assert artifacts and len(artifacts) == 2
    assert artifacts[0]["tool_call_id"] == "chunk-call"
    assert artifacts[0]["tool"] == "search_documents"
    assert artifacts[0]["args"]["action"] == "search_chunks"
    assert "chunk evidence" in artifacts[0]["output"]
    assert artifacts[1]["tool_call_id"] == "document-call"
    assert artifacts[1]["args"]["action"] == "read_document"
    assert "full document text" in artifacts[1]["output"]

    recovered = workflow._recover_terminal_response(state)
    assert recovered is not None
    assert recovered.tool_artifacts == artifacts


def test_rag_search_passes_authoritative_allowance_and_persists_pack(monkeypatch):
    workflow = _make_workflow()
    workflow.rag_agent = object()
    workflow.agents = {}
    captured = {}

    async def fake_execute_search_documents_action(**kwargs):
        captured.update(kwargs)
        pack = {
            "records": [{"evidence_id": "E1", "content": "bounded"}],
            "evidence_ids": ["E1"],
            "token_count": 7,
            "omitted_count": 0,
            "truncated_count": 0,
            "count_strategy": "test",
        }
        return (
            "BEGIN UNTRUSTED EVIDENCE E1\nbounded\nEND UNTRUSTED EVIDENCE E1",
            "search_chunks",
            pack,
        )

    monkeypatch.setattr(
        "app.ai.workflow.rag_loop.execute_search_documents_action",
        fake_execute_search_documents_action,
    )
    monkeypatch.setattr(
        "app.ai.workflow.rag_loop.apply_tool_output_offload",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("bounded evidence serialization must remain the ToolMessage")
        ),
    )
    state = {
        "conversation_id": "conv-1",
        "user_id": "owner",
        "context": {},
        "messages": [
            HumanMessage(content="What is revenue?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "search-1",
                        "name": "search_documents",
                        "args": {"action": "search_chunks", "query": "revenue"},
                    }
                ],
            ),
        ],
        "response": AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content=""),
            metadata={
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "request_budget": {"evidence_token_allowance": 321},
            },
        ),
    }

    asyncio.run(workflow._rag_tools_node(state))

    assert captured["question"] == "What is revenue?"
    assert captured["evidence_max_tokens"] == 321
    assert captured["evidence_provider"] == "gemini"
    assert captured["evidence_model"] == "gemini-2.5-flash"
    tool_message = state["messages"][-1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content.startswith("BEGIN UNTRUSTED EVIDENCE E1")
    artifact = state["context"]["tool_artifacts"][0]
    assert artifact["rag_evidence"]["records"][0]["evidence_id"] == "E1"


def test_rag_zero_allowance_does_not_fall_back_to_independent_budget(monkeypatch):
    workflow = _make_workflow()
    workflow.rag_agent = object()
    workflow.agents = {}
    captured = {}

    async def fake_execute_search_documents_action(**kwargs):
        captured.update(kwargs)
        return "", "search_chunks", {
            "records": [],
            "evidence_ids": [],
            "token_count": 0,
            "omitted_count": 1,
            "truncated_count": 0,
        }

    monkeypatch.setattr(
        "app.ai.workflow.rag_loop.execute_search_documents_action",
        fake_execute_search_documents_action,
    )
    state = {
        "conversation_id": "conv-1",
        "user_id": "owner",
        "context": {},
        "messages": [
            HumanMessage(content="question"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "search-1",
                        "name": "search_documents",
                        "args": {"action": "search_chunks", "query": "question"},
                    }
                ],
            ),
        ],
        "response": AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content=""),
            metadata={"request_budget": {"evidence_token_allowance": 0}},
        ),
    }

    asyncio.run(workflow._rag_tools_node(state))

    assert captured["evidence_max_tokens"] == 0
    assert state["messages"][-1].content == ""


def test_rag_search_calls_share_one_cumulative_evidence_allowance(monkeypatch):
    workflow = _make_workflow()
    workflow.rag_agent = object()
    workflow.agents = {}
    allowances: list[int] = []

    async def fake_execute_search_documents_action(**kwargs):
        allowance = kwargs["evidence_max_tokens"]
        allowances.append(allowance)
        used = min(60, allowance)
        return "bounded", "search_chunks", {
            "records": [{"evidence_id": f"E{len(allowances)}"}],
            "evidence_ids": [f"E{len(allowances)}"],
            "token_count": used,
            "omitted_count": 0,
            "truncated_count": 0,
        }

    monkeypatch.setattr(
        "app.ai.workflow.rag_loop.execute_search_documents_action",
        fake_execute_search_documents_action,
    )
    state = {
        "conversation_id": "conv-1",
        "user_id": "owner",
        "context": {},
        "messages": [
            HumanMessage(content="question"),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "s1", "name": "search_documents", "args": {}},
                    {"id": "s2", "name": "search_documents", "args": {}},
                ],
            ),
        ],
        "response": AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content=""),
            metadata={"request_budget": {"evidence_token_allowance": 100}},
        ),
    }

    asyncio.run(workflow._rag_tools_node(state))

    assert allowances == [100, 40]


def test_mixed_rag_actions_charge_non_pack_content_before_later_search(monkeypatch):
    workflow = _make_workflow()
    workflow.rag_agent = object()
    workflow.agents = {}
    search_allowances: list[int] = []

    async def fake_execute_search_documents_action(**kwargs):
        action = kwargs["tool_args"]["action"]
        if action == "list_documents":
            return "one two three four", action, {"documents": []}
        search_allowances.append(kwargs["evidence_max_tokens"])
        return "pack", action, {
            "records": [{"evidence_id": "E1"}],
            "evidence_ids": ["E1"],
            "token_count": 1,
            "omitted_count": 0,
            "truncated_count": 0,
        }

    class FourWordCounter:
        def count_text(self, **kwargs):
            return SimpleNamespace(tokens=len(kwargs["text"].split()), strategy="words")

    monkeypatch.setattr(
        "app.ai.workflow.rag_loop.execute_search_documents_action",
        fake_execute_search_documents_action,
    )
    state = {
        "conversation_id": "conv-1",
        "user_id": "owner",
        "context": {},
        "messages": [
            HumanMessage(content="question"),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "l1", "name": "search_documents", "args": {"action": "list_documents"}},
                    {"id": "s1", "name": "search_documents", "args": {"action": "search_chunks"}},
                ],
            ),
        ],
        "response": AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content=""),
            metadata={
                "request_budget": {"evidence_token_allowance": 100},
                "_evidence_token_counter": FourWordCounter(),
            },
        ),
    }

    asyncio.run(workflow._rag_tools_node(state))

    assert search_allowances == [96]


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


def test_rag_action_named_tool_call_is_canonicalized_to_search_documents(monkeypatch):
    workflow = _make_workflow()
    workflow.rag_agent = object()
    workflow.agents = {}

    async def fake_execute_search_documents_action(**kwargs):
        assert kwargs["tool_args"] == {
            "document_id": "doc-1",
            "pattern": "gabapentin",
            "action": "grep_document",
        }
        return "MATCHES for 'gabapentin' in document:\n\n1. gabapentin", "grep_document", {}

    async def fail_execute_tool_calls(**_kwargs):
        raise AssertionError("RAG document actions must not use generic deferred tool execution")

    monkeypatch.setattr(
        "app.ai.workflow.rag_loop.execute_search_documents_action",
        fake_execute_search_documents_action,
    )
    monkeypatch.setattr("app.ai.workflow.rag_loop.execute_tool_calls", fail_execute_tool_calls)

    state = {
        "conversation_id": "conv-1",
        "user_id": "user-1",
        "context": {},
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "grep-call",
                        "name": "grep_document",
                        "args": {"document_id": "doc-1", "pattern": "gabapentin"},
                    }
                ],
            )
        ],
        "response": AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="final"),
            metadata={},
        ),
    }

    asyncio.run(workflow._rag_tools_node(state))

    artifacts = state["context"].get("tool_artifacts")
    assert artifacts and len(artifacts) == 1
    assert artifacts[0]["tool"] == "search_documents"
    assert artifacts[0]["args"]["action"] == "grep_document"
    assert "gabapentin" in artifacts[0]["output"]

    tool_message = state["messages"][-1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.name == "search_documents"


@pytest.mark.asyncio
async def test_rag_tools_node_tracks_document_tool_error_streak(monkeypatch):
    monkeypatch.setattr(settings, "tool_execution_consecutive_errors_limit", 2, raising=False)
    graph = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    graph.rag_agent = object()
    graph.agents = {}

    async def fake_execute_search_documents_action(**kwargs):
        return "Error: Unknown action: nope", "nope", {}

    monkeypatch.setattr(
        "app.ai.workflow.rag_loop.execute_search_documents_action",
        fake_execute_search_documents_action,
    )
    monkeypatch.setattr(
        "app.ai.workflow.rag_loop.apply_tool_output_offload",
        lambda **kwargs: (kwargs["output_text"], None),
    )

    state = {
        "conversation_id": "conv-1",
        "user_id": "user-1",
        "context": {},
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "rag-call-1",
                        "name": "search_documents",
                        "args": {"action": "nope"},
                    }
                ],
            )
        ],
    }

    await graph._rag_tools_node(state)

    streak = state["context"]["tool_error_streak"]
    assert streak["count"] == 1
    assert streak["signature"]["tool"] == "search_documents"


def test_rag_repeated_error_forces_final_response(monkeypatch):
    monkeypatch.setattr(settings, "tool_execution_consecutive_errors_limit", 2, raising=False)
    graph = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    state = {
        "context": {
            "tool_error_streak": {
                "count": 2,
                "limit": 2,
                "signature": {
                    "tool": "search_documents",
                    "error_type": "validation",
                    "args": '{"action":"not_a_real_action"}',
                },
            }
        }
    }

    decision = graph._should_continue_rag(state)

    assert decision == "rag_agent"
    assert state["context"]["rag_force_final_response"] is True
    assert "repeated tool errors" in state["context"]["rag_tool_budget_notice"].lower()


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
    rag_agent = SimpleNamespace(
        list_conversation_documents=AsyncMock(return_value=documents_page)
    )

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
