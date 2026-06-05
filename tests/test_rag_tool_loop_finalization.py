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

from langchain_core.messages import AIMessage, ToolMessage

from app.ai.agents.rag_agent import RAGAgent
from app.ai.graph import MultiAgentWorkflow
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole


def _make_workflow():
    return MultiAgentWorkflow.__new__(MultiAgentWorkflow)


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
        "app.ai.graph.execute_search_documents_action",
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
        "app.ai.graph.execute_search_documents_action",
        fake_execute_search_documents_action,
    )
    monkeypatch.setattr("app.ai.graph.execute_tool_calls", fail_execute_tool_calls)

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
