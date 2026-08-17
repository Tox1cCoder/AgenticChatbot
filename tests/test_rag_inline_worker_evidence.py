from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.ai.agents.rag_agent import RAGAgent
from app.ai.graph import MultiAgentWorkflow
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole


@pytest.mark.asyncio
async def test_reachable_inline_rag_worker_feeds_real_role_group_to_rag_agent(monkeypatch):
    """Exercise the actual RAGAgent message builder through the reachable worker loop."""
    agent = object.__new__(RAGAgent)
    agent.settings = SimpleNamespace(agentic_preview_chars=500)
    agent.agentic_max_iterations = 5
    agent.tools = [SimpleNamespace(name="search_documents")]
    agent.mcp_manager = None
    agent._tools_generation_seen = -1
    agent._init_tools = AsyncMock()
    agent._build_skills_suffix = lambda **_kwargs: ""
    agent._convert_history_to_langchain_messages = lambda _history: []
    agent._get_tools_for_binding = lambda **_kwargs: []
    agent._resolve_runtime_model_config = lambda *_args, **_kwargs: SimpleNamespace(
        provider="test",
        model="test-model",
        capabilities={"supports_vision": False},
    )
    invocations: list[list[object]] = []

    async def fake_invoke(**kwargs):
        invocations.append(list(kwargs["messages"]))
        if len(invocations) == 1:
            return AgentResponse(
                agent_type=AgentType.RAG,
                agent_id="rag_agent",
                message=AgentMessage(
                    role=MessageRole.ASSISTANT,
                    content="searching",
                    tool_calls=[
                        {
                            "id": "search-1",
                            "name": "search_documents",
                            "args": {"action": "search_chunks", "query": "revenue"},
                        }
                    ],
                ),
                metadata={
                    "provider": "test",
                    "model": "test-model",
                    "request_budget": {"evidence_token_allowance": 77},
                },
            )
        return AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="grounded final"),
            metadata={},
        )

    agent._invoke_agentic_rag_model = fake_invoke
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.rag_agent = agent
    workflow.agents = {"rag_agent": agent}
    captured_action: dict = {}

    async def fake_search_action(**kwargs):
        captured_action.update(kwargs)
        return "BEGIN UNTRUSTED EVIDENCE E1\nEND UNTRUSTED EVIDENCE E1", "search_chunks", {
            "records": [{"evidence_id": "E1", "document_id": str(UUID(int=1))}],
            "evidence_ids": ["E1"],
            "token_count": 12,
            "omitted_count": 0,
            "truncated_count": 0,
        }

    monkeypatch.setattr("app.ai.graph.execute_search_documents_action", fake_search_action)
    monkeypatch.setattr(
        "app.ai.graph.apply_tool_output_offload",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("structured evidence must not be generically offloaded")
        ),
    )

    response = await workflow._run_agent_in_isolated_context(
        agent_name="rag_agent",
        task_prompt="What is revenue?",
        parent_state={
            "conversation_id": str(UUID(int=9)),
            "user_id": "owner",
            "device_id": "device-1",
            "context": {},
            "messages": [],
        },
    )

    assert response.message.content == "grounded final"
    assert captured_action["question"] == "What is revenue?"
    assert captured_action["evidence_provider"] == "test"
    assert captured_action["evidence_model"] == "test-model"
    assert captured_action["evidence_max_tokens"] == 77
    assert len(invocations) == 2
    assert isinstance(invocations[1][-2], AIMessage)
    assert invocations[1][-2].tool_calls[0]["id"] == "search-1"
    assert isinstance(invocations[1][-1], ToolMessage)
    assert invocations[1][-1].tool_call_id == "search-1"
    assert "BEGIN UNTRUSTED EVIDENCE E1" in str(invocations[1][-1].content)
    assert response.tool_artifacts[0]["rag_evidence"]["evidence_ids"] == ["E1"]
