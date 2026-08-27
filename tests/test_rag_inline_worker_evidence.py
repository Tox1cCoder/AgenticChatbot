from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.ai.agents.rag_agent import RAGAgent
from app.ai.graph import MultiAgentWorkflow
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from app.core.config import settings


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

    monkeypatch.setattr(
        "app.ai.rag_tool_actions.execute_search_documents_action", fake_search_action
    )
    monkeypatch.setattr(
        "app.ai.rag_tool_actions.apply_tool_output_offload",
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


@pytest.mark.asyncio
async def test_inline_worker_charges_non_pack_content_before_later_search(monkeypatch):
    allowances: list[int] = []

    class WordCounter:
        def count_text(self, **kwargs):
            return SimpleNamespace(tokens=len(kwargs["text"].split()), strategy="words")

    responses = [
        AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[
                    {"id": "l1", "name": "search_documents", "args": {"action": "list_documents"}},
                    {"id": "s1", "name": "search_documents", "args": {"action": "search_chunks"}},
                ],
            ),
            metadata={
                "provider": "test",
                "model": "test",
                "request_budget": {"evidence_token_allowance": 50},
                "evidence_tokenization": {
                    "reference": "word-counter",
                    "provider": "test",
                    "model": "test",
                    "fallback": "deterministic_local_conservative",
                },
            },
        ),
        AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="done"),
            metadata={},
        ),
    ]

    async def process_message(_message, _conversation_id):
        return responses.pop(0)

    agent = SimpleNamespace(
        process_message=process_message,
        agent_config_key="rag",
        tool_state_key="rag",
        agent_type=AgentType.RAG,
        _take_evidence_token_counter=lambda descriptor, **_kwargs: WordCounter(),
    )
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.rag_agent = agent
    workflow.agents = {"rag_agent": agent}

    async def fake_action(**kwargs):
        action = kwargs["tool_args"]["action"]
        if action == "list_documents":
            return "one two three", action, {"documents": []}
        allowances.append(kwargs["evidence_max_tokens"])
        return "pack", action, {
            "records": [{"evidence_id": "E1"}],
            "evidence_ids": ["E1"],
            "token_count": 1,
            "omitted_count": 0,
            "truncated_count": 0,
        }

    monkeypatch.setattr("app.ai.rag_tool_actions.execute_search_documents_action", fake_action)
    monkeypatch.setattr(
        "app.ai.rag_tool_actions.apply_tool_output_offload",
        lambda **kwargs: (kwargs["output_text"], None),
    )

    response = await workflow._run_agent_in_isolated_context(
        agent_name="rag_agent",
        task_prompt="question",
        parent_state={"conversation_id": "conv", "user_id": "owner", "context": {}},
    )

    assert response.message.content == "done"
    assert allowances == [47]


@pytest.mark.asyncio
async def test_inline_worker_omits_oversized_non_pack_result_before_next_request(monkeypatch):
    import json

    class CharacterCounter:
        def count_text(self, **kwargs):
            return SimpleNamespace(tokens=len(kwargs["text"]), strategy="characters")

    oversized = json.dumps(
        {"rows": [{"id": index, "value": "x" * 20} for index in range(20)]},
        separators=(",", ":"),
    )
    invocations: list[AgentMessage] = []
    responses = [
        AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[
                    {
                        "id": "list-oversized",
                        "name": "search_documents",
                        "args": {"action": "list_documents"},
                    }
                ],
            ),
            metadata={
                "provider": "test",
                "model": "test",
                "request_budget": {"evidence_token_allowance": 80},
                "evidence_tokenization": {
                    "reference": "character-counter",
                    "provider": "test",
                    "model": "test",
                    "fallback": "deterministic_local_conservative",
                },
            },
        ),
        AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="done"),
            metadata={},
        ),
    ]

    async def process_message(message, _conversation_id):
        invocations.append(message)
        return responses.pop(0)

    agent = SimpleNamespace(
        process_message=process_message,
        agent_config_key="rag",
        tool_state_key="rag",
        agent_type=AgentType.RAG,
        _take_evidence_token_counter=lambda descriptor, **_kwargs: CharacterCounter(),
    )
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.rag_agent = agent
    workflow.agents = {"rag_agent": agent}

    async def fake_action(**kwargs):
        return oversized, kwargs["tool_args"]["action"], {"documents": []}

    monkeypatch.setattr("app.ai.rag_tool_actions.execute_search_documents_action", fake_action)
    monkeypatch.setattr(
        "app.ai.rag_tool_actions.apply_tool_output_offload",
        lambda **kwargs: (kwargs["output_text"], None),
    )

    response = await workflow._run_agent_in_isolated_context(
        agent_name="rag_agent",
        task_prompt="question",
        parent_state={"conversation_id": "conv", "user_id": "owner", "context": {}},
    )

    assert response.message.content == "done"
    result_message = invocations[1].metadata["rag_tool_messages"][-1]
    assert isinstance(result_message, ToolMessage)
    assert len(result_message.content) <= 80
    assert result_message.content != oversized
    assert json.loads(result_message.content)["reason"] == "context_budget"
    assert invocations[1].metadata["tool_context"][-1] == result_message.content


@pytest.mark.asyncio
async def test_inline_worker_keeps_tool_text_when_no_allowance_was_propagated(monkeypatch):
    """Absent ``evidence_token_allowance`` is unknown, not zero.

    The delegated worker feeds the previous result back through both
    ``rag_tool_messages`` and the legacy ``tool_context`` list; blanking it when
    no budget was propagated would hand the model an empty search result.
    """
    invocations: list[AgentMessage] = []
    responses = [
        AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[
                    {
                        "id": "scan-1",
                        "name": "search_documents",
                        "args": {"action": "scan_all"},
                    }
                ],
            ),
            metadata={},
        ),
        AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="done"),
            metadata={},
        ),
    ]

    async def process_message(message, _conversation_id):
        invocations.append(message)
        return responses.pop(0)

    agent = SimpleNamespace(
        process_message=process_message,
        agent_config_key="rag",
        tool_state_key="rag",
        agent_type=AgentType.RAG,
    )
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.rag_agent = agent
    workflow.agents = {"rag_agent": agent}

    async def fake_action(**kwargs):
        return "SCAN RESULT", kwargs["tool_args"]["action"], {"documents": []}

    monkeypatch.setattr("app.ai.rag_tool_actions.execute_search_documents_action", fake_action)
    monkeypatch.setattr(
        "app.ai.rag_tool_actions.apply_tool_output_offload",
        lambda **kwargs: (kwargs["output_text"], None),
    )

    response = await workflow._run_agent_in_isolated_context(
        agent_name="rag_agent",
        task_prompt="question",
        parent_state={"conversation_id": "conv", "user_id": "owner", "context": {}},
    )

    assert response.message.content == "done"
    assert invocations[1].metadata["tool_context"] == ["SCAN RESULT"]
    assert invocations[1].metadata["rag_tool_messages"][-1].content == "SCAN RESULT"


def _descriptor(reference: str) -> dict[str, str]:
    return {
        "reference": reference,
        "provider": "test",
        "model": "test",
        "fallback": "deterministic_local_conservative",
    }


def _searching_response(reference: str) -> AgentResponse:
    return AgentResponse(
        agent_type=AgentType.RAG,
        agent_id="rag_agent",
        message=AgentMessage(
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=[
                {
                    "id": "scan-1",
                    "name": "search_documents",
                    "args": {"action": "scan_all"},
                }
            ],
        ),
        metadata={
            "provider": "test",
            "model": "test",
            "request_budget": {"evidence_token_allowance": 500},
            "evidence_tokenization": _descriptor(reference),
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exit_kind", "action_result", "pause_reason"),
    [
        ("max_iterations", "SCAN RESULT", "worker_max_iterations"),
        ("tool_error_streak", "Error executing scan_all: boom", "consecutive_tool_errors"),
    ],
)
async def test_inline_worker_stop_exits_never_return_counter_descriptor(
    monkeypatch,
    exit_kind,
    action_result,
    pause_reason,
):
    """Every stop exit of the delegated loop must release the counter reference."""
    monkeypatch.setattr(settings, "agentic_max_iterations", 1, raising=False)
    monkeypatch.setattr(settings, "tool_execution_consecutive_errors_limit", 1, raising=False)
    taken: list[dict] = []
    discarded: list[dict] = []
    responses = [_searching_response(f"{exit_kind}-counter")]

    async def process_message(_message, _conversation_id):
        return responses.pop(0)

    agent = SimpleNamespace(
        process_message=process_message,
        agent_config_key="rag",
        tool_state_key="rag",
        agent_type=AgentType.RAG,
        _take_evidence_token_counter=lambda descriptor, **_kwargs: (
            taken.append(descriptor)
            or SimpleNamespace(
                count_text=lambda **kwargs: SimpleNamespace(
                    tokens=len(kwargs["text"].split()), strategy="words"
                )
            )
        ),
        _discard_evidence_token_counter=discarded.append,
    )
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.rag_agent = agent
    workflow.agents = {"rag_agent": agent}

    async def fake_action(**kwargs):
        return action_result, kwargs["tool_args"]["action"], {"documents": []}

    monkeypatch.setattr("app.ai.rag_tool_actions.execute_search_documents_action", fake_action)
    monkeypatch.setattr(
        "app.ai.rag_tool_actions.apply_tool_output_offload",
        lambda **kwargs: (kwargs["output_text"], None),
    )

    response = await workflow._run_agent_in_isolated_context(
        agent_name="rag_agent",
        task_prompt="question",
        parent_state={"conversation_id": "conv", "user_id": "owner", "context": {}},
    )

    assert response.metadata["pause_reason"] == pause_reason
    assert "evidence_tokenization" not in response.metadata
    assert "_evidence_token_counter" not in response.metadata
    assert [entry["reference"] for entry in taken] == [f"{exit_kind}-counter"]
    assert discarded == []
    assert not responses


@pytest.mark.asyncio
async def test_inline_worker_strips_private_counter_when_a_call_is_refused():
    consumed: list[dict] = []
    response = AgentResponse(
        agent_type=AgentType.RAG,
        agent_id="rag_agent",
        message=AgentMessage(
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=[
                {
                    "id": "search-approval",
                    "name": "search_documents",
                    "args": {"action": "search_chunks", "query": "revenue"},
                }
            ],
        ),
        metadata={
            "request_budget": {"evidence_token_allowance": 50},
            "evidence_tokenization": {
                "reference": "approval-counter",
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "fallback": "deterministic_local_conservative",
            },
        },
    )

    # The worker no longer exits early for approval, so the model has to stop
    # asking or the loop runs to its iteration limit.
    replies = [
        response,
        AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="done"),
            metadata={},
        ),
    ]

    async def process_message(_message, _conversation_id):
        return replies.pop(0)

    class _Counter:
        def count_text(self, **kwargs):
            return SimpleNamespace(tokens=len(str(kwargs["text"]).split()), strategy="words")

    agent = SimpleNamespace(
        process_message=process_message,
        agent_config_key="rag",
        tool_state_key="rag",
        agent_type=AgentType.RAG,
        _take_evidence_token_counter=lambda descriptor, **_kwargs: (
            consumed.append(descriptor) or _Counter()
        ),
    )
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.rag_agent = agent
    workflow.agents = {"rag_agent": agent}

    approved = await workflow._run_agent_in_isolated_context(
        agent_name="rag_agent",
        task_prompt="question",
        parent_state={"conversation_id": "conv", "user_id": "owner", "context": {}},
    )

    # The counter is consumed before the approval decision, so the descriptor
    # is stripped whether the call is refused or run. A live counter object
    # surviving in metadata would be checkpointed.
    assert "evidence_tokenization" not in approved.metadata
    assert "_evidence_token_counter" not in approved.metadata
    assert consumed[0]["reference"] == "approval-counter"
