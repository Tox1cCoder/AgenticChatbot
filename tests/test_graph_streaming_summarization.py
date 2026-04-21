from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.modules.setdefault("qdrant_client", MagicMock())
sys.modules.setdefault("qdrant_client.models", MagicMock())
sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("langchain_openai", MagicMock())
sys.modules.setdefault("langchain", MagicMock())
sys.modules.setdefault("langchain.agents", MagicMock())

from app.ai.graph import MultiAgentWorkflow
from app.ai.schemas import AgentResponse, AgentType, MessageRole, WorkflowExecutionRequest


def _build_response(content: str = "Hello") -> AgentResponse:
    from app.ai.schemas import AgentMessage

    return AgentResponse(
        agent_type=AgentType.CHAT,
        agent_id="chat_agent",
        message=AgentMessage(role=MessageRole.ASSISTANT, content=content),
        metadata={},
    )


@pytest.mark.asyncio
async def test_summarization_node_skips_when_streaming_defers_summary(monkeypatch):
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    state = {
        "conversation_id": "conv-1",
        "context": {"defer_summarization_until_after_stream": True},
    }

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("summarize_for_state should not run on the streaming hot path")

    monkeypatch.setattr("app.ai.graph.summarize_for_state", fail_if_called)

    result = await workflow._summarization_node(state)

    assert result is state


@pytest.mark.asyncio
async def test_execute_request_stream_marks_state_to_defer_summarization(monkeypatch):
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.checkpointer = None

    async def fake_route_node(state):
        state["selected_agent"] = "chat_agent"
        return state

    async def fake_astream(current_state, config=None, stream_mode=None):
        assert current_state["context"]["defer_summarization_until_after_stream"] is True
        yield (
            "messages",
            (
                SimpleNamespace(
                    content_blocks=[{"type": "text", "text": "Hello"}],
                    chunk_position="last",
                ),
                {},
            ),
        )

    workflow._route_node = fake_route_node
    workflow.graph = SimpleNamespace(astream=fake_astream)
    workflow._get_conversation_history = AsyncMock(return_value=[])
    workflow._recover_terminal_response = lambda *args, **kwargs: _build_response()

    request = WorkflowExecutionRequest(
        message="Hi",
        conversation_id="conv-1",
        user_id="00000000-0000-0000-0000-000000000001",
    )

    events = [event async for event in workflow.execute_request_stream(request)]

    assert [event["type"] for event in events] == ["agent_selected", "token", "complete"]


@pytest.mark.asyncio
async def test_execute_request_stream_runs_deferred_summarization_after_stream(monkeypatch):
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.checkpointer = object()

    async def fake_route_node(state):
        state["selected_agent"] = "chat_agent"
        return state

    async def fake_astream(current_state, config=None, stream_mode=None):
        yield (
            "messages",
            (
                SimpleNamespace(
                    content_blocks=[{"type": "text", "text": "Hello"}],
                    chunk_position="last",
                ),
                {},
            ),
        )

    async def fake_aget_state(config):
        return snapshot

    snapshot = SimpleNamespace(
        next=[],
        values={
            "selected_agent": "chat_agent",
            "context": {"defer_summarization_until_after_stream": True},
        },
    )

    called = False

    async def fake_deferred_summary(config, final_state, conversation_id):
        nonlocal called
        called = True

    workflow._route_node = fake_route_node
    workflow.graph = SimpleNamespace(astream=fake_astream, aget_state=fake_aget_state)
    workflow._get_conversation_history = AsyncMock(return_value=[])
    workflow._recover_terminal_response = lambda *args, **kwargs: _build_response()
    workflow._persist_deferred_stream_summarization = fake_deferred_summary

    request = WorkflowExecutionRequest(
        message="Hi",
        conversation_id="conv-1",
        user_id="00000000-0000-0000-0000-000000000001",
    )

    events = [event async for event in workflow.execute_request_stream(request)]

    assert events[-1]["type"] == "complete"
    assert called is True
