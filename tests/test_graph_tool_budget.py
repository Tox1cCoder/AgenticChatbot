from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.ai.agents.base_agent import BaseAgent
from app.ai.graph import MultiAgentWorkflow
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from app.core.config import settings
from app.core.runtime_modeling import ResolvedRuntimeModelConfig


def _tool_call() -> dict:
    return {"id": "tool-call-1", "name": "lookup", "args": {"query": "status"}}


def test_route_tool_output_soft_limit_routes_to_final_synthesis(monkeypatch):
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.agents = {"chat_agent": object()}
    monkeypatch.setattr(settings, "react_agent_max_iterations", 10)
    monkeypatch.setattr(settings, "auto_continue_enabled", True)
    monkeypatch.setattr(settings, "auto_continue_soft_limit_ratio", 0.5)

    state = {
        "selected_agent": "chat_agent",
        "iteration_count": 5,
        "messages": [
            HumanMessage(content="What happened?"),
            AIMessage(content="", tool_calls=[_tool_call()]),
            ToolMessage(content="The lookup result", tool_call_id="tool-call-1", name="lookup"),
        ],
        "context": {},
    }

    route = workflow._route_tool_output(state)

    assert route == "chat_agent"
    assert state["context"]["force_final_response"] is True
    assert state["context"]["tool_budget"]["reason"] == "soft_budget"
    assert state["context"]["tool_budget"]["count"] == 5
    assert state["context"]["tool_budget"]["limit"] == 5


def test_graph_config_clamps_recursion_limit_for_final_synthesis(monkeypatch):
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.checkpointer = None
    monkeypatch.setattr(settings, "react_agent_max_iterations", 10)
    monkeypatch.setattr(settings, "planning_max_iterations", 10)
    monkeypatch.setattr(settings, "react_agent_recursion_limit", 15)

    config = workflow._build_graph_config()

    assert config["recursion_limit"] == 25


def test_graph_config_accounts_for_planning_iteration_budget(monkeypatch):
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.checkpointer = None
    monkeypatch.setattr(settings, "react_agent_max_iterations", 5)
    monkeypatch.setattr(settings, "planning_max_iterations", 20)
    monkeypatch.setattr(settings, "react_agent_recursion_limit", 0)

    config = workflow._build_graph_config()

    assert config["recursion_limit"] == 45


@pytest.mark.asyncio
async def test_chat_node_disables_tools_for_forced_final_response():
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow._get_conversation_history = AsyncMock(return_value=[])
    workflow.chat_agent = SimpleNamespace()
    workflow.chat_agent.invoke_model_with_history = AsyncMock(
        return_value=AgentResponse(
            agent_type=AgentType.CHAT,
            agent_id="chat_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="Final answer"),
            metadata={},
        )
    )

    state = {
        "selected_agent": "chat_agent",
        "messages": [
            HumanMessage(content="What happened?"),
            AIMessage(content="", tool_calls=[_tool_call()]),
            ToolMessage(content="The lookup result", tool_call_id="tool-call-1", name="lookup"),
        ],
        "context": {
            "force_final_response": True,
            "tool_budget": {
                "reason": "soft_budget",
                "scope": "runtime",
                "count": 5,
                "limit": 5,
            },
        },
    }

    result = await workflow._chat_node(state)

    call_kwargs = workflow.chat_agent.invoke_model_with_history.call_args.kwargs
    assert call_kwargs["disable_tools"] is True
    assert "tool-use budget" in call_kwargs["tool_budget_notice"].lower()
    assert result["response"].metadata["tool_budget_exhausted"]["limit"] == 5
    assert "force_final_response" not in result["context"]


class _DummyAgent(BaseAgent):
    def _init_gemini(self) -> None:
        self.gemini_client = None
        self.langchain_model = None

    @property
    def agent_type(self) -> AgentType:
        return AgentType.CHAT

    @property
    def agent_id(self) -> str:
        return "dummy_agent"

    def _get_base_system_prompt(self) -> str:
        return "Dummy system prompt."


class _FakeLLM:
    def __init__(self) -> None:
        self.messages = None
        self.config = None

    async def ainvoke(self, messages, config=None):
        self.messages = messages
        self.config = config
        return AIMessage(content="Final answer from plain model")


@pytest.mark.asyncio
async def test_base_agent_disable_tools_uses_plain_model_and_budget_notice(monkeypatch):
    agent = _DummyAgent(agent_config_key="chat")
    fake_llm = _FakeLLM()

    monkeypatch.setattr(agent, "_init_tools", AsyncMock())
    monkeypatch.setattr(
        agent,
        "_resolve_runtime_model_config",
        lambda _user_id, _model_request=None: ResolvedRuntimeModelConfig(
            agent_key="chat",
            provider="gemini",
            model="dummy-model",
            temperature=0.0,
            api_key=None,
            key_source="none",
            source="test",
            capabilities={"supports_tool_calling": True},
        ),
    )
    monkeypatch.setattr(
        agent,
        "_create_langchain_model_from_runtime",
        lambda _runtime_config, **_kwargs: (fake_llm, False),
    )
    monkeypatch.setattr(
        agent,
        "_get_llm_with_tools",
        Mock(side_effect=AssertionError("tools must not be bound for final synthesis")),
    )
    monkeypatch.setattr(
        agent,
        "_get_tools_for_binding",
        Mock(side_effect=AssertionError("tool schemas must not be counted when disabled")),
    )

    response = await agent.invoke_model_with_history(
        messages=[HumanMessage(content="Use the result")],
        conversation_history=[],
        persona=None,
        disable_tools=True,
        tool_budget_notice="Tool-use budget reached. Produce the final answer now.",
    )

    assert response.message.content == "Final answer from plain model"
    assert isinstance(fake_llm.messages[0], SystemMessage)
    assert "Tool-use budget reached" in fake_llm.messages[0].content


@pytest.mark.asyncio
async def test_base_agent_ainvoke_with_retries_forwards_run_config():
    agent = _DummyAgent(agent_config_key="chat")
    fake_llm = _FakeLLM()
    run_config = {
        "tags": ["internal", "planning_subagent"],
        "metadata": {"internal": True, "purpose": "planning_subagent"},
    }

    await agent._ainvoke_with_retries(
        fake_llm,
        [HumanMessage(content="worker prompt")],
        run_config=run_config,
    )

    assert fake_llm.config is run_config


@pytest.mark.asyncio
async def test_tool_node_resolves_pending_calls_when_tool_map_empty(monkeypatch):
    graph = MultiAgentWorkflow.__new__(MultiAgentWorkflow)

    class _Agent:
        agent_config_key = "chat"
        tool_state_key = "chat"

    graph._resolve_runtime_agent = lambda state, selected: _Agent()

    async def _empty_tool_map(*args, **kwargs):
        return {}

    monkeypatch.setattr("app.ai.workflow.tool_loop.ensure_agent_tool_map", _empty_tool_map)

    state = {
        "selected_agent": "chat_agent",
        "messages": [
            HumanMessage(content="run something"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "missing_tool",
                        "args": {},
                    }
                ],
            ),
        ],
    }

    updated = await graph._tool_node(state)

    assert isinstance(updated["messages"][-1], ToolMessage)
    assert updated["messages"][-1].tool_call_id == "call-1"
    assert "missing_tool" in updated["messages"][-1].content


def test_apply_tool_outputs_tracks_same_error_streak(monkeypatch):
    monkeypatch.setattr(settings, "tool_execution_consecutive_errors_limit", 2, raising=False)
    graph = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    state = {"messages": [], "context": {}}

    artifact = {
        "tool_call_id": "call-1",
        "tool": "read_file",
        "args": {"path": "missing.txt"},
        "status": "error",
        "error_type": "not_found",
        "output": "missing",
    }

    graph._apply_tool_outputs_to_state(
        state,
        tool_outputs=[{"tool_call_id": "call-1", "name": "read_file", "content": "missing"}],
        tool_artifacts=[artifact],
    )
    graph._apply_tool_outputs_to_state(
        state,
        tool_outputs=[{"tool_call_id": "call-2", "name": "read_file", "content": "missing"}],
        tool_artifacts=[{**artifact, "tool_call_id": "call-2"}],
    )

    streak = state["context"]["tool_error_streak"]
    assert streak["count"] == 2
    assert streak["limit"] == 2
    assert streak["signature"]["tool"] == "read_file"
    assert streak["signature"]["args"] == '{"path":"missing.txt"}'


def test_apply_tool_outputs_preserves_flagged_skill_terminal_errors(monkeypatch):
    monkeypatch.setattr(settings, "tool_result_max_chars", 120, raising=False)
    graph = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    state = {"messages": [], "context": {}}
    terminal_content = '{"status":"error","untrusted_terminal_output":"' + "x" * 4000 + '"}'

    graph._apply_tool_outputs_to_state(
        state,
        tool_outputs=[
            {
                "tool_call_id": "call-skill",
                "name": "client__demo__run_skill_command",
                "content": terminal_content,
                "preserve_full_content": True,
            },
            {
                "tool_call_id": "call-2",
                "name": "big_tool",
                "content": "y" * 4000,
            },
        ],
        truncate_outputs=True,
    )

    skill_message, other_message = state["messages"]
    assert skill_message.content == terminal_content
    assert len(other_message.content) <= 120


def test_apply_tool_outputs_lifts_but_does_not_persist_internal_rich_candidates():
    graph = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    state = {"messages": [], "context": {}}
    candidate = {
        "id": "image:tool-content:call-1:0",
        "type": "image",
        "display_policy": "inline_only",
        "payload": {"data": "YWJj", "mime_type": "image/png"},
    }
    artifact = {
        "tool_call_id": "call-1",
        "tool": "image_tool",
        "args": {},
        "status": "success",
        "output": "[image/png image]",
        "_rich_item_candidates": [candidate],
    }

    graph._apply_tool_outputs_to_state(
        state,
        tool_outputs=[
            {
                "tool_call_id": "call-1",
                "name": "image_tool",
                "content": "[image/png image]",
            }
        ],
        tool_artifacts=[artifact],
    )

    assert state["context"]["rich_item_candidates"] == [candidate]
    assert "_rich_item_candidates" not in artifact
    assert "_rich_item_candidates" not in state["context"]["tool_artifacts"][0]
