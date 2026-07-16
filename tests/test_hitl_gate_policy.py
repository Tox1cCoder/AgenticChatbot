"""The graph gate honors a per-turn per-user policy and resolves server provenance."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from app.ai import graph as graph_module
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from app.ai.workflow import tool_loop as tool_loop_module


class _FakeManager:
    def __init__(self, mapping):
        self._mapping = mapping

    def get_server_for_tool(self, tool):
        return self._mapping.get(id(tool))


def _workflow_stub(tool_map, manager, monkeypatch):
    """A bare object exposing just what _needs_approval / _should_call_tools touch."""
    wf = graph_module.MultiAgentWorkflow.__new__(graph_module.MultiAgentWorkflow)
    agent = object()
    wf.agents = {"chat_agent": agent}

    async def _fake_tool_map(*_a, **_k):
        return tool_map

    async def _fake_manager():
        return manager

    # ``_run_agent_in_isolated_context`` still lives in ``app.ai.graph``; the
    # tool/HITL helpers (``_needs_approval`` etc.) were relocated to
    # ``app.ai.workflow.tool_loop`` (Task 8), so patch the names where each
    # consumer now resolves them.
    monkeypatch.setattr(graph_module, "ensure_agent_tool_map", _fake_tool_map)
    monkeypatch.setattr(tool_loop_module, "ensure_agent_tool_map", _fake_tool_map)
    monkeypatch.setattr(tool_loop_module, "get_global_mcp_manager", _fake_manager)
    return wf


@pytest.mark.asyncio
async def test_gate_gates_all_tools_from_a_server_via_policy(monkeypatch):
    server_tool = SimpleNamespace(name="search", metadata={})
    tool_map = {"search": server_tool}
    manager = _FakeManager({id(server_tool): "tavily"})
    wf = _workflow_stub(tool_map, manager, monkeypatch)

    state = {
        "selected_agent": "chat_agent",
        "conversation_id": "c1",
        "user_id": "u1",
        "device_id": None,
        "context": {
            "hitl_policy": {
                "master_enabled": True,
                "servers": {"tavily": True},
                "tools": {},
                "global_tools": [],
            }
        },
        "messages": [
            AIMessage(content="", tool_calls=[{"name": "search", "args": {}, "id": "call-1"}])
        ],
    }
    assert await wf._should_call_tools(state) == "approval"


@pytest.mark.asyncio
async def test_gate_lets_tool_override_exempt_a_server_tool(monkeypatch):
    tool = SimpleNamespace(
        name="client__desktop_commander__list_files",
        metadata={
            "server_name": "desktop_commander",
            "qualified_tool_id": "desktop_commander::list_files",
            "tool_origin": "client_mcp",
        },
    )
    tool_map = {tool.name: tool}
    wf = _workflow_stub(tool_map, _FakeManager({}), monkeypatch)

    state = {
        "selected_agent": "chat_agent",
        "conversation_id": "c1",
        "user_id": "u1",
        "device_id": None,
        "context": {
            "hitl_policy": {
                "master_enabled": True,
                "servers": {"desktop_commander": True},
                "tools": {"desktop_commander::list_files": False},
                "global_tools": [],
            }
        },
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "client__desktop_commander__list_files", "args": {}, "id": "call-1"}
                ],
            )
        ],
    }
    assert await wf._should_call_tools(state) == "tools"


@pytest.mark.asyncio
async def test_gate_master_off_never_gates(monkeypatch):
    wf = _workflow_stub({}, _FakeManager({}), monkeypatch)
    state = {
        "selected_agent": "chat_agent",
        "conversation_id": "c1",
        "user_id": "u1",
        "device_id": None,
        "context": {
            "hitl_policy": {
                "master_enabled": False,
                "servers": {"tavily": True},
                "tools": {},
                "global_tools": [],
            }
        },
        "messages": [AIMessage(content="", tool_calls=[{"name": "search", "args": {}, "id": "x"}])],
    }
    assert await wf._should_call_tools(state) == "tools"


@pytest.mark.asyncio
async def test_generic_worker_uses_parent_state_hitl_policy(monkeypatch):
    server_tool = SimpleNamespace(name="search", metadata={})
    tool_map = {"search": server_tool}
    manager = _FakeManager({id(server_tool): "tavily"})
    wf = _workflow_stub(tool_map, manager, monkeypatch)

    class _FakeAgent:
        agent_config_key = "chat"

        async def invoke_model_with_history(self, **_kwargs):
            return AgentResponse(
                agent_type=AgentType.CHAT,
                agent_id="chat_agent",
                message=AgentMessage(
                    role=MessageRole.ASSISTANT,
                    content="",
                    tool_calls=[{"name": "search", "args": {}, "id": "call-1"}],
                ),
                metadata={},
            )

    wf.agents = {"chat_agent": _FakeAgent()}
    parent_state = {
        "conversation_id": "c1",
        "user_id": "u1",
        "device_id": None,
        "context": {
            "hitl_policy": {
                "master_enabled": True,
                "servers": {"tavily": True},
                "tools": {},
                "global_tools": [],
            }
        },
    }

    response = await wf._run_agent_in_isolated_context(
        agent_name="chat_agent",
        task_prompt="use search",
        parent_state=parent_state,
    )
    assert response.metadata["requires_approval"] is True
    assert response.metadata["pause_reason"] == "awaiting_approval"
