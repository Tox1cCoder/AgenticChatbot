import json
from types import SimpleNamespace

import pytest

from app.ai.agents.base_agent import BaseAgent
from app.ai.schemas import AgentType
from app.ai.tool_execution import execute_tool_calls


class _BindingTestAgent(BaseAgent):
    def _init_gemini(self) -> None:
        self.gemini_client = None
        self.langchain_model = None

    @property
    def agent_type(self) -> AgentType:
        return AgentType.CHAT

    @property
    def agent_id(self) -> str:
        return "binding-test"

    def _get_base_system_prompt(self) -> str:
        return "binding-test"


def test_client_scoped_binding_excludes_server_mcp_tools(monkeypatch):
    agent = _BindingTestAgent(agent_config_key="search")
    agent.tools = [
        SimpleNamespace(name="time__get_current_time"),
        SimpleNamespace(name="tavily__search"),
    ]
    agent.mcp_manager = object()

    monkeypatch.setattr(
        "app.ai.agents.base_agent.should_use_deferred_loading",
        lambda _agent_key: True,
    )
    monkeypatch.setattr(
        "app.ai.agents.base_agent.build_deferred_tool_list",
        lambda **kwargs: [SimpleNamespace(name="tool_search"), *kwargs["all_mcp_tools"]],
    )
    monkeypatch.setattr(
        agent,
        "_get_client_runtime_tools",
        lambda **kwargs: [SimpleNamespace(name="client__time__get_current_time")],
    )
    monkeypatch.setattr(
        "app.ai.agents.base_agent.get_active_client_runtime_session",
        lambda **kwargs: SimpleNamespace(session_id="session-a"),
    )

    class _DeferredStateStub:
        def get_loaded_client_tools(
            self, conversation_id: str, agent_key: str, device_id=None, session_id=None
        ):
            return [
                SimpleNamespace(
                    tool_name="client__time__get_current_time",
                    device_id=str(device_id),
                )
            ]

    monkeypatch.setattr(
        "app.ai.deferred_tool_state.get_deferred_tool_state",
        lambda: _DeferredStateStub(),
    )

    tools = agent._get_tools_for_binding(
        conversation_id="conversation-1",
        user_id="user-1",
        device_id="device-a",
        tool_scope="client_only",
    )

    assert [tool.name for tool in tools] == [
        "tool_search",
        "client__time__get_current_time",
    ]


def test_client_scoped_real_deferred_binding_omits_web_research(monkeypatch):
    agent = _BindingTestAgent(agent_config_key="search")
    agent.tools = []
    agent.mcp_manager = None
    monkeypatch.setattr(
        "app.ai.agents.base_agent.should_use_deferred_loading",
        lambda _agent_key: True,
    )
    monkeypatch.setattr(agent, "_get_client_runtime_tools", lambda **kwargs: [])
    monkeypatch.setattr(agent, "_get_skills_internal_tools", lambda **kwargs: [])

    client_tools = agent._get_tools_for_binding(
        conversation_id="conversation-1",
        user_id="user-1",
        device_id="device-a",
        tool_scope="client_only",
    )
    default_tools = agent._get_tools_for_binding(
        conversation_id="conversation-1",
        user_id="user-1",
        device_id="device-a",
        tool_scope="default",
    )

    assert "tool_search" in [tool.name for tool in client_tools]
    assert "web_research" not in [tool.name for tool in client_tools]
    assert "web_research" in [tool.name for tool in default_tools]


def test_client_scope_without_a_device_fails_closed_at_binding(monkeypatch):
    agent = _BindingTestAgent(agent_config_key="search")
    agent.tools = []
    agent.mcp_manager = None
    monkeypatch.setattr(
        "app.ai.agents.base_agent.should_use_deferred_loading",
        lambda _agent_key: True,
    )
    monkeypatch.setattr(agent, "_get_client_runtime_tools", lambda **kwargs: [])
    monkeypatch.setattr(agent, "_get_skills_internal_tools", lambda **kwargs: [])

    tools = agent._get_tools_for_binding(
        conversation_id="conversation-1",
        user_id="user-1",
        device_id=None,
        tool_scope="client_only",
    )

    assert "web_research" not in [tool.name for tool in tools]


@pytest.mark.asyncio
async def test_client_scoped_execution_rejects_stale_web_research_entry():
    class _StaleWebResearch:
        name = "web_research"

        def __init__(self):
            self.calls = 0

        async def ainvoke(self, args):
            self.calls += 1
            return '{"results": [{"title": "server result"}]}'

    stale_tool = _StaleWebResearch()

    outputs, artifacts, _ = await execute_tool_calls(
        tool_calls=[
            {
                "id": "stale-research",
                "name": "web_research",
                "args": {"query": "private device query"},
            }
        ],
        tool_map={"web_research": stale_tool},
        device_id="device-a",
        tool_scope="client_only",
    )

    payload = json.loads(outputs[0]["content"])
    assert stale_tool.calls == 0
    assert payload["error_type"] == "permission"
    assert payload["retryable"] is False
    assert artifacts[0]["status"] == "error"


@pytest.mark.asyncio
async def test_client_scope_without_a_device_rejects_stale_web_research_entry():
    class _StaleWebResearch:
        name = "web_research"

        def __init__(self):
            self.calls = 0

        async def ainvoke(self, args):
            self.calls += 1
            return '{"results": [{"title": "server result"}]}'

    stale_tool = _StaleWebResearch()

    outputs, artifacts, _ = await execute_tool_calls(
        tool_calls=[
            {
                "id": "stale-research",
                "name": "web_research",
                "args": {"query": "private device query"},
            }
        ],
        tool_map={"web_research": stale_tool},
        device_id=None,
        tool_scope="client_only",
    )

    payload = json.loads(outputs[0]["content"])
    assert stale_tool.calls == 0
    assert payload["error_type"] == "permission"
    assert payload["retryable"] is False
    assert artifacts[0]["status"] == "error"


def test_deferred_binding_keeps_graph_injected_hand_off_available(monkeypatch):
    from app.ai.hand_off_tool import create_hand_off_tool

    agent = _BindingTestAgent(agent_config_key="canvas")
    agent.tools = []
    agent.mcp_manager = None

    monkeypatch.setattr(
        "app.ai.agents.base_agent.should_use_deferred_loading",
        lambda _agent_key: True,
    )
    monkeypatch.setattr(
        "app.ai.agents.base_agent.get_available_skill_summaries",
        lambda **kwargs: [],
    )

    hand_off = create_hand_off_tool(source_agent_id="chat_agent", allowed_targets=["search_agent"])
    tools = agent._get_tools_for_binding(
        conversation_id="conversation-1",
        internal_tools=[hand_off],
    )

    assert "hand_off" in [tool.name for tool in tools]
