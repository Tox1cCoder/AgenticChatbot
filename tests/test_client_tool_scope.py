from types import SimpleNamespace

from app.ai.agents.base_agent import BaseAgent
from app.ai.schemas import AgentType


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


def test_deferred_binding_keeps_hand_off_available(monkeypatch):
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

    tools = agent._get_tools_for_binding(conversation_id="conversation-1")

    assert "hand_off" in [tool.name for tool in tools]
