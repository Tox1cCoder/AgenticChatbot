from __future__ import annotations

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


def _bound_names(monkeypatch, *, offload_enabled: bool) -> list[str]:
    agent = _BindingTestAgent(agent_config_key="search")
    agent.tools = []
    agent.mcp_manager = None
    monkeypatch.setattr(
        "app.ai.agents.base_agent.settings.tool_result_offload_enabled",
        offload_enabled,
        raising=False,
    )
    monkeypatch.setattr(
        "app.ai.agents.base_agent.should_use_deferred_loading",
        lambda _agent_key: True,
    )
    monkeypatch.setattr(
        "app.ai.agents.base_agent.build_deferred_tool_list",
        lambda **kwargs: list(kwargs.get("internal_tools") or []),
    )
    monkeypatch.setattr(agent, "_get_client_runtime_tools", lambda **kwargs: [])
    monkeypatch.setattr(
        agent, "_get_skills_internal_tools", lambda **kwargs: [SimpleNamespace(name="noop")]
    )
    return [tool.name for tool in agent._get_tools_for_binding(conversation_id="c1")]


def test_read_tool_result_is_bound_when_offload_is_enabled(monkeypatch):
    assert "read_tool_result" in _bound_names(monkeypatch, offload_enabled=True)


def test_read_tool_result_is_absent_when_offload_is_disabled(monkeypatch):
    assert "read_tool_result" not in _bound_names(monkeypatch, offload_enabled=False)


def test_tool_context_prompt_points_at_the_reader():
    from app.ai.prompts import TOOL_CONTEXT_SUFFIX

    assert "read_tool_result" in TOOL_CONTEXT_SUFFIX


def test_tool_context_prompt_offload_bullet_is_tool_neutral():
    """The offload bullet must read correctly for read_file, SQL tools, and
    client skills, not only for a search tool. It must also agree in register
    with the offload notice text in tool_result_blob_service.py, which already
    says "do not re-run the tool".
    """
    from app.ai.prompts import TOOL_CONTEXT_SUFFIX

    assert "search" not in TOOL_CONTEXT_SUFFIX.lower()
    assert "do not re-run the tool" in TOOL_CONTEXT_SUFFIX.lower()
