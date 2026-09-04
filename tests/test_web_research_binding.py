from __future__ import annotations

from types import SimpleNamespace

from app.ai.agents.base_agent import BaseAgent
from app.ai.deferred_tool_binding import _get_required_pinned_specs
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


def test_tavily_and_brave_are_no_longer_pinned():
    for agent_key in ("chat", "search"):
        specs = _get_required_pinned_specs(agent_key)
        assert "tavily::tavily_search" not in specs
        assert "brave_image_search::brave_image_search" not in specs


def test_time_stays_pinned_for_the_search_agent():
    assert "time::get_current_time" in _get_required_pinned_specs("search")


def _bound_names(monkeypatch, agent_key: str) -> list[str]:
    agent = _BindingTestAgent(agent_config_key=agent_key)
    agent.tools = []
    agent.mcp_manager = None
    monkeypatch.setattr("app.ai.agents.base_agent.should_use_deferred_loading", lambda _key: True)
    monkeypatch.setattr(
        "app.ai.agents.base_agent.build_deferred_tool_list",
        lambda **kwargs: list(kwargs.get("internal_tools") or []),
    )
    monkeypatch.setattr(agent, "_get_client_runtime_tools", lambda **kwargs: [])
    monkeypatch.setattr(agent, "_get_skills_internal_tools", lambda **kwargs: [])
    return [tool.name for tool in agent._get_tools_for_binding(conversation_id="c1")]


def test_web_research_is_bound_for_chat_and_search(monkeypatch):
    assert "web_research" in _bound_names(monkeypatch, "chat")
    assert "web_research" in _bound_names(monkeypatch, "search")


def test_web_research_is_not_bound_for_other_agents(monkeypatch):
    assert "web_research" not in _bound_names(monkeypatch, "rag")


def test_bound_web_research_has_no_usage_recorder_dependency(monkeypatch):
    """Provider-native selection makes the internal tool independent of billing."""

    captured: dict = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(name="web_research", metadata={})

    monkeypatch.setattr("app.ai.agents.base_agent.create_web_research_tool", _capture)
    agent = _BindingTestAgent(agent_config_key="search", recorder=object())
    agent.tools = []
    agent.mcp_manager = None
    monkeypatch.setattr("app.ai.agents.base_agent.should_use_deferred_loading", lambda _key: True)
    monkeypatch.setattr(
        "app.ai.agents.base_agent.build_deferred_tool_list",
        lambda **kwargs: list(kwargs.get("internal_tools") or []),
    )
    monkeypatch.setattr(agent, "_get_client_runtime_tools", lambda **kwargs: [])
    monkeypatch.setattr(agent, "_get_skills_internal_tools", lambda **kwargs: [])

    agent._get_tools_for_binding(conversation_id="c1")

    assert captured == {"tool_scope": "default"}


def test_media_guidance_describes_web_research_only():
    """The model reaches images through web_research and nothing else.

    The guidance spans two texts that are both always in context: the system
    prompt says when a visual is worth having, the tool description says how to
    drive the arguments. Neither may name the raw provider tool or Tavily's own
    image flag.
    """
    from app.ai.prompts import MEDIA_CAPABILITY_SNIPPET
    from app.ai.web_research_tool import _DESCRIPTION

    guidance = f"{MEDIA_CAPABILITY_SNIPPET}\n{_DESCRIPTION}"

    assert "web_research" in MEDIA_CAPABILITY_SNIPPET
    assert "image_query" in guidance
    assert "image_intent" in _DESCRIPTION
    assert "brave_image_search" not in guidance
    assert "include_images" not in guidance
