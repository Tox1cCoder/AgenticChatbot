"""What the chat and search agents actually bind for web work.

Three product tools plus the result reader, and no raw provider under any
name. The combined ``web_research`` tool these replace is gone; a test asserting
its absence is what stops it being reintroduced as a convenience.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.ai.agents.base_agent import BaseAgent
from app.ai.deferred_tool_binding import RAW_WEB_TOOL_NAMES, _get_required_pinned_specs
from app.ai.schemas import AgentType

PRODUCT_WEB_TOOLS = ("web_search", "web_open", "image_search")


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


def _bound_names(monkeypatch, agent_key: str, **context) -> list[str]:
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
    return [
        tool.name for tool in agent._get_tools_for_binding(conversation_id="c1", **context)
    ]


@pytest.mark.parametrize("agent_key", ["chat", "search"])
@pytest.mark.parametrize("tool_name", PRODUCT_WEB_TOOLS)
def test_each_product_tool_is_bound_for_chat_and_search(monkeypatch, agent_key, tool_name):
    assert tool_name in _bound_names(monkeypatch, agent_key)


def test_the_result_reader_is_bound_alongside_them(monkeypatch):
    assert "read_tool_result" in _bound_names(monkeypatch, "search")


def test_the_combined_research_tool_is_gone(monkeypatch):
    assert "web_research" not in _bound_names(monkeypatch, "chat")
    assert "web_research" not in _bound_names(monkeypatch, "search")


def test_product_web_tools_are_not_bound_for_other_agents(monkeypatch):
    names = _bound_names(monkeypatch, "rag")

    assert not set(PRODUCT_WEB_TOOLS) & set(names)


def test_product_web_tools_are_absent_in_client_only_scope(monkeypatch):
    names = _bound_names(monkeypatch, "search", tool_scope="client_only", device_id="device-a")

    assert not set(PRODUCT_WEB_TOOLS) & set(names)


def test_no_raw_provider_survives_the_binding_filter(monkeypatch):
    """The denylist is applied where every tool source has already merged, so a
    raw tool arriving as a pin, a loaded deferred tool, or a client tool is
    dropped the same way."""
    agent = _BindingTestAgent(agent_config_key="search")
    agent.tools = []
    agent.mcp_manager = None
    monkeypatch.setattr("app.ai.agents.base_agent.should_use_deferred_loading", lambda _key: False)
    monkeypatch.setattr(agent, "_get_skills_internal_tools", lambda **kwargs: [])
    monkeypatch.setattr(
        agent,
        "_get_client_runtime_tools",
        lambda **kwargs: [SimpleNamespace(name=name) for name in RAW_WEB_TOOL_NAMES],
    )
    monkeypatch.setattr(
        "app.ai.agents.base_agent.settings.enable_client_runtime_bridge", False, raising=False
    )

    names = {tool.name for tool in agent._get_tools_for_binding(conversation_id="c1")}

    assert names.isdisjoint(RAW_WEB_TOOL_NAMES)


def test_bound_product_tools_carry_only_the_scope_dependency(monkeypatch):
    """Provider-native selection makes these tools independent of billing."""

    captured: list[dict] = []

    def _capture(name):
        def _factory(**kwargs):
            captured.append({"name": name, **kwargs})
            return SimpleNamespace(name=name, metadata={})

        return _factory

    for name, attribute in (
        ("web_search", "create_web_search_tool"),
        ("web_open", "create_web_open_tool"),
        ("image_search", "create_image_search_tool"),
    ):
        monkeypatch.setattr(f"app.ai.agents.base_agent.{attribute}", _capture(name))
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

    assert [entry["name"] for entry in captured] == list(PRODUCT_WEB_TOOLS)
    assert all(set(entry) == {"name", "tool_scope"} for entry in captured)
    assert all(entry["tool_scope"] == "default" for entry in captured)


def test_media_guidance_describes_image_search_only():
    """The model reaches images through image_search and nothing else.

    The guidance spans two texts that are both always in context: the system
    prompt says when a visual is worth having, the tool description says how to
    drive the arguments. Neither may name the raw provider tool.
    """
    from app.ai.prompts import MEDIA_CAPABILITY_SNIPPET
    from app.ai.web_tools import IMAGE_SEARCH_DESCRIPTION

    guidance = f"{MEDIA_CAPABILITY_SNIPPET}\n{IMAGE_SEARCH_DESCRIPTION}"

    assert "image_search" in MEDIA_CAPABILITY_SNIPPET
    assert "web_research" not in guidance
    assert "intent" in IMAGE_SEARCH_DESCRIPTION
    assert "brave_image_search" not in guidance
    assert "include_images" not in guidance
