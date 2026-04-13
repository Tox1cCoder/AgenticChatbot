from __future__ import annotations

from types import SimpleNamespace

from app.ai.deferred_tool_binding import _get_pinned_specs, build_deferred_tool_list
from app.ai.prompts import SEARCH_SYSTEM_PROMPT, SEARCH_WITH_RESULTS_SYSTEM_PROMPT


class _FakeManager:
    def __init__(self, server_by_tool_name: dict[str, str]):
        self._server_by_tool_name = server_by_tool_name

    def get_server_for_tool(self, tool):
        return self._server_by_tool_name.get(tool.name)


def test_search_agent_auto_pins_time_and_tavily_tools(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "mcp_tool_search_pinned_tools", [], raising=False)

    pinned_specs = _get_pinned_specs("search")

    assert "time::get_current_time" in pinned_specs
    assert "tavily::tavily_search" in pinned_specs


def test_search_agent_deferred_binding_includes_time_and_tavily(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "mcp_tool_search_pinned_tools", [], raising=False)

    time_tool = SimpleNamespace(name="get_current_time")
    tavily_tool = SimpleNamespace(name="tavily_search")
    manager = _FakeManager(
        {
            "get_current_time": "time",
            "tavily_search": "tavily",
        }
    )

    tools = build_deferred_tool_list(
        conversation_id="conversation-1",
        agent_key="search",
        mcp_manager=manager,
        all_mcp_tools=[time_tool, tavily_tool],
    )

    tool_names = [tool.name for tool in tools]
    assert "tool_search" in tool_names
    assert "get_current_time" in tool_names
    assert "tavily_search" in tool_names


def test_search_agent_binding_exposes_time_and_tavily_tools(monkeypatch):
    from app.ai.agents.search_agent import SearchAgent
    from app.core.config import settings

    monkeypatch.setattr(settings, "mcp_tool_search_pinned_tools", [], raising=False)
    monkeypatch.setattr(
        "app.ai.agents.base_agent.should_use_deferred_loading",
        lambda _agent_key: True,
    )

    agent = SearchAgent()
    agent.mcp_manager = _FakeManager(
        {
            "get_current_time": "time",
            "tavily_search": "tavily",
        }
    )
    agent.tools = [
        SimpleNamespace(name="get_current_time"),
        SimpleNamespace(name="tavily_search"),
    ]

    tools = agent._get_tools_for_binding(conversation_id="conversation-1")

    tool_names = [tool.name for tool in tools]
    assert "tool_search" in tool_names
    assert "get_current_time" in tool_names
    assert "tavily_search" in tool_names


def test_search_prompt_orders_tool_search_time_then_web_search():
    assert "If the search tool is not loaded yet, use `tool_search`" in SEARCH_SYSTEM_PROMPT
    assert "call `get_current_time`, then call the web search tool" in SEARCH_SYSTEM_PROMPT
    assert "Never make `tavily_search` your first actual web-search call in a turn" in (
        SEARCH_SYSTEM_PROMPT
    )


def test_search_with_results_prompt_keeps_time_before_first_web_search():
    assert "If the only result so far is tool discovery" in SEARCH_WITH_RESULTS_SYSTEM_PROMPT
    assert "call `get_current_time` before your first actual web-search tool" in (
        SEARCH_WITH_RESULTS_SYSTEM_PROMPT
    )
