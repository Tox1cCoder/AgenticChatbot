from __future__ import annotations

from types import SimpleNamespace

from app.ai.deferred_tool_binding import _get_pinned_specs, build_deferred_tool_list
from app.ai.prompts import SEARCH_SYSTEM_PROMPT, SEARCH_WITH_RESULTS_SYSTEM_PROMPT


class _FakeManager:
    def __init__(self, server_by_tool_name: dict[str, str]):
        self._server_by_tool_name = server_by_tool_name

    def get_server_for_tool(self, tool):
        return self._server_by_tool_name.get(tool.name)


def test_search_agent_auto_pins_time_only(monkeypatch):
    """Tavily is no longer a required pin: research now reaches it through the
    server-orchestrated web_research tool (see test_web_research_binding.py)."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "mcp_tool_search_pinned_tools", [], raising=False)

    pinned_specs = _get_pinned_specs("search")

    assert "time::get_current_time" in pinned_specs
    assert "tavily::tavily_search" not in pinned_specs


def test_search_agent_does_not_pin_brave_image_search(monkeypatch):
    """Image search reaches Brave via web_research now; the raw pin is gone."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "mcp_tool_search_pinned_tools", [], raising=False)

    pinned_specs = _get_pinned_specs("search")

    assert "brave_image_search::brave_image_search" not in pinned_specs


def test_search_agent_binding_keeps_all_required_pins_under_default_cap(monkeypatch):
    """Regression guard: a pin cap smaller than the required set must not drop
    any of search's four required pins (time + three widget tools). Tavily and
    Brave are no longer required pins, so the cap no longer needs to cover them."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "mcp_tool_search_pinned_tools", [], raising=False)
    # Cap below the four required pins to prove required pins bypass the cap.
    monkeypatch.setattr(settings, "mcp_tool_search_max_pinned_tools", 2, raising=False)

    all_tools = [
        SimpleNamespace(name="get_current_time"),
        SimpleNamespace(name="widget_create"),
        SimpleNamespace(name="widget_update"),
        SimpleNamespace(name="widget_get_state"),
    ]
    manager = _FakeManager(
        {
            "get_current_time": "time",
            "widget_create": "widgets",
            "widget_update": "widgets",
            "widget_get_state": "widgets",
        }
    )

    tools = build_deferred_tool_list(
        conversation_id="conversation-1",
        agent_key="search",
        mcp_manager=manager,
        all_mcp_tools=all_tools,
    )

    tool_names = {tool.name for tool in tools}
    assert {
        "get_current_time",
        "widget_create",
        "widget_update",
        "widget_get_state",
    } <= tool_names


def test_search_agent_deferred_binding_includes_time_only(monkeypatch):
    """Tavily is no longer auto-pinned at this layer; the agent-level
    web_research binding (see test_web_research_binding.py) is the research path now."""
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
    assert "tavily_search" not in tool_names


def test_search_agent_binding_exposes_time_and_web_research_tools(monkeypatch):
    """Tavily is no longer pinned directly; the search agent reaches it (and
    Brave) through the server-orchestrated web_research tool instead."""
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
    assert "web_research" in tool_names
    assert "tavily_search" not in tool_names


def test_search_prompt_orders_tool_search_time_then_web_search():
    assert "If the search tool is not loaded yet, use `tool_search`" in SEARCH_SYSTEM_PROMPT
    assert "call `get_current_time`, then call that search tool" in SEARCH_SYSTEM_PROMPT
    assert (
        "Do not make a web/news search your first actual web retrieval call in a turn"
        in SEARCH_SYSTEM_PROMPT
    )


def test_search_with_results_prompt_keeps_time_before_first_web_search():
    assert "If the only result so far is tool discovery" in SEARCH_WITH_RESULTS_SYSTEM_PROMPT
    assert "call `get_current_time` before your first actual web-search tool" in (
        SEARCH_WITH_RESULTS_SYSTEM_PROMPT
    )


def test_search_prompt_exempts_image_reference_search_from_time_lookup():
    """The time-before-search rule applies to web/news/Tavily search, not to
    pure visual reference searches like "what does X look like?"."""
    prompt = SEARCH_SYSTEM_PROMPT
    assert "Image reference searches" in prompt
    assert "do not require" in prompt
    # The time rule is still present for actual web/news search.
    assert "call `get_current_time`, then call that search tool" in prompt


def test_search_agent_does_not_pin_any_tavily_tools(monkeypatch):
    """No Tavily tool is pinned directly anymore; web_research is the sole
    path to Tavily (see test_web_research_binding.py)."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "mcp_tool_search_pinned_tools", [], raising=False)

    pinned_specs = _get_pinned_specs("search")

    assert "tavily::tavily_search" not in pinned_specs
    assert "tavily::tavily_extract" not in pinned_specs
    assert "tavily::tavily_map" not in pinned_specs
    assert "tavily::tavily_crawl" not in pinned_specs


def test_search_prompt_allows_extract_map_and_crawl_via_tool_search():
    prompt = SEARCH_SYSTEM_PROMPT

    assert "specific URL" in prompt
    assert "site structure" in prompt
    assert "bounded site" in prompt
    assert "Never make `tavily_search` your first actual web-search call" not in prompt
