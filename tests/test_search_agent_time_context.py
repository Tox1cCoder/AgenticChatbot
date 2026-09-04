"""How the search agent learns what "now" means.

It does not ask a tool. The server injects the current date into the prompt and
``normalize_web_search`` anchors every query in Python, so the model never
spends a turn on a time lookup before searching — and a model that skips the
lookup can no longer produce a query stamped with its training year.
"""

from __future__ import annotations

from types import SimpleNamespace

from app.ai.deferred_tool_binding import _get_pinned_specs, build_deferred_tool_list
from app.ai.prompts import SEARCH_SYSTEM_PROMPT, SEARCH_WITH_RESULTS_SYSTEM_PROMPT


class _FakeManager:
    def __init__(self, server_by_tool_name: dict[str, str]):
        self._server_by_tool_name = server_by_tool_name

    def get_server_for_tool(self, tool):
        return self._server_by_tool_name.get(tool.name)


def test_the_search_agent_pins_widgets_and_nothing_else(monkeypatch):
    """Tavily reaches the model through the product tools, and time is no
    longer a required pin: it was only ever pinned to satisfy a prompt rule
    that no longer exists."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "mcp_tool_search_pinned_tools", [], raising=False)

    pinned_specs = _get_pinned_specs("search")

    assert pinned_specs == [
        "widgets::widget_create",
        "widgets::widget_update",
        "widgets::widget_get_state",
    ]


def test_search_agent_does_not_pin_brave_image_search(monkeypatch):
    """Image search reaches Brave via image_search now; the raw pin is gone."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "mcp_tool_search_pinned_tools", [], raising=False)

    assert "brave_image_search::brave_image_search" not in _get_pinned_specs("search")


def test_search_agent_binding_keeps_all_required_pins_under_default_cap(monkeypatch):
    """Regression guard: a pin cap smaller than the required set must not drop
    any of search's required widget pins."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "mcp_tool_search_pinned_tools", [], raising=False)
    monkeypatch.setattr(settings, "mcp_tool_search_max_pinned_tools", 1, raising=False)

    all_tools = [
        SimpleNamespace(name="widget_create"),
        SimpleNamespace(name="widget_update"),
        SimpleNamespace(name="widget_get_state"),
    ]
    manager = _FakeManager(
        {
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

    assert {"widget_create", "widget_update", "widget_get_state"} <= {
        tool.name for tool in tools
    }


def test_search_agent_deferred_binding_excludes_the_raw_providers(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "mcp_tool_search_pinned_tools", [], raising=False)

    manager = _FakeManager({"get_current_time": "time", "tavily_search": "tavily"})

    tools = build_deferred_tool_list(
        conversation_id="conversation-1",
        agent_key="search",
        mcp_manager=manager,
        all_mcp_tools=[
            SimpleNamespace(name="get_current_time"),
            SimpleNamespace(name="tavily_search"),
        ],
    )

    tool_names = [tool.name for tool in tools]
    assert "tool_search" in tool_names
    assert "tavily_search" not in tool_names


def test_search_agent_binding_exposes_the_product_web_tools(monkeypatch):
    from app.ai.agents.search_agent import SearchAgent
    from app.core.config import settings

    monkeypatch.setattr(settings, "mcp_tool_search_pinned_tools", [], raising=False)
    monkeypatch.setattr(
        "app.ai.agents.base_agent.should_use_deferred_loading",
        lambda _agent_key: True,
    )

    agent = SearchAgent()
    agent.mcp_manager = _FakeManager({"tavily_search": "tavily"})
    agent.tools = [SimpleNamespace(name="tavily_search")]

    tool_names = [tool.name for tool in agent._get_tools_for_binding(conversation_id="c-1")]

    assert "tool_search" in tool_names
    assert {"web_search", "web_open", "image_search"} <= set(tool_names)
    assert "tavily_search" not in tool_names


def test_the_search_prompt_no_longer_choreographs_a_time_tool():
    """The old rule made a time lookup mandatory before every web search. It
    cost a turn, and the model still had to remember to apply the answer."""
    assert "get_current_time" not in SEARCH_SYSTEM_PROMPT
    assert "get_current_time" not in SEARCH_WITH_RESULTS_SYSTEM_PROMPT


def test_the_search_prompt_states_the_product_tool_contract():
    prompt = SEARCH_SYSTEM_PROMPT

    assert "Express the user's goal as a concrete objective." in prompt
    assert 'freshness="recent"' in prompt
    assert 'freshness="as_of"' in prompt
    assert "Search first. Open only the few URLs" in prompt
    assert "Every web_open call must include the exact question to extract." in prompt
    assert "do not repeat an unchanged query" in prompt


def test_the_search_prompt_never_names_a_raw_provider():
    for prompt in (SEARCH_SYSTEM_PROMPT, SEARCH_WITH_RESULTS_SYSTEM_PROMPT):
        assert "tavily" not in prompt.lower()
        assert "brave" not in prompt.lower()


def test_search_agent_does_not_pin_any_tavily_tools(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "mcp_tool_search_pinned_tools", [], raising=False)

    pinned_specs = _get_pinned_specs("search")

    assert not any(spec.startswith("tavily::") for spec in pinned_specs)
