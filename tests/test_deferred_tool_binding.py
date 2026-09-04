"""The deferred binding layer's own policy: what it pins and what it hides.

Agent-level binding lives in test_web_tool_binding.py; this file owns the
two decisions this module makes on every agent's behalf — the required pins and
the raw-provider denylist merged into ordinary discovery.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.ai.deferred_tool_binding import (
    RAW_WEB_TOOL_NAMES,
    _get_required_pinned_specs,
    build_deferred_tool_list,
)


class _FakeManager:
    def __init__(self, server_by_tool_name: dict[str, str]):
        self._server_by_tool_name = server_by_tool_name

    def get_server_for_tool(self, tool):
        return self._server_by_tool_name.get(tool.name)


def _tools(*names: str) -> list[SimpleNamespace]:
    return [SimpleNamespace(name=name) for name in names]


def _manager(**mapping: str) -> _FakeManager:
    return _FakeManager(dict(mapping))


def test_the_denylist_names_exactly_the_three_replaced_providers():
    assert set(RAW_WEB_TOOL_NAMES) == {"tavily_search", "tavily_extract", "brave_image_search"}


@pytest.mark.parametrize("raw_name", sorted(RAW_WEB_TOOL_NAMES))
def test_a_pinned_raw_web_tool_is_still_dropped(raw_name, monkeypatch):
    """The denylist has to outrank configuration. A pin is the one path that
    binds a tool without discovery, so a stale pin would reopen the bypass."""
    from app.core.config import settings

    monkeypatch.setattr(
        settings, "mcp_tool_search_pinned_tools", [f"tavily::{raw_name}"], raising=False
    )

    tools = build_deferred_tool_list(
        conversation_id="conversation-1",
        agent_key="search",
        mcp_manager=_manager(**{raw_name: "tavily"}),
        all_mcp_tools=_tools(raw_name),
    )

    assert raw_name not in {tool.name for tool in tools}


def test_an_internal_tool_named_like_a_raw_provider_is_dropped_too(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "mcp_tool_search_pinned_tools", [], raising=False)

    tools = build_deferred_tool_list(
        conversation_id="conversation-1",
        agent_key="chat",
        mcp_manager=None,
        all_mcp_tools=[],
        internal_tools=_tools("tavily_search", "web_search"),
    )

    names = {tool.name for tool in tools}
    assert "tavily_search" not in names
    assert "web_search" in names


def test_the_bound_tool_search_carries_the_denylist(monkeypatch):
    """Hiding a tool from binding but not from discovery would let the model
    autoload it back in the same turn."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "mcp_tool_search_pinned_tools", [], raising=False)
    captured: dict = {}

    def _capture(allowlist=None, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(name="tool_search")

    monkeypatch.setattr("app.ai.deferred_tool_binding.create_tool_search_tool", _capture)

    build_deferred_tool_list(
        conversation_id="conversation-1",
        agent_key="chat",
        mcp_manager=None,
        all_mcp_tools=[],
    )

    assert set(captured["excluded_tool_names"]) >= set(RAW_WEB_TOOL_NAMES)


def test_a_caller_supplied_exclusion_is_merged_not_replaced(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "mcp_tool_search_pinned_tools", [], raising=False)

    tools = build_deferred_tool_list(
        conversation_id="conversation-1",
        agent_key="chat",
        mcp_manager=None,
        all_mcp_tools=[],
        internal_tools=_tools("tavily_search", "widget_create", "web_search"),
        excluded_tool_names={"widget_create"},
    )

    assert {tool.name for tool in tools} == {"web_search", "tool_search"}


def test_an_authorized_diagnostic_caller_can_bind_the_raw_tools(monkeypatch):
    """Provider debugging needs the real tool. The opt-in is an explicit
    keyword only server code can pass — never a request field."""
    from app.core.config import settings

    monkeypatch.setattr(
        settings, "mcp_tool_search_pinned_tools", ["tavily::tavily_search"], raising=False
    )

    tools = build_deferred_tool_list(
        conversation_id="conversation-1",
        agent_key="search",
        mcp_manager=_manager(tavily_search="tavily"),
        all_mcp_tools=_tools("tavily_search"),
        allow_raw_web_tools=True,
    )

    assert "tavily_search" in {tool.name for tool in tools}


def test_the_search_agent_no_longer_requires_a_time_pin():
    """The server injects the current date into the prompt and the Python
    normalizer anchors every search, so a time tool round trip before searching
    buys nothing and costs a turn."""
    assert "time::get_current_time" not in _get_required_pinned_specs("search")


def test_widget_pins_survive_the_change():
    specs = _get_required_pinned_specs("search")

    assert "widgets::widget_create" in specs
    assert "widgets::widget_update" in specs
    assert "widgets::widget_get_state" in specs
