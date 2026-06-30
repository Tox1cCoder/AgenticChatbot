"""Brave Image Search pinning for the chat agent (image_search.md Phase 4).

Brave is pinned only for ``chat`` and ``search``. Other agents
(rag/planning/canvas/image_generator) must not pin it by default — they can
still discover it via ``tool_search``. Required agent pins must survive the
user-configurable pin cap.
"""

from __future__ import annotations

from types import SimpleNamespace

from app.ai.deferred_tool_binding import (
    _get_pinned_specs,
    build_deferred_tool_list,
    get_pinned_tools,
)

_BRAVE_SPEC = "brave_image_search::brave_image_search"


class _FakeManager:
    def __init__(self, server_by_tool_name: dict[str, str]):
        self._server_by_tool_name = server_by_tool_name

    def get_server_for_tool(self, tool):
        return self._server_by_tool_name.get(tool.name)


def test_chat_agent_pins_brave_image_search(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "mcp_tool_search_pinned_tools", [], raising=False)

    assert _BRAVE_SPEC in _get_pinned_specs("chat")


def test_chat_binding_exposes_brave_when_tool_available(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "mcp_tool_search_pinned_tools", [], raising=False)

    manager = _FakeManager({"brave_image_search": "brave_image_search"})
    tools = build_deferred_tool_list(
        conversation_id="conversation-1",
        agent_key="chat",
        mcp_manager=manager,
        all_mcp_tools=[SimpleNamespace(name="brave_image_search")],
    )

    assert "brave_image_search" in {tool.name for tool in tools}


def test_non_chat_search_agents_do_not_pin_brave(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "mcp_tool_search_pinned_tools", [], raising=False)

    for agent_key in ("rag", "planning", "canvas", "image_generator"):
        assert _BRAVE_SPEC not in _get_pinned_specs(agent_key)


def test_user_pins_capped_without_dropping_required_brave(monkeypatch):
    """A full user-configured pin list (capped at max_pinned) must not evict the
    agent-required Brave pin."""
    from app.core.config import settings

    user_pins = [f"extra::tool_{i}" for i in range(5)]
    monkeypatch.setattr(settings, "mcp_tool_search_pinned_tools", user_pins, raising=False)
    monkeypatch.setattr(settings, "mcp_tool_search_max_pinned_tools", 5, raising=False)

    server_map = {f"tool_{i}": "extra" for i in range(5)}
    server_map["brave_image_search"] = "brave_image_search"
    manager = _FakeManager(server_map)
    all_tools = [SimpleNamespace(name=f"tool_{i}") for i in range(5)]
    all_tools.append(SimpleNamespace(name="brave_image_search"))

    pinned = get_pinned_tools(manager, all_tools, agent_key="chat")

    assert "brave_image_search" in {tool.name for tool in pinned}
