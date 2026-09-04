"""Brave Image Search reachability.

Brave is pinned for no agent, and since the product tools landed it is not
reachable through ordinary discovery either: ``image_search`` owns the turn
budget and the deterministic selection, and a model that could call the raw
tool would bypass both. Chat and search get ``image_search`` bound directly
(see test_web_tool_binding.py). The only way back to the raw tool is the
server-side ``allow_raw_web_tools`` opt-in.
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


def test_no_agent_pins_brave_image_search(monkeypatch):
    """Brave is not a system-required pin for any agent, chat and search
    included."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "mcp_tool_search_pinned_tools", [], raising=False)

    for agent_key in ("chat", "search", "rag", "planning", "canvas", "image_generator"):
        assert _BRAVE_SPEC not in _get_pinned_specs(agent_key)


def test_chat_binding_does_not_auto_expose_brave(monkeypatch):
    """Brave is not auto-bound just because the raw tool is available."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "mcp_tool_search_pinned_tools", [], raising=False)

    manager = _FakeManager({"brave_image_search": "brave_image_search"})
    tools = build_deferred_tool_list(
        conversation_id="conversation-1",
        agent_key="chat",
        mcp_manager=manager,
        all_mcp_tools=[SimpleNamespace(name="brave_image_search")],
    )

    assert "brave_image_search" not in {tool.name for tool in tools}


def test_a_configured_brave_pin_resolves_but_no_longer_binds(monkeypatch):
    """Pin resolution and binding are different questions now.

    ``get_pinned_tools`` still honours the operator's configuration, but
    ``build_deferred_tool_list`` drops the raw provider afterwards. A pin was
    the one path that bound a tool without discovery, so leaving it open would
    have reopened the bypass the denylist exists to close. The authorized way
    back is ``allow_raw_web_tools``, which only server code can pass."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "mcp_tool_search_pinned_tools", [_BRAVE_SPEC], raising=False)
    monkeypatch.setattr(settings, "mcp_tool_search_max_pinned_tools", 5, raising=False)

    manager = _FakeManager({"brave_image_search": "brave_image_search"})
    all_tools = [SimpleNamespace(name="brave_image_search")]

    pinned = get_pinned_tools(manager, all_tools, agent_key="chat")
    assert "brave_image_search" in {tool.name for tool in pinned}

    bound = build_deferred_tool_list(
        conversation_id="conversation-1",
        agent_key="chat",
        mcp_manager=manager,
        all_mcp_tools=all_tools,
    )
    assert "brave_image_search" not in {tool.name for tool in bound}

    authorized = build_deferred_tool_list(
        conversation_id="conversation-1",
        agent_key="chat",
        mcp_manager=manager,
        all_mcp_tools=all_tools,
        allow_raw_web_tools=True,
    )
    assert "brave_image_search" in {tool.name for tool in authorized}
