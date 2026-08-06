"""Brave Image Search reachability (Task 6: route agents through web_research).

Brave is no longer pinned for any agent — pinning the raw provider tool
would let the model bypass the web_research tool's turn budget and visual
verifier. It stays reachable via ``tool_search`` for any agent, and chat/search
additionally get ``web_research`` bound directly (see
test_web_research_binding.py). An operator can still opt a raw tool into the
pinned set manually via ``mcp_tool_search_pinned_tools``.
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
    included — the pin removal in Task 6 is the intended change."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "mcp_tool_search_pinned_tools", [], raising=False)

    for agent_key in ("chat", "search", "rag", "planning", "canvas", "image_generator"):
        assert _BRAVE_SPEC not in _get_pinned_specs(agent_key)


def test_chat_binding_does_not_auto_expose_brave(monkeypatch):
    """Brave is not auto-bound just because the raw tool is available; it
    stays reachable only through tool_search discovery or web_research."""
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


def test_user_configured_brave_pin_still_binds(monkeypatch):
    """Only the system-required default pin was removed: an operator can still
    opt Brave into the pinned set explicitly via mcp_tool_search_pinned_tools."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "mcp_tool_search_pinned_tools", [_BRAVE_SPEC], raising=False)
    monkeypatch.setattr(settings, "mcp_tool_search_max_pinned_tools", 5, raising=False)

    manager = _FakeManager({"brave_image_search": "brave_image_search"})
    all_tools = [SimpleNamespace(name="brave_image_search")]

    pinned = get_pinned_tools(manager, all_tools, agent_key="chat")

    assert "brave_image_search" in {tool.name for tool in pinned}
