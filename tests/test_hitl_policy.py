"""Per-user HITL policy resolution and precedence (tool-overrides-server)."""

from types import SimpleNamespace

from app.ai.hitl_config import (
    any_call_requires_approval,
    build_global_policy,
    identity_requires_approval,
    policy_from_context,
    resolve_call_identity,
)


def _tool(name, **metadata):
    return SimpleNamespace(name=name, metadata=dict(metadata))


class _FakeManager:
    def __init__(self, mapping):
        self._mapping = mapping  # id(tool) -> server

    def get_server_for_tool(self, tool):
        return self._mapping.get(id(tool))


def _policy(master=True, servers=None, tools=None, global_tools=None):
    return {
        "master_enabled": master,
        "servers": servers or {},
        "tools": tools or {},
        "global_tools": global_tools or [],
    }


def test_resolve_identity_for_client_tool_uses_name_and_metadata():
    tool = _tool(
        "client__desktop_commander__start_process",
        server_name="desktop_commander",
        qualified_tool_id="desktop_commander::start_process",
        tool_origin="client_mcp",
    )
    identity = resolve_call_identity(
        {"name": "client__desktop_commander__start_process"},
        tool_map={"client__desktop_commander__start_process": tool},
    )
    assert identity.server_name == "desktop_commander"
    assert identity.qualified_tool_id == "desktop_commander::start_process"
    assert identity.origin == "client_mcp"


def test_resolve_identity_for_client_tool_parses_name_without_metadata():
    identity = resolve_call_identity({"name": "client__excel__write_cell"})
    assert identity.server_name == "excel"
    assert identity.qualified_tool_id == "excel::write_cell"
    assert identity.origin == "client_mcp"


def test_resolve_identity_for_server_tool_uses_manager():
    tool = _tool("search")  # bare server-tool name, no metadata
    manager = _FakeManager({id(tool): "tavily"})
    identity = resolve_call_identity(
        {"name": "search"}, tool_map={"search": tool}, mcp_manager=manager
    )
    assert identity.server_name == "tavily"
    assert identity.qualified_tool_id == "tavily::search"
    assert identity.origin == "server_mcp"


def test_resolve_identity_for_server_tool_prefers_metadata():
    tool = _tool(
        "calculate",
        server_name="calculator",
        qualified_tool_id="calculator::calculate",
        tool_origin="server_mcp",
    )
    identity = resolve_call_identity({"name": "calculate"}, tool_map={"calculate": tool})
    assert identity.server_name == "calculator"
    assert identity.qualified_tool_id == "calculator::calculate"
    assert identity.origin == "server_mcp"


def test_resolve_identity_for_aliased_deferred_server_tool_without_manager_id_hit():
    tool = _tool(
        "brave__search",
        tool_origin="server_mcp",
        aliased_from_tool_name="search",
        call_name="brave__search",
    )
    identity = resolve_call_identity({"name": "brave__search"}, tool_map={"brave__search": tool})
    assert identity.server_name == "brave"
    assert identity.qualified_tool_id == "brave::search"
    assert identity.origin == "server_mcp"


def test_precedence_tool_qualified_overrides_server():
    policy = _policy(
        servers={"desktop_commander": True},
        tools={"desktop_commander::list_files": False},
    )
    gated = SimpleNamespace(
        name="client__desktop_commander__run",
        server_name="desktop_commander",
        qualified_tool_id="desktop_commander::run",
        origin="client_mcp",
    )
    exempt = SimpleNamespace(
        name="client__desktop_commander__list_files",
        server_name="desktop_commander",
        qualified_tool_id="desktop_commander::list_files",
        origin="client_mcp",
    )
    assert identity_requires_approval(gated, policy) is True  # inherits server ON
    assert identity_requires_approval(exempt, policy) is False  # tool override SKIP


def test_precedence_tool_can_force_on_when_server_off():
    policy = _policy(servers={"excel": False}, tools={"excel::delete_sheet": True})
    ident = SimpleNamespace(
        name="client__excel__delete_sheet",
        server_name="excel",
        qualified_tool_id="excel::delete_sheet",
        origin="client_mcp",
    )
    assert identity_requires_approval(ident, policy) is True


def test_precedence_master_off_disables_everything():
    policy = _policy(master=False, servers={"excel": True})
    ident = SimpleNamespace(
        name="client__excel__x",
        server_name="excel",
        qualified_tool_id="excel::x",
        origin="client_mcp",
    )
    assert identity_requires_approval(ident, policy) is False


def test_precedence_legacy_global_floor_still_gates():
    policy = _policy(global_tools=["dangerous_tool"])
    ident = SimpleNamespace(
        name="dangerous_tool",
        server_name=None,
        qualified_tool_id=None,
        origin="internal",
    )
    assert identity_requires_approval(ident, policy) is True


def test_any_call_requires_approval_short_circuits_on_first_gated():
    policy = _policy(servers={"desktop_commander": True})
    calls = [
        {"name": "client__time_server__now"},
        {"name": "client__desktop_commander__run"},
    ]
    assert any_call_requires_approval(calls, policy=policy) is True


def test_policy_from_context_falls_back_to_global(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "enable_human_in_the_loop", True)
    monkeypatch.setattr(settings, "hitl_tools_require_approval", ["legacy_tool"])
    policy = policy_from_context(None)
    assert policy == build_global_policy()
    assert policy["global_tools"] == ["legacy_tool"]
    assert policy["master_enabled"] is True
