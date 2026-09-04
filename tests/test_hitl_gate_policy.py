"""The graph gate honors a per-turn per-user policy and resolves server provenance."""

from types import SimpleNamespace

from app.ai import graph as graph_module
from app.ai.hitl_config import calls_requiring_approval
from app.ai.workflow import tool_loop as tool_loop_module


class _FakeManager:
    def __init__(self, mapping):
        self._mapping = mapping

    def get_server_for_tool(self, tool):
        return self._mapping.get(id(tool))


def _workflow_stub(tool_map, manager, monkeypatch):
    """A bare object exposing just what _needs_approval / _should_call_tools touch."""
    wf = graph_module.MultiAgentWorkflow.__new__(graph_module.MultiAgentWorkflow)
    agent = object()
    wf.agents = {"chat_agent": agent}

    async def _fake_tool_map(*_a, **_k):
        return tool_map

    async def _fake_manager():
        return manager

    # ``_run_agent_in_isolated_context`` still lives in ``app.ai.graph``; the
    # tool/HITL helpers (``_needs_approval`` etc.) were relocated to
    # ``app.ai.workflow.tool_loop`` (Task 8), so patch the names where each
    # consumer now resolves them.
    monkeypatch.setattr(graph_module, "ensure_agent_tool_map", _fake_tool_map)
    monkeypatch.setattr(tool_loop_module, "ensure_agent_tool_map", _fake_tool_map)
    monkeypatch.setattr(tool_loop_module, "get_global_mcp_manager", _fake_manager)
    return wf


def _policy(*, master_enabled=True, global_tools=(), client_mcp=None):
    return {
        "master_enabled": master_enabled,
        "client_rules": {
            "client_mcp": client_mcp or {"servers": {}, "tools": {}},
            "client_skill": {"servers": {}, "tools": {}},
        },
        "global_tools": list(global_tools),
    }


def test_gate_gates_all_tools_from_a_server_via_policy():
    """A standard specialist gates on the same policy every other caller uses."""
    server_tool = SimpleNamespace(name="search", metadata={})
    manager = _FakeManager({id(server_tool): "tavily"})
    calls = [{"name": "search", "args": {}, "id": "call-1"}]

    gated = calls_requiring_approval(
        calls,
        policy=_policy(global_tools=["search"]),
        tool_map={"search": server_tool},
        mcp_manager=manager,
    )

    assert gated == {"call-1"}


def test_gate_lets_tool_override_exempt_a_server_tool():
    tool = SimpleNamespace(
        name="client__desktop_commander__list_files",
        metadata={
            "server_name": "desktop_commander",
            "qualified_tool_id": "desktop_commander::list_files",
            "tool_origin": "client_mcp",
        },
    )
    calls = [{"name": tool.name, "args": {}, "id": "call-1"}]

    gated = calls_requiring_approval(
        calls,
        policy=_policy(
            client_mcp={
                "servers": {"desktop_commander": True},
                "tools": {"desktop_commander::list_files": False},
            }
        ),
        tool_map={tool.name: tool},
        mcp_manager=_FakeManager({}),
    )

    assert gated == set()


def test_gate_master_off_never_gates():
    calls = [{"name": "search", "args": {}, "id": "x"}]

    gated = calls_requiring_approval(
        calls,
        policy=_policy(master_enabled=False, global_tools=["search"]),
        tool_map={},
        mcp_manager=_FakeManager({}),
    )

    assert gated == set()


