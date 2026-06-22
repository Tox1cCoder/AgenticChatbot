"""HITL fires correctly for client (sidecar) tools and deferred (search-loaded) tools."""

from types import SimpleNamespace

import pytest

from app.ai.hitl_config import any_call_requires_approval, resolve_call_identity


class _FakeManager:
    def __init__(self, mapping):
        self._mapping = mapping

    def get_server_for_tool(self, tool):
        return self._mapping.get(id(tool))


def test_client_server_rule_gates_a_sidecar_tool_by_name_alone():
    # Sidecar tool, NOT yet in any tool_map (e.g. resolved purely from the call name).
    policy = {"master_enabled": True, "servers": {"desktop_commander": True},
              "tools": {}, "global_tools": []}
    calls = [{"name": "client__desktop_commander__start_process", "args": {}, "id": "c1"}]
    assert any_call_requires_approval(calls, policy=policy) is True


def test_deferred_server_tool_gated_by_server_after_autoload():
    # A server tool discovered + autoloaded via tool_search this turn: bare name, no
    # metadata, server resolved through the MCP manager (the deferred-binding path).
    loaded_tool = SimpleNamespace(name="run_query", metadata={})
    tool_map = {"run_query": loaded_tool}
    manager = _FakeManager({id(loaded_tool): "postgres"})
    policy = {"master_enabled": True, "servers": {"postgres": True},
              "tools": {}, "global_tools": []}

    identity = resolve_call_identity({"name": "run_query"}, tool_map=tool_map, mcp_manager=manager)
    assert identity.server_name == "postgres"

    calls = [{"name": "run_query", "args": {}, "id": "c1"}]
    assert any_call_requires_approval(calls, policy=policy, tool_map=tool_map, mcp_manager=manager) is True


@pytest.mark.asyncio
async def test_prepare_interrupt_payload_carries_client_provenance():
    from app.ai import graph as graph_module

    wf = graph_module.MultiAgentWorkflow.__new__(graph_module.MultiAgentWorkflow)
    client_tool = SimpleNamespace(
        name="client__excel__delete_sheet",
        metadata={"server_name": "excel", "qualified_tool_id": "excel::delete_sheet",
                  "tool_origin": "client_mcp"},
    )

    # agent=None + explicit tool_map => _prepare_interrupt_payload skips building a real map.
    payload = await wf._prepare_interrupt_payload(
        {"context": {}, "device_id": "dev-1"},
        tool_calls=[{"name": "client__excel__delete_sheet", "args": {}, "id": "c1"}],
        agent=None,
        tool_map={client_tool.name: client_tool},
    )
    prov = payload["metadata"]["tool_provenance"]
    entry = next(iter(prov.values()))
    assert entry["server_name"] == "excel"
    assert entry["qualified_tool_id"] == "excel::delete_sheet"
    assert entry["tool_origin"] == "client_mcp"
