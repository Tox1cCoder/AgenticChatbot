"""HITL fires correctly for client (sidecar) tools and deferred (search-loaded) tools."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

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
    assert (
        any_call_requires_approval(calls, policy=policy, tool_map=tool_map, mcp_manager=manager)
        is True
    )


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


@pytest.mark.asyncio
async def test_prepare_interrupt_payload_redacts_sensitive_args_in_prompt():
    # The approval prompt (action_requests) must not surface a sensitive-keyed
    # argument value, but must keep normal args visible for the approver — and
    # must NOT mutate the original tool call that executes on approval.
    from app.ai import graph as graph_module

    wf = graph_module.MultiAgentWorkflow.__new__(graph_module.MultiAgentWorkflow)
    tool_call = {
        "name": "client__skill_demo__mutate",
        "args": {"calendar_id": "primary", "api_token": "SUPER-SECRET"},
        "id": "c1",
    }
    original_args = tool_call["args"]

    payload = await wf._prepare_interrupt_payload(
        {"context": {}, "device_id": "dev-1"},
        tool_calls=[tool_call],
        agent=None,
        tool_map={},
    )

    prompt_args = payload["action_requests"][0]["args"]
    assert prompt_args["calendar_id"] == "primary"
    assert prompt_args["api_token"] == "<redacted>"
    # The real tool call is untouched, so execution on approval uses real args.
    assert original_args["api_token"] == "SUPER-SECRET"


@pytest.mark.asyncio
async def test_approval_helpers_rebuild_the_live_scoped_handoff_map(monkeypatch):
    """HITL must inspect the same graph-scoped handoff tool that will execute."""
    from app.ai import graph as graph_module
    from app.ai.hand_off_tool import create_hand_off_tool

    wf = graph_module.MultiAgentWorkflow.__new__(graph_module.MultiAgentWorkflow)
    handoff_tool = create_hand_off_tool(["search_agent"])
    observed_internal_tools: list[list[object] | None] = []

    async def fake_ensure_agent_tool_map(agent, **kwargs):
        observed_internal_tools.append(kwargs.get("internal_tools"))
        return {"hand_off": handoff_tool}

    monkeypatch.setattr(
        "app.ai.workflow.tool_loop.ensure_agent_tool_map", fake_ensure_agent_tool_map
    )
    monkeypatch.setattr(
        "app.ai.workflow.tool_loop.get_global_mcp_manager", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        "app.ai.workflow.tool_loop.any_call_requires_approval", lambda *args, **kwargs: False
    )
    state = {"context": {}, "device_id": "device-1"}
    calls = [{"name": "hand_off", "args": {"target_agent": "search_agent"}, "id": "h1"}]

    assert await wf._needs_approval(
        state,
        calls,
        agent=object(),
        internal_tools=[handoff_tool],
    ) is False
    await wf._prepare_interrupt_payload(
        state,
        tool_calls=calls,
        agent=object(),
        internal_tools=[handoff_tool],
    )

    assert observed_internal_tools == [[handoff_tool], [handoff_tool]]
