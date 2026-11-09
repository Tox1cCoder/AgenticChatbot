"""Model-visible discovery stays compact and distinguishes misses from outages."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.ai.client_tool_catalog import ClientToolDescriptor
from app.ai.deferred_tool_state import DeferredToolState
from app.ai.mcp_tool_catalog import McpToolCatalog, ToolDescriptor
from app.ai.tool_context import ToolContext
from app.ai.tool_search_scoring import rank_tool_candidates
from app.ai.tool_search_tool import (
    ToolSearchOutput,
    _execute_tool_search,
    create_tool_search_tool,
    tool_search,
)


@pytest.fixture
def search_runtime(monkeypatch):
    catalog = McpToolCatalog(mcp_manager=object())
    state = DeferredToolState()

    async def manager():
        return object()

    async def get_catalog(_manager):
        return catalog

    monkeypatch.setattr("app.ai.tool_search_tool.get_global_mcp_manager", manager)
    monkeypatch.setattr("app.ai.tool_search_tool.get_tool_catalog", get_catalog)
    monkeypatch.setattr("app.ai.tool_search_tool.get_deferred_tool_state", lambda: state)
    monkeypatch.setattr(
        "app.ai.tool_search_tool.get_tool_context",
        lambda: ToolContext(conversation_id="contract-conversation", agent_key="chat"),
    )
    monkeypatch.setattr("app.ai.tool_search_tool.settings.mcp_tool_search_debug_scores", False)
    return catalog


def _populate(catalog, count=1):
    catalog._tools = [
        ToolDescriptor(
            tool_name="read_file" if index == 0 else f"read_file_variant_{index}",
            server_name="files",
            description="Read a local text file.",
            arg_names=["path"],
            required_arg_names=["path"],
            schema_fingerprint=f"fp-{index}",
        )
        for index in range(count)
    ]
    catalog._tools_by_server = {"files": catalog._tools}
    catalog._server_name_lower_map = {"files": "files"}


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", [lambda: tool_search, create_tool_search_tool])
async def test_empty_discovery_has_no_redundant_context(search_runtime, factory):
    query = (
        "chấm công hệ thống timesheet attendance tạo đơn overtime OT "
        "nghỉ phép xem giờ vào ra hôm nay"
    )
    payload = await factory().ainvoke({"query": query})

    assert payload == '{"results":[],"next_action":"refine_search"}'
    assert len(payload) == 44
    assert ToolSearchOutput.model_validate_json(payload).results == []


@pytest.mark.asyncio
async def test_loaded_discovery_is_actionable_without_duplicate_recommendation(search_runtime):
    _populate(search_runtime)

    result = await _execute_tool_search(query="read_file")

    assert set(result) == {"results", "next_action"}
    assert result["next_action"] == "call_a_loaded_tool"
    assert result["results"] == [
        {
            "tool_name": "read_file",
            "description": "Read a local text file.",
            "arg_hints": "(path*)",
            "is_loaded": True,
        }
    ]


@pytest.mark.asyncio
async def test_inventory_omits_empty_discovery_metadata(search_runtime):
    _populate(search_runtime)

    result = await _execute_tool_search()

    assert set(result) == {"mode", "inventory", "next_action"}
    assert result["mode"] == "inventory"
    assert result["inventory"][0]["server_name"] == "files"
    assert result["next_action"] == "inspect_inventory"


@pytest.mark.asyncio
async def test_pagination_is_only_reported_when_results_are_truncated(search_runtime):
    _populate(search_runtime, count=3)

    truncated = await _execute_tool_search(query="read file", top_k=1)
    complete = await _execute_tool_search(query="read file", top_k=3)

    assert truncated["more_available"] is True
    assert len(truncated["results"]) == 1
    assert "more_available" not in complete
    assert len(complete["results"]) == 3


@pytest.mark.asyncio
async def test_debug_setting_retains_ranking_diagnostics(search_runtime, monkeypatch):
    _populate(search_runtime)
    monkeypatch.setattr("app.ai.tool_search_tool.settings.mcp_tool_search_debug_scores", True)

    result = await _execute_tool_search(query="read_file")

    assert result["debug_scores"][0]["confidence"] == "high"
    assert result["debug_scores"][0]["score"] > 0
    assert result["debug_scores"][0]["reasons"]
    assert "confidence" not in result["results"][0]
    assert "match_reasons" not in result["results"][0]


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["read_file", None])
async def test_server_catalog_failure_is_not_a_search_miss(search_runtime, monkeypatch, query):
    async def unavailable(_manager):
        raise RuntimeError("private connection details must stay in logs")

    monkeypatch.setattr("app.ai.tool_search_tool.get_tool_catalog", unavailable)

    result = await _execute_tool_search(query=query)

    assert result["error"] == "catalog_unavailable"
    assert result["unavailable_sources"] == ["server"]
    assert result["next_action"] == "report_unavailable"
    assert "connection" in result["guidance"].lower()
    assert "private connection details" not in json.dumps(result)


@pytest.mark.asyncio
async def test_client_only_catalog_failure_has_recovery_guidance(search_runtime, monkeypatch):
    _populate(search_runtime)
    monkeypatch.setattr(
        "app.ai.tool_search_tool.get_tool_context",
        lambda: ToolContext(device_id="device-1", user_id="user-1", tool_scope="client_only"),
    )

    def unavailable(*_args):
        raise RuntimeError("client catalog unavailable")

    monkeypatch.setattr("app.ai.tool_search_tool.get_client_tool_catalog", unavailable)

    result = await _execute_tool_search(query="read file")

    assert result["error"] == "catalog_unavailable"
    assert result["unavailable_sources"] == ["client"]
    assert result["next_action"] == "report_unavailable"


@pytest.mark.asyncio
async def test_healthy_server_results_survive_client_failure(search_runtime, monkeypatch):
    _populate(search_runtime)
    monkeypatch.setattr(
        "app.ai.tool_search_tool.get_tool_context",
        lambda: ToolContext(
            conversation_id="contract-conversation",
            agent_key="chat",
            device_id="device-1",
            user_id="user-1",
            tool_scope="default",
        ),
    )

    def unavailable(*_args):
        raise RuntimeError("client catalog unavailable")

    monkeypatch.setattr("app.ai.tool_search_tool.get_client_tool_catalog", unavailable)

    result = await _execute_tool_search(query="read_file")

    assert result["results"][0]["is_loaded"] is True
    assert result["next_action"] == "call_a_loaded_tool"
    assert result["unavailable_sources"] == ["client"]
    assert "error" not in result


@pytest.mark.asyncio
async def test_healthy_client_results_survive_server_failure(search_runtime, monkeypatch):
    descriptor = ClientToolDescriptor(
        tool_name="client__files__read_file",
        server_name="files",
        description="Read a local text file.",
        arg_names=["path"],
        required_arg_names=["path"],
        device_id="device-1",
        session_id="session-1",
        qualified_tool_id="files::read_file",
        origin="client_mcp",
    )

    async def unavailable(_manager):
        raise RuntimeError("server catalog unavailable")

    monkeypatch.setattr("app.ai.tool_search_tool.get_tool_catalog", unavailable)
    monkeypatch.setattr(
        "app.ai.tool_search_tool.get_tool_context",
        lambda: ToolContext(
            conversation_id="contract-conversation",
            agent_key="chat",
            device_id="device-1",
            user_id="user-1",
            tool_scope="default",
        ),
    )
    monkeypatch.setattr(
        "app.ai.tool_search_tool.get_client_tool_catalog",
        lambda *_args: SimpleNamespace(
            tool_count=1,
            session_id="session-1",
            search_scored=lambda **kwargs: rank_tool_candidates(
                query=kwargs["query"],
                candidates=[descriptor],
            ),
        ),
    )

    result = await _execute_tool_search(query="read file")

    assert result["results"][0]["tool_name"] == "client__files__read_file"
    assert result["unavailable_sources"] == ["server"]
    assert "error" not in result


@pytest.mark.asyncio
async def test_client_only_empty_inventory_is_an_inventory(search_runtime, monkeypatch):
    monkeypatch.setattr(
        "app.ai.tool_search_tool.get_tool_context",
        lambda: ToolContext(device_id="device-1", user_id="user-1", tool_scope="client_only"),
    )
    monkeypatch.setattr(
        "app.ai.tool_search_tool.get_client_tool_catalog",
        lambda *_args: SimpleNamespace(tool_count=0, get_server_inventory=lambda **_kwargs: []),
    )

    result = await _execute_tool_search()

    assert result == {"mode": "inventory", "inventory": [], "next_action": "inspect_inventory"}


@pytest.mark.asyncio
async def test_inventory_can_fall_back_to_healthy_client_catalog(search_runtime, monkeypatch):
    async def unavailable(_manager):
        raise RuntimeError("server catalog unavailable")

    monkeypatch.setattr("app.ai.tool_search_tool.get_tool_catalog", unavailable)
    monkeypatch.setattr(
        "app.ai.tool_search_tool.get_tool_context",
        lambda: ToolContext(device_id="device-1", user_id="user-1"),
    )
    monkeypatch.setattr(
        "app.ai.tool_search_tool.get_client_tool_catalog",
        lambda *_args: SimpleNamespace(
            get_server_inventory=lambda **_kwargs: [
                {"server_name": "files", "description": "Local files", "tool_count": 1}
            ],
        ),
    )

    result = await _execute_tool_search()

    assert result["inventory"][0]["server_name"] == "files"
    assert result["next_action"] == "inspect_inventory"
    assert result["unavailable_sources"] == ["server"]
    assert "error" not in result
    ToolSearchOutput.model_validate(result)


@pytest.mark.asyncio
async def test_both_catalog_failures_are_reported(search_runtime, monkeypatch):
    async def unavailable_server(_manager):
        raise RuntimeError("server unavailable")

    def unavailable_client(*_args):
        raise RuntimeError("client unavailable")

    monkeypatch.setattr("app.ai.tool_search_tool.get_tool_catalog", unavailable_server)
    monkeypatch.setattr("app.ai.tool_search_tool.get_client_tool_catalog", unavailable_client)
    monkeypatch.setattr(
        "app.ai.tool_search_tool.get_tool_context",
        lambda: ToolContext(device_id="device-1", user_id="user-1"),
    )

    result = await _execute_tool_search(query="read file")

    assert result["error"] == "catalog_unavailable"
    assert result["unavailable_sources"] == ["server", "client"]
    assert result["next_action"] == "report_unavailable"
    ToolSearchOutput.model_validate(result)


@pytest.mark.asyncio
async def test_failure_guidance_survives_the_tool_execution_pipeline(search_runtime, monkeypatch):
    from app.ai.tool_execution import execute_tool_calls

    async def unavailable(_manager):
        raise RuntimeError("private connection details")

    monkeypatch.setattr("app.ai.tool_search_tool.get_tool_catalog", unavailable)
    # The execution map refresh is independent of returning the model-visible result.
    monkeypatch.setattr("app.ai.tool_search_tool.settings.mcp_tool_search_enabled", False)

    outputs, _artifacts, _images = await execute_tool_calls(
        tool_calls=[{"id": "search-call", "name": "tool_search", "args": {"query": "read file"}}],
        tool_map={"tool_search": tool_search},
    )

    result = json.loads(outputs[0]["content"])
    assert result["error"] == "catalog_unavailable"
    assert result["next_action"] == "report_unavailable"
    assert "connection" in result["guidance"]
    assert "private connection details" not in outputs[0]["content"]


@pytest.mark.asyncio
async def test_real_catalog_refresh_failure_is_not_cached_as_an_empty_search(
    search_runtime, monkeypatch
):
    from app.ai.mcp_tool_catalog import get_tool_catalog

    class UnavailableManager:
        async def get_all_tools_info(self):
            raise RuntimeError("private MCP transport error")

        def get_servers_status(self):
            return {}

    catalog = McpToolCatalog(UnavailableManager())
    monkeypatch.setattr("app.ai.mcp_tool_catalog._catalog_instance", catalog)
    monkeypatch.setattr("app.ai.tool_search_tool.get_tool_catalog", get_tool_catalog)

    discovery = await _execute_tool_search(query="read file")
    inventory = await _execute_tool_search()

    for result in [discovery, inventory]:
        assert result["error"] == "catalog_unavailable"
        assert result["unavailable_sources"] == ["server"]
        assert result["next_action"] == "report_unavailable"
        assert "private MCP transport error" not in json.dumps(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("healthy_server", [False, True])
async def test_real_manager_transport_failure_is_reported_and_can_recover(
    search_runtime,
    monkeypatch,
    healthy_server,
):
    from app.ai.mcp_integration import MCPManager
    from app.ai.mcp_tool_catalog import get_tool_catalog

    class OfflineClient:
        def session(self, _server_name):
            raise ConnectionError("private unavailable-server URL")

    manager = MCPManager()
    manager.config = {"servers": {"timesheet": {"enabled": True, "transport": "stdio"}}}
    manager.client = OfflineClient()
    if healthy_server:
        manager.config["servers"]["files"] = {"enabled": True, "transport": "stdio"}
        manager._index_server_tools(
            "files",
            [
                SimpleNamespace(
                    name="read_file", description="Read a local text file.", args_schema={}
                )
            ],
        )

    async def get_manager():
        return manager

    monkeypatch.setattr("app.ai.mcp_tool_catalog._catalog_instance", None)
    monkeypatch.setattr("app.ai.tool_search_tool.get_global_mcp_manager", get_manager)
    monkeypatch.setattr("app.ai.tool_search_tool.get_tool_catalog", get_tool_catalog)

    result = await _execute_tool_search(query="read_file")

    assert result["unavailable_sources"] == ["server"]
    assert "private unavailable-server URL" not in json.dumps(result)
    if healthy_server:
        assert result["results"][0]["is_loaded"] is True
        assert result["next_action"] == "call_a_loaded_tool"
        assert "error" not in result
    else:
        assert result["error"] == "catalog_unavailable"
        assert result["next_action"] == "report_unavailable"

    # A successful empty tool list is healthy, and recovery needs no generation bump.
    manager._index_server_tools("timesheet", [])
    recovered = await _execute_tool_search(query="read_file")
    assert "unavailable_sources" not in recovered
    assert "error" not in recovered


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["read file", None])
async def test_disallowed_client_catalog_cannot_turn_a_miss_into_an_outage(
    search_runtime,
    monkeypatch,
    query,
):
    async def unavailable_server(_manager):
        raise RuntimeError("server catalog unavailable")

    def unavailable_client(*_args):
        raise RuntimeError("disallowed client catalog must not be inspected")

    monkeypatch.setattr("app.ai.tool_search_tool.get_client_tool_catalog", unavailable_client)
    monkeypatch.setattr("app.ai.tool_search_tool.get_tool_catalog", unavailable_server)
    monkeypatch.setattr(
        "app.ai.tool_search_tool.get_tool_context",
        lambda: ToolContext(device_id="device-1", user_id="user-1"),
    )

    # Neither origin is authorized, so neither outage is relevant to this request.
    result = await _execute_tool_search(query=query, server_allowlist=[], client_allowlist=[])

    assert "unavailable_sources" not in result
    assert "error" not in result
    assert result["next_action"] == ("refine_search" if query else "inspect_inventory")


@pytest.mark.asyncio
async def test_disconnected_client_is_distinct_from_a_connected_empty_catalog(
    search_runtime,
    monkeypatch,
):
    from app.ai.client_tool_catalog import ClientToolCatalog

    catalog = ClientToolCatalog("device-1", "user-1")
    monkeypatch.setattr("app.ai.tool_search_tool.get_client_tool_catalog", lambda *_args: catalog)
    monkeypatch.setattr(
        "app.ai.tool_search_tool.get_tool_context",
        lambda: ToolContext(device_id="device-1", user_id="user-1", tool_scope="client_only"),
    )

    disconnected = await _execute_tool_search(query="read file")
    assert disconnected["error"] == "catalog_unavailable"
    assert disconnected["unavailable_sources"] == ["client"]
    assert disconnected["next_action"] == "report_unavailable"

    catalog._session_id = "active-session"
    connected = await _execute_tool_search(query="read file")
    assert connected == {"results": [], "next_action": "refine_search"}


@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs", [{"query": "timesheet"}, {"server_name": "timesheet"}])
async def test_exact_offline_integration_cannot_fuzzy_resolve_to_a_healthy_neighbor(
    search_runtime,
    kwargs,
):
    async def tools_info():
        return [
            {
                "name": "read_docs",
                "server_name": "timesheet-docs",
                "description": "Read timesheet documentation.",
                "args_schema": {},
            }
        ]

    search_runtime._mcp_manager = SimpleNamespace(
        get_all_tools_info=tools_info,
        get_servers_status=lambda: {"timesheet": {"description": "Employee attendance"}},
        get_unavailable_servers=lambda: {"timesheet"},
    )
    await search_runtime.refresh_if_needed()

    result = await _execute_tool_search(**kwargs)

    assert result["resolved_server_name"] == "timesheet"
    assert result["results"] == []
    assert result["error"] == "catalog_unavailable"
    assert result["next_action"] == "report_unavailable"
