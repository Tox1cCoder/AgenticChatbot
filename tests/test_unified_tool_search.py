import pytest

from app.ai.client_tool_catalog import ClientToolDescriptor
from app.ai.mcp_tool_catalog import ToolDescriptor
from app.ai.tool_context import ToolContext
from app.ai.tool_search_tool import (
    _execute_tool_search,
    _merge_search_results,
)


@pytest.mark.asyncio
async def test_execute_tool_search_passes_query_through_unchanged(monkeypatch):
    class FakeCatalog:
        def __init__(self):
            self.query = None

        def search(self, query=None, top_k=5, server_name=None, allowlist=None):
            self.query = query
            return []

        def is_ambiguous(self, tool_name):
            return False

    fake_catalog = FakeCatalog()

    async def fake_get_global_mcp_manager():
        return object()

    async def fake_get_tool_catalog(_manager):
        return fake_catalog

    monkeypatch.setattr(
        "app.ai.tool_search_tool.get_global_mcp_manager",
        fake_get_global_mcp_manager,
    )
    monkeypatch.setattr(
        "app.ai.tool_search_tool.get_tool_catalog",
        fake_get_tool_catalog,
    )
    monkeypatch.setattr(
        "app.ai.tool_search_tool.get_tool_context",
        lambda: ToolContext(),
    )

    result = await _execute_tool_search(query="edit file on my computer")

    assert result["query"] == "edit file on my computer"
    assert fake_catalog.query == "edit file on my computer"


@pytest.mark.asyncio
async def test_execute_tool_search_preserves_client_execution_scope_for_autoload(monkeypatch):
    class FakeServerCatalog:
        def search(self, query=None, top_k=5, server_name=None, allowlist=None):
            return []

        def is_ambiguous(self, tool_name):
            return False

    class FakeClientCatalog:
        tool_count = 1
        session_id = "session-7"
        catalog_version = 7

        def search(self, query=None, top_k=5, server_name=None, allowlist=None):
            return [
                ClientToolDescriptor(
                    tool_name="client__shell_execute",
                    server_name="native",
                    description="Run a shell command locally.",
                    arg_names=["command"],
                    required_arg_names=["command"],
                    qualified_tool_id="native::shell_execute",
                    origin="client_native",
                    device_id="device-123",
                    session_id="session-7",
                    catalog_version=7,
                    tool_instance_id="instance-7",
                )
            ]

    class DeferredStateStub:
        def __init__(self):
            self.client_calls = []

        def autoload(self, **kwargs):
            return []

        def autoload_client_tools(self, **kwargs):
            self.client_calls.append(kwargs)
            return kwargs["references"]

    deferred_state = DeferredStateStub()

    async def fake_get_global_mcp_manager():
        return object()

    async def fake_get_tool_catalog(_manager):
        return FakeServerCatalog()

    monkeypatch.setattr(
        "app.ai.tool_search_tool.get_global_mcp_manager",
        fake_get_global_mcp_manager,
    )
    monkeypatch.setattr(
        "app.ai.tool_search_tool.get_tool_catalog",
        fake_get_tool_catalog,
    )
    monkeypatch.setattr(
        "app.ai.tool_search_tool.get_client_tool_catalog",
        lambda device_id, user_id: FakeClientCatalog(),
    )
    monkeypatch.setattr(
        "app.ai.tool_search_tool.get_deferred_tool_state",
        lambda: deferred_state,
    )
    monkeypatch.setattr(
        "app.ai.tool_search_tool.get_tool_context",
        lambda: ToolContext(
            conversation_id="conversation-1",
            user_id="user-1",
            agent_key="chat",
            device_id="device-123",
        ),
    )

    result = await _execute_tool_search(query="run shell command")

    assert result["loaded_count"] == 1
    assert deferred_state.client_calls
    autoload_call = deferred_state.client_calls[0]
    assert autoload_call["device_id"] == "device-123"
    assert autoload_call["session_id"] == "session-7"
    assert autoload_call["references"][0].tool_instance_id == "instance-7"
    assert autoload_call["references"][0].catalog_version == 7


def test_merge_search_results_prefers_client_variant_for_same_qualified_tool():
    server_results = [
        ToolDescriptor(
            tool_name="edit_block",
            server_name="desktop_commander",
            description="Apply surgical edits to files.",
            arg_names=["file_path", "old_string", "new_string"],
            required_arg_names=["file_path"],
            schema_fingerprint="server-fingerprint",
        )
    ]
    client_results = [
        ClientToolDescriptor(
            tool_name="client__desktop_commander__edit_block",
            server_name="desktop_commander",
            description="Apply surgical edits to files.",
            arg_names=["file_path", "old_string", "new_string"],
            required_arg_names=["file_path"],
            qualified_tool_id="desktop_commander::edit_block",
            origin="client_mcp",
            device_id="device-123",
        )
    ]

    public_results, internal_results = _merge_search_results(
        server_results=server_results,
        client_results=client_results,
        query="edit file",
        top_k=5,
    )

    assert len(public_results) == 1
    assert public_results[0]["tool_name"] == "client__desktop_commander__edit_block"
    assert internal_results[0]["is_client_tool"] is True


def test_merge_search_results_keeps_distinct_tools_when_qualified_ids_differ():
    server_results = [
        ToolDescriptor(
            tool_name="edit_block",
            server_name="server_editor",
            description="Apply surgical edits to files.",
            arg_names=["file_path", "old_string", "new_string"],
            required_arg_names=["file_path"],
            schema_fingerprint="server-fingerprint",
        )
    ]
    client_results = [
        ClientToolDescriptor(
            tool_name="client__desktop_commander__edit_block",
            server_name="desktop_commander",
            description="Apply surgical edits to files.",
            arg_names=["file_path", "old_string", "new_string"],
            required_arg_names=["file_path"],
            qualified_tool_id="desktop_commander::edit_block",
            origin="client_mcp",
            device_id="device-123",
        )
    ]

    public_results, _ = _merge_search_results(
        server_results=server_results,
        client_results=client_results,
        query="edit file",
        top_k=5,
    )

    assert [result["tool_name"] for result in public_results] == [
        "edit_block",
        "client__desktop_commander__edit_block",
    ]
