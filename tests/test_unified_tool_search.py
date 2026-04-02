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
