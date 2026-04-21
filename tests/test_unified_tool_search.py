import pytest

from app.ai.client_tool_catalog import ClientToolDescriptor
from app.ai.mcp_tool_catalog import McpToolCatalog, ToolDescriptor
from app.ai.tool_context import ToolContext
from app.ai.tool_search_tool import (
    _execute_tool_search,
    _merge_search_results,
)

# ---------------------------------------------------------------------------
# Phase 0 Regression: inventory mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_search_without_query_returns_inventory_metadata(monkeypatch):
    """tool_search() with no query must return server-level inventory summaries,
    not the first top_k tools from a stable slice."""

    class FakeServerCatalog:
        def search(self, query=None, top_k=5, server_name=None, allowlist=None):
            # Should NOT be called in inventory mode
            raise AssertionError("search() must not be called in inventory mode")

        def is_ambiguous(self, tool_name):
            return False

        def get_server_inventory(self, allowlist=None):
            return [
                {"server_name": "tavily", "tool_count": 2},
                {"server_name": "time", "tool_count": 1},
            ]

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
        "app.ai.tool_search_tool.get_tool_context",
        lambda: ToolContext(),
    )

    result = await _execute_tool_search(query=None)

    # Must return inventory metadata, not a tools list
    assert "inventory" in result
    assert result["mode"] == "inventory"
    servers = {s["server_name"] for s in result["inventory"]}
    assert "tavily" in servers
    assert "time" in servers


def test_server_inventory_includes_server_descriptions():
    """Inventory mode should expose a short server description so the model can
    identify the right integration before choosing tools."""
    catalog = McpToolCatalog(mcp_manager=object())
    catalog._tools_by_server = {
        "canva-dev": [
            ToolDescriptor(
                tool_name="read_canva_docs",
                server_name="canva-dev",
                description="Read Canva developer docs.",
                arg_names=[],
                required_arg_names=[],
                schema_fingerprint="fp-1",
            )
        ]
    }
    catalog._server_descriptions = {"canva-dev": "Canva developer tooling and documentation"}

    inventory = catalog.get_server_inventory()

    assert inventory == [
        {
            "server_name": "canva-dev",
            "description": "Canva developer tooling and documentation",
            "tool_count": 1,
        }
    ]


def test_resolve_server_name_fuzzy_matches_tokenized_variant():
    """Server resolution should recover from plausible identifier variants
    without requiring hard-coded aliases."""
    catalog = McpToolCatalog(mcp_manager=object())
    catalog._tools_by_server = {
        "excel": [
            ToolDescriptor(
                tool_name="excel_read_sheet",
                server_name="excel",
                description="Read values from an Excel sheet.",
                arg_names=["file"],
                required_arg_names=["file"],
                schema_fingerprint="fp-excel",
            )
        ],
        "canva-dev": [
            ToolDescriptor(
                tool_name="read_canva_docs",
                server_name="canva-dev",
                description="Read Canva developer documentation.",
                arg_names=[],
                required_arg_names=[],
                schema_fingerprint="fp-canva",
            )
        ],
    }
    catalog._server_name_lower_map = {"excel": "excel", "canva-dev": "canva-dev"}
    catalog._server_descriptions = {
        "excel": "Excel spreadsheet automation tools",
        "canva-dev": "Canva developer tooling and documentation",
    }

    assert catalog.resolve_server_name("excel-server") == "excel"


def test_resolve_server_name_fuzzy_returns_none_for_weak_match():
    """Fuzzy server resolution should stay conservative when the input does not
    clearly identify a configured server."""
    catalog = McpToolCatalog(mcp_manager=object())
    catalog._tools_by_server = {
        "excel": [
            ToolDescriptor(
                tool_name="excel_read_sheet",
                server_name="excel",
                description="Read values from an Excel sheet.",
                arg_names=["file"],
                required_arg_names=["file"],
                schema_fingerprint="fp-excel",
            )
        ]
    }
    catalog._server_name_lower_map = {"excel": "excel"}
    catalog._server_descriptions = {"excel": "Excel spreadsheet automation tools"}

    assert catalog.resolve_server_name("totally-unrelated") is None


# ---------------------------------------------------------------------------
# Phase 0 Regression: same-name server tools do not collapse
# ---------------------------------------------------------------------------


def test_merge_search_results_keeps_same_name_tools_from_different_servers():
    """Two server tools with the same name but different servers must both
    appear in the merged results. Phase 3 assigns deterministic aliases
    (call_name) so the dedup logic can distinguish them."""
    # Simulate catalog collision detection: set call_name as the catalog would
    tool_a = ToolDescriptor(
        tool_name="search",
        server_name="tavily",
        description="Web search via Tavily.",
        arg_names=["query"],
        required_arg_names=["query"],
        schema_fingerprint="fp-a",
        call_name="tavily__search",
    )
    tool_b = ToolDescriptor(
        tool_name="search",
        server_name="brave",
        description="Web search via Brave.",
        arg_names=["query"],
        required_arg_names=["query"],
        schema_fingerprint="fp-b",
        call_name="brave__search",
    )

    public_results, internal_results = _merge_search_results(
        server_results=[tool_a, tool_b],
        client_results=[],
        query="web search",
        top_k=5,
    )

    # Both must remain independently representable
    assert len(public_results) == 2
    assert len(internal_results) == 2
    # Different servers must be distinguishable in internal results
    internal_servers = {r["server_name"] for r in internal_results}
    assert "tavily" in internal_servers
    assert "brave" in internal_servers
    # Public results expose call_name aliases so the model knows what to invoke
    public_names = {r["tool_name"] for r in public_results}
    assert "tavily__search" in public_names
    assert "brave__search" in public_names


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
async def test_execute_tool_search_promotes_named_integration_query_to_server_inventory(
    monkeypatch,
):
    class FakeCatalog:
        def __init__(self):
            self.calls = []

        def resolve_server_name(self, server_name):
            if server_name == "Canva":
                return "canva-dev"
            return None

        def search(self, query=None, top_k=5, server_name=None, allowlist=None):
            self.calls.append((query, server_name))
            return [
                ToolDescriptor(
                    tool_name="read_canva_docs",
                    server_name="canva-dev",
                    description="Read Canva developer docs.",
                    arg_names=[],
                    required_arg_names=[],
                    schema_fingerprint="fp-canva",
                )
            ]

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

    result = await _execute_tool_search(query="Canva")

    assert result["mode"] == "per_server_inventory"
    assert result["resolved_server_name"] == "canva-dev"
    assert fake_catalog.calls == [(None, "canva-dev")]


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
                    tool_name="client__desktop_commander__start_process",
                    server_name="desktop_commander",
                    description="Start a local process via Desktop Commander.",
                    arg_names=["command"],
                    required_arg_names=["command"],
                    qualified_tool_id="desktop_commander::start_process",
                    origin="client_mcp",
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

    result = await _execute_tool_search(query="start local process")

    assert result["loaded_count"] == 1
    assert deferred_state.client_calls
    autoload_call = deferred_state.client_calls[0]
    assert autoload_call["device_id"] == "device-123"
    assert autoload_call["session_id"] == "session-7"
    assert autoload_call["references"][0].tool_instance_id == "instance-7"
    assert autoload_call["references"][0].catalog_version == 7


@pytest.mark.asyncio
async def test_execute_tool_search_client_scope_uses_client_inventory_only(monkeypatch):
    class FakeServerCatalog:
        def get_server_inventory(self, allowlist=None):
            return [
                {
                    "server_name": "server-only",
                    "description": "Server inventory that must be hidden from sidecar scope",
                    "tool_count": 1,
                }
            ]

    class FakeClientCatalog:
        tool_count = 1
        session_id = "session-7"
        catalog_version = 7

        def get_server_inventory(self, allowlist=None):
            return [
                {
                    "server_name": "sidecar-only",
                    "description": "Client inventory visible from the sidecar",
                    "tool_count": 1,
                }
            ]

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
        "app.ai.tool_search_tool.get_tool_context",
        lambda: type(
            "ClientScopeContext",
            (),
            {
                "conversation_id": "conversation-1",
                "user_id": "user-1",
                "agent_key": "chat",
                "device_id": "device-123",
                "tool_scope": "client_only",
            },
        )(),
    )

    result = await _execute_tool_search(query=None)

    assert result["mode"] == "inventory"
    assert result["inventory"] == [
        {
            "server_name": "sidecar-only",
            "description": "Client inventory visible from the sidecar",
            "tool_count": 1,
        }
    ]


@pytest.mark.asyncio
async def test_execute_tool_search_client_scope_excludes_server_results(monkeypatch):
    class FakeServerCatalog:
        def search(self, query=None, top_k=5, server_name=None, allowlist=None):
            return [
                ToolDescriptor(
                    tool_name="server_edit_file",
                    server_name="server-editor",
                    description="Edit files on the server runtime.",
                    arg_names=["path"],
                    required_arg_names=["path"],
                    schema_fingerprint="server-fp",
                )
            ]

        def is_ambiguous(self, tool_name):
            return False

    class FakeClientCatalog:
        tool_count = 1
        session_id = "session-7"
        catalog_version = 7

        def search(self, query=None, top_k=5, server_name=None, allowlist=None):
            return [
                ClientToolDescriptor(
                    tool_name="client__desktop_commander__edit_block",
                    server_name="desktop_commander",
                    description="Edit files on the connected sidecar.",
                    arg_names=["file_path"],
                    required_arg_names=["file_path"],
                    qualified_tool_id="desktop_commander::edit_block",
                    origin="client_mcp",
                    device_id="device-123",
                )
            ]

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
        "app.ai.tool_search_tool.get_tool_context",
        lambda: type(
            "ClientScopeContext",
            (),
            {
                "conversation_id": "conversation-1",
                "user_id": "user-1",
                "agent_key": "chat",
                "device_id": "device-123",
                "tool_scope": "client_only",
            },
        )(),
    )

    result = await _execute_tool_search(query="edit a local file")

    assert [entry["tool_name"] for entry in result["results"]] == [
        "client__desktop_commander__edit_block"
    ]


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
