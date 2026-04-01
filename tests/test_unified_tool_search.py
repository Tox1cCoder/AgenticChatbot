import pytest

from app.ai.tool_context import ToolContext
from app.ai.tool_search_tool import _execute_tool_search, _expand_tool_search_query


def test_expand_tool_search_query_adds_local_file_aliases():
    expanded = _expand_tool_search_query("edit a file on my computer")

    assert expanded is not None
    tokens = set(expanded.split())

    assert {"edit", "write", "update", "modify"} <= tokens
    assert {"file", "files", "filesystem", "path"} <= tokens
    assert {"computer", "local", "device", "client"} <= tokens


def test_expand_tool_search_query_deduplicates_shell_aliases():
    expanded = _expand_tool_search_query("run shell command")

    assert expanded is not None
    tokens = expanded.split()

    assert len(tokens) == len(set(tokens))
    assert {"run", "execute", "shell", "terminal", "command"} <= set(tokens)


@pytest.mark.asyncio
async def test_execute_tool_search_uses_expanded_query(monkeypatch):
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
    assert fake_catalog.query is not None

    expanded_tokens = set(fake_catalog.query.split())
    assert {"edit", "write", "file", "filesystem", "computer", "local"} <= expanded_tokens
