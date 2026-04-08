from types import SimpleNamespace

from app.core.mcp_adapter_utils import clean_mcp_tool_name, clone_mcp_tool


def test_clean_mcp_tool_name_strips_default_api_prefix():
    assert clean_mcp_tool_name("default_api:inspect") == "inspect"


def test_clean_mcp_tool_name_strips_matching_server_alias_prefix():
    assert clean_mcp_tool_name("widgets:widget_create", server_name="widgets") == "widget_create"


def test_clean_mcp_tool_name_preserves_unknown_prefixes():
    assert clean_mcp_tool_name("urn:tool") == "urn:tool"


def test_clone_mcp_tool_uses_server_name_when_normalizing():
    raw_schema = {"type": "object", "properties": {"mode": {"type": "string"}}}
    tool = SimpleNamespace(
        name="default_api:inspect",
        description="Inspect something",
        args_schema=raw_schema,
    )

    cloned = clone_mcp_tool(tool, server_name="demo")

    assert cloned is not tool
    assert cloned.name == "inspect"
    assert cloned.args_schema == raw_schema
