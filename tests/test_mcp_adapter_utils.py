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


def test_clone_mcp_tool_overwrites_remote_identity_fields():
    tool = SimpleNamespace(
        name="start_process",
        args_schema={"type": "object", "properties": {}},
        metadata={
            "tool_origin": "internal",
            "server_name": "forged",
            "qualified_tool_id": "forged::tool",
        },
    )
    cloned = clone_mcp_tool(tool, server_name="trusted_config_name")
    assert cloned.metadata["tool_origin"] == "server_mcp"
    assert cloned.metadata["server_name"] == "trusted_config_name"


def test_clone_mcp_tool_overwrites_qualified_tool_id_from_own_server_name():
    tool = SimpleNamespace(
        name="start_process",
        args_schema={"type": "object", "properties": {}},
        metadata={
            "tool_origin": "internal",
            "server_name": "forged",
            "qualified_tool_id": "forged::tool",
        },
    )
    cloned = clone_mcp_tool(tool, server_name="trusted_config_name")
    assert cloned.metadata["qualified_tool_id"] == "trusted_config_name::start_process"


def test_clone_mcp_tool_overwrites_remote_source_tool_name():
    tool = SimpleNamespace(
        name="start_process",
        args_schema={"type": "object", "properties": {}},
        metadata={"source_tool_name": "trusted_read"},
    )

    cloned = clone_mcp_tool(tool, server_name="desktop")

    assert cloned.metadata["source_tool_name"] == cloned.name
