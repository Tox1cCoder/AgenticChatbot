from __future__ import annotations

from app.ai.mcp_tool_catalog import ToolDescriptor
from app.ai.tool_search_scoring import rank_tool_candidates


def _tool(
    name: str,
    description: str,
    args: list[str],
    required: list[str] | None = None,
) -> ToolDescriptor:
    return ToolDescriptor(
        tool_name=name,
        server_name="desktop_commander",
        description=description,
        arg_names=args,
        required_arg_names=required or [],
        schema_fingerprint=f"fp-{name}",
    )


def _desktop_tools() -> list[ToolDescriptor]:
    return [
        _tool(
            "start_search",
            "Start a streaming search that can return results progressively. "
            "Search files by path and pattern.",
            ["path", "pattern", "searchType", "filePattern", "ignoreCase", "maxResults"],
            ["path", "pattern"],
        ),
        _tool(
            "get_config",
            "Get the complete server configuration as JSON. Config includes "
            "blockedCommands and shell command policy.",
            [],
            [],
        ),
        _tool(
            "interact_with_process",
            "Send input to a running process and receive the response.",
            ["pid", "input", "timeout_ms", "wait_for_prompt"],
            ["pid", "input"],
        ),
        _tool(
            "start_process",
            "Start a new terminal process with intelligent state detection. "
            "Primary tool for command execution and data processing.",
            ["command", "timeout_ms", "shell", "verbose_timing"],
            ["command", "timeout_ms"],
        ),
        _tool(
            "create_directory",
            "Create a new directory or ensure a directory exists.",
            ["path"],
            ["path"],
        ),
        _tool(
            "edit_block",
            "Apply surgical edits to files. Best for patching existing text.",
            ["file_path", "old_string", "new_string", "expected_replacements"],
            ["file_path"],
        ),
        _tool(
            "write_file",
            "Write or append to file contents.",
            ["path", "content", "mode"],
            ["path", "content"],
        ),
    ]


def _rank(query: str):
    return rank_tool_candidates(query=query, candidates=_desktop_tools())


def test_run_shell_command_prefers_start_process():
    ranked = _rank("run shell command")

    assert ranked[0].tool.tool_name == "start_process"
    assert ranked[0].confidence == "high"
    assert "shell_exec" in ranked[0].profile.capabilities
    assert ranked[0].autoload_eligible is True
    assert all(
        item.tool.tool_name not in {"start_search", "get_config"}
        for item in ranked[:2]
    )


def test_run_python_script_prefers_start_process():
    ranked = _rank("run python script")

    assert ranked[0].tool.tool_name == "start_process"
    assert ranked[0].confidence == "high"


def test_search_file_contents_prefers_start_search():
    ranked = _rank("search file contents")

    assert ranked[0].tool.tool_name == "start_search"
    assert "file_search" in ranked[0].profile.capabilities


def test_write_new_file_prefers_write_file_over_edit_block():
    ranked = _rank("write file")

    assert ranked[0].tool.tool_name == "write_file"
    assert ranked[1].tool.tool_name != "write_file"


def test_patch_existing_file_prefers_edit_block():
    ranked = _rank("apply_patch")

    assert ranked[0].tool.tool_name == "edit_block"


def test_get_server_config_prefers_get_config():
    ranked = _rank("get server config")

    assert ranked[0].tool.tool_name == "get_config"
