import asyncio
import json
import logging
import sys
from pathlib import Path

import pytest

from app.ai.mcp_integration import MCPManager


def test_default_stdio_servers_resolve_from_outside_repository(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    manager = MCPManager()
    manager._ensure_config_loaded()
    server_config = manager._build_server_config()

    assert server_config
    for config in server_config.values():
        assert config["command"] == sys.executable
        script = Path(config["args"][0])
        assert script.is_absolute()
        assert script.is_file()


def test_explicit_stdio_script_resolves_relative_to_config_file(tmp_path, monkeypatch):
    config_dir = tmp_path / "custom"
    config_dir.mkdir()
    server_script = config_dir / "server.py"
    server_script.write_text("print('ok')", encoding="utf-8")
    config_path = config_dir / "mcp_config.json"
    config_path.write_text(
        json.dumps(
            {
                "mcp_servers": {
                    "custom": {
                        "enabled": True,
                        "transport": "stdio",
                        "command": sys.executable,
                        "args": ["server.py"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    manager = MCPManager(config_path=str(config_path))
    manager._ensure_config_loaded()

    resolved = manager._build_server_config()["custom"]

    assert resolved["args"] == [str(server_script.resolve())]


def test_explicit_stdio_resolves_all_path_fields_without_touching_path_commands_or_urls(
    tmp_path, monkeypatch
):
    config_dir = tmp_path / "custom"
    command = config_dir / "bin" / "custom-runner"
    asset = config_dir / "assets" / "schema.json"
    executable_arg = config_dir / "helper.exe"
    runtime_dir = config_dir / "runtime"
    for path in (command, asset, executable_arg):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    runtime_dir.mkdir()

    # A same-named config-relative file must not turn a bare PATH command into
    # a file path. Commands such as npx/python are intentionally PATH-resolved.
    (config_dir / "npx").write_text("fixture", encoding="utf-8")
    (config_dir / "npx.cmd").write_text("fixture", encoding="utf-8")
    config_path = config_dir / "mcp_config.json"
    config_path.write_text(
        json.dumps(
            {
                "mcp_servers": {
                    "custom": {
                        "enabled": True,
                        "transport": "stdio",
                        "command": "bin/custom-runner",
                        "args": [
                            "--watch",
                            "assets/schema.json",
                            "helper.exe",
                            "missing-server.py",
                            "postgresql://db.example/service",
                        ],
                        "cwd": "runtime",
                    },
                    "package": {
                        "enabled": True,
                        "transport": "stdio",
                        "command": "npx",
                        "args": ["--yes", "@scope/package", "https://example.test/spec"],
                    },
                    "windows-package": {
                        "enabled": True,
                        "transport": "stdio",
                        "command": "npx.cmd",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    manager = MCPManager(config_path=str(config_path))
    manager._ensure_config_loaded()
    resolved = manager._build_server_config()

    assert resolved["custom"]["command"] == str(command.resolve())
    assert resolved["custom"]["args"] == [
        "--watch",
        str(asset.resolve()),
        str(executable_arg.resolve()),
        str((config_dir / "missing-server.py").resolve()),
        "postgresql://db.example/service",
    ]
    assert resolved["custom"]["cwd"] == str(runtime_dir.resolve())
    assert resolved["package"]["command"] == "npx"
    assert resolved["package"]["args"] == [
        "--yes",
        "@scope/package",
        "https://example.test/spec",
    ]
    assert resolved["windows-package"]["command"] == "npx.cmd"


@pytest.mark.asyncio
async def test_server_mcp_manager_cleans_up_stdio_session_without_cancel_scope_error(
    tmp_path,
):
    config_path = tmp_path / "mcp_config.json"
    server_script = Path("app/ai/mcp_servers/time_server.py").resolve()
    config_path.write_text(
        json.dumps(
            {
                "mcp_servers": {
                    "time": {
                        "enabled": True,
                        "transport": "stdio",
                        "command": sys.executable,
                        "args": [str(server_script)],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    manager = MCPManager(config_path=str(config_path))

    await manager.initialize()
    tool = await manager.get_tool_by_name("get_current_time", server_name="time")
    assert tool is not None

    result = await tool.ainvoke({"timezone": "UTC", "format": "%Y-%m-%d"})
    assert "UTC" in str(result)

    await manager.cleanup()


@pytest.mark.asyncio
async def test_server_mcp_manager_closes_session_opened_by_another_task_without_warning(
    tmp_path,
    caplog,
):
    config_path = tmp_path / "mcp_config.json"
    server_script = Path("app/ai/mcp_servers/time_server.py").resolve()
    config_path.write_text(
        json.dumps(
            {
                "mcp_servers": {
                    "time": {
                        "enabled": True,
                        "transport": "stdio",
                        "command": sys.executable,
                        "args": [str(server_script)],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    manager = MCPManager(config_path=str(config_path))

    await manager.initialize()
    load_task = asyncio.create_task(
        manager.get_tool_by_name("get_current_time", server_name="time")
    )
    tool = await load_task
    assert tool is not None

    with caplog.at_level(logging.WARNING, logger="app.ai.mcp_integration"):
        await manager.cleanup()

    assert "Error closing session for time" not in caplog.text


@pytest.mark.asyncio
async def test_server_mcp_tool_cancellation_does_not_escape_as_unbound_local_error(
    tmp_path,
):
    server_script = tmp_path / "slow_mcp_server.py"
    server_script.write_text(
        "\n".join(
            [
                "import asyncio",
                "from mcp.server.fastmcp import FastMCP",
                "",
                "mcp = FastMCP('Slow')",
                "",
                "@mcp.tool()",
                "async def slow_echo(value: str, delay: float = 10.0) -> str:",
                "    await asyncio.sleep(delay)",
                "    return value",
                "",
                "if __name__ == '__main__':",
                "    mcp.run(transport='stdio')",
                "",
            ]
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "mcp_config.json"
    config_path.write_text(
        json.dumps(
            {
                "mcp_servers": {
                    "slow": {
                        "enabled": True,
                        "transport": "stdio",
                        "command": sys.executable,
                        "args": [str(server_script)],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    manager = MCPManager(config_path=str(config_path))

    await manager.initialize()
    tool = await manager.get_tool_by_name("slow_echo", server_name="slow")
    assert tool is not None

    task = asyncio.create_task(tool.ainvoke({"value": "hello", "delay": 5.0}))
    await asyncio.sleep(0.2)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    await manager.cleanup()
