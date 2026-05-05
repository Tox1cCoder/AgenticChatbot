import asyncio
import json
import logging
import sys
from pathlib import Path

import pytest

from app.ai.mcp_integration import MCPManager


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
