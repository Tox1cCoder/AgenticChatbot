"""Session lifecycle of the sidecar's local MCP manager, against a real stdio server.

A server process that lives only for one call loses everything it keeps between
calls: Desktop Commander reports a started process as running, and the next
call cannot find it. These tests pin the behavior that fixes that.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import psutil
import pytest
from mcp.shared.exceptions import McpError

from client_backend.schemas.mcp_config import MCPProfileScope
from client_backend.services.local_mcp_manager import LocalMCPManager
from client_backend.services.mcp_config_store import MCPConfigStore

FIXTURE_SERVER = Path(__file__).resolve().parents[1] / "fixtures" / "mcp_stateful_server.py"


def _store(tmp_path: Path) -> MCPConfigStore:
    application_root = tmp_path / "application"
    registry_path = application_root / "app" / "ai" / "mcp_config.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps({"schemaVersion": 2, "servers": {}}), encoding="utf-8")
    store = MCPConfigStore(
        MCPProfileScope(user_id="user-1", device_identifier="device-a"),
        registry_path=registry_path,
        application_root=application_root,
        profile_root=tmp_path / "profiles",
    )
    store.save_custom_server(
        "stateful",
        {"transport": "stdio", "command": sys.executable, "args": [str(FIXTURE_SERVER)]},
        env={},
        headers={},
    )
    return store


@pytest.fixture
async def manager(tmp_path):
    manager = LocalMCPManager(store=_store(tmp_path))
    await manager.initialize()
    try:
        yield manager
    finally:
        await manager.shutdown()


def _text(result) -> str:
    return "".join(block["text"] for block in result if block.get("type") == "text")


async def _call(manager: LocalMCPManager, tool: str, arguments=None, timeout: float = 15.0):
    return _text(await manager.call_tool(f"stateful::{tool}", arguments or {}, timeout=timeout))


async def _wait_until_gone(pid: int, seconds: float = 10.0) -> bool:
    deadline = asyncio.get_running_loop().time() + seconds
    while asyncio.get_running_loop().time() < deadline:
        if not psutil.pid_exists(pid):
            return True
        await asyncio.sleep(0.1)
    return False


async def test_consecutive_calls_reach_the_same_server_state(manager):
    assert await _call(manager, "increment") == "1"
    assert await _call(manager, "increment") == "2"


async def test_timed_out_call_keeps_the_server_session(manager):
    assert await _call(manager, "increment") == "1"

    with pytest.raises(TimeoutError):
        await _call(manager, "wait", {"seconds": 5}, timeout=0.3)

    assert await _call(manager, "increment") == "2"


async def test_server_process_lives_until_shutdown(manager):
    pid = int(await _call(manager, "process_id"))
    assert psutil.pid_exists(pid)

    await manager.shutdown()

    assert await _wait_until_gone(pid)


async def test_crash_mid_call_is_reported_and_not_retried(manager, tmp_path):
    marker = tmp_path / "crash-calls.txt"

    with pytest.raises(McpError):
        await _call(manager, "crash", {"marker_path": str(marker)})

    # The call may already have had its effect, so it must reach the server
    # exactly once.
    assert marker.read_text(encoding="utf-8").splitlines() == ["crash"]


async def test_call_after_a_crash_is_served_by_a_fresh_server(manager, tmp_path):
    assert await _call(manager, "increment") == "1"
    with pytest.raises(McpError):
        await _call(manager, "crash", {"marker_path": str(tmp_path / "crash-calls.txt")})

    assert await _call(manager, "increment") == "1"


async def test_server_that_died_while_idle_is_restarted_for_the_next_call(manager):
    pid = int(await _call(manager, "process_id"))
    psutil.Process(pid).kill()
    assert await _wait_until_gone(pid)

    new_pid = int(await _call(manager, "process_id"))

    assert new_pid != pid
