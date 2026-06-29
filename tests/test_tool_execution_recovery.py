from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from app.ai.tool_execution import execute_tool_calls
from app.core.config import settings


@pytest.mark.asyncio
async def test_execute_tool_calls_recovers_missing_client_tool_from_active_catalog(monkeypatch):
    invoked_args: list[dict] = []

    class _ClientTool:
        name = "client__desktop_commander__start_process"

        async def ainvoke(self, args: dict):
            invoked_args.append(args)
            return "started"

    async def _noop_refresh(**kwargs):
        return None

    monkeypatch.setattr(
        "app.ai.tool_execution._refresh_tool_map_after_search",
        _noop_refresh,
    )
    monkeypatch.setattr(
        "app.ai.tool_execution.get_client_runtime_tools",
        lambda **kwargs: [_ClientTool()],
    )
    monkeypatch.setattr("app.ai.tool_execution.is_client_tool", lambda _tool: True)
    monkeypatch.setattr(
        "app.ai.tool_execution.get_client_tool_device_id",
        lambda _tool: "device-123",
    )
    monkeypatch.setattr(
        "app.ai.tool_execution._mark_tool_used_if_deferred",
        lambda _tool_name: None,
    )

    outputs, artifacts, images = await execute_tool_calls(
        tool_calls=[
            {
                "id": "tool-1",
                "name": "client__desktop_commander__start_process",
                "args": {"command": "notepad.exe"},
            }
        ],
        tool_map={},
        device_id="device-123",
        user_id="user-123",
        conversation_id="conversation-1",
        agent=SimpleNamespace(agent_config_key="chat"),
    )

    assert invoked_args == [{"command": "notepad.exe"}]
    assert outputs == [
        {
            "tool_call_id": "tool-1",
            "name": "client__desktop_commander__start_process",
            "content": "started",
            "render": artifacts[0]["render"],
        }
    ]
    assert artifacts[0]["status"] == "success"
    assert images == []


@pytest.mark.asyncio
async def test_execute_tool_calls_recovers_aliased_server_tool(monkeypatch):
    from app.ai.deferred_tool_state import get_deferred_tool_state, reset_deferred_tool_state
    from app.ai.mcp_tool_catalog import ToolReference

    invoked = []

    class _ServerTool:
        name = "search"

        async def ainvoke(self, args):
            invoked.append(args)
            return "searched"

    brave_tool = _ServerTool()

    class _Manager:
        async def get_tools(self):
            return [brave_tool]

        def get_server_for_tool(self, tool):
            return "brave"

    async def _manager():
        return _Manager()

    async def _noop_refresh(**kwargs):
        return None

    reset_deferred_tool_state()
    try:
        get_deferred_tool_state().autoload(
            conversation_id="conv-1",
            agent_key="chat",
            references=[
                ToolReference(
                    tool_name="search",
                    server_name="brave",
                    call_name="brave__search",
                )
            ],
        )
        monkeypatch.setattr("app.ai.tool_execution._refresh_tool_map_after_search", _noop_refresh)
        monkeypatch.setattr("app.ai.mcp_registry.get_global_mcp_manager", _manager)

        outputs, artifacts, _ = await execute_tool_calls(
            tool_calls=[{"id": "call-1", "name": "brave__search", "args": {"query": "x"}}],
            tool_map={},
            conversation_id="conv-1",
            user_id="user-1",
            agent=SimpleNamespace(agent_config_key="chat", tool_state_key="chat"),
        )

        assert invoked == [{"query": "x"}]
        assert outputs[0]["name"] == "brave__search"
        assert artifacts[0]["status"] == "success"
    finally:
        reset_deferred_tool_state()


@pytest.mark.asyncio
async def test_recover_missing_tool_does_not_bind_unloaded_server_tool(monkeypatch):
    from app.ai.deferred_tool_state import reset_deferred_tool_state
    from app.ai.tool_execution import _recover_missing_tool
    from app.core.config import settings

    reset_deferred_tool_state()
    monkeypatch.setattr(settings, "mcp_tool_search_enabled", True)

    dangerous_tool = SimpleNamespace(
        name="delete_everything",
        description="Unselected server tool",
        metadata={
            "tool_origin": "server_mcp",
            "server_name": "admin",
            "qualified_tool_id": "admin::delete_everything",
        },
    )

    class _Manager:
        async def get_tools(self):
            return [dangerous_tool]

        def get_server_for_tool(self, tool):
            return "admin"

    async def _manager():
        return _Manager()

    reset_deferred_tool_state()
    try:
        monkeypatch.setattr("app.ai.mcp_registry.get_global_mcp_manager", _manager)
        tool_map: dict[str, object] = {}

        recovered = await _recover_missing_tool(
            tool_name="delete_everything",
            tool_map=tool_map,
            agent=SimpleNamespace(
                agent_config_key="custom",
                tool_state_key="custom_agent:abc",
            ),
            conversation_id="conv-1",
            user_id="user-1",
            device_id=None,
        )

        assert recovered is None
        assert "delete_everything" not in tool_map
    finally:
        reset_deferred_tool_state()


@pytest.mark.asyncio
async def test_execute_tool_calls_times_out_slow_tool(monkeypatch):
    monkeypatch.setattr(settings, "tool_execution_timeout", 0.01)
    monkeypatch.setattr(settings, "tool_execution_max_retries", 0)

    class _SlowTool:
        name = "slow_tool"
        metadata = {}

        async def ainvoke(self, args):
            await asyncio.sleep(1)
            return "too late"

    outputs, artifacts, _ = await execute_tool_calls(
        tool_calls=[{"id": "call-1", "name": "slow_tool", "args": {}}],
        tool_map={"slow_tool": _SlowTool()},
    )

    payload = json.loads(outputs[0]["content"])
    assert payload["status"] == "error"
    assert payload["error_type"] == "timeout"
    assert payload["retryable"] is True
    assert artifacts[0]["status"] == "error"
    assert artifacts[0]["error_type"] == "timeout"


@pytest.mark.asyncio
async def test_execute_tool_calls_retries_retry_safe_transient_failure(monkeypatch):
    monkeypatch.setattr(settings, "tool_execution_timeout", 1)
    monkeypatch.setattr(settings, "tool_execution_max_retries", 1)
    calls = 0

    class _RetrySafeTool:
        name = "safe_reader"
        metadata = {"retry_safe": True}

        async def ainvoke(self, args):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ConnectionError("connection reset")
            return "ok"

    outputs, artifacts, _ = await execute_tool_calls(
        tool_calls=[{"id": "call-1", "name": "safe_reader", "args": {}}],
        tool_map={"safe_reader": _RetrySafeTool()},
    )

    assert calls == 2
    assert outputs[0]["content"] == "ok"
    assert artifacts[0]["status"] == "success"


@pytest.mark.asyncio
async def test_execute_tool_calls_does_not_auto_retry_unknown_side_effect_tool(monkeypatch):
    monkeypatch.setattr(settings, "tool_execution_timeout", 1)
    monkeypatch.setattr(settings, "tool_execution_max_retries", 2)
    calls = 0

    class _MaybeSideEffectTool:
        name = "send_message"
        metadata = {}

        async def ainvoke(self, args):
            nonlocal calls
            calls += 1
            raise ConnectionError("connection reset")

    outputs, artifacts, _ = await execute_tool_calls(
        tool_calls=[{"id": "call-1", "name": "send_message", "args": {"body": "hi"}}],
        tool_map={"send_message": _MaybeSideEffectTool()},
    )

    payload = json.loads(outputs[0]["content"])
    assert calls == 1
    assert payload["error_type"] == "network"
    assert payload["retryable"] is True
    assert artifacts[0]["attempts"] == 1
