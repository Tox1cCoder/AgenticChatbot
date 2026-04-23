from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.ai.tool_execution import execute_tool_calls


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
