from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from client_backend.api import mcp as mcp_api


class _ManagerStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.tools = [
            SimpleNamespace(
                name="inspect",
                description="Alpha inspector",
                input_schema={"type": "object", "properties": {}},
                server_name="alpha",
                qualified_id="alpha::inspect",
            ),
            SimpleNamespace(
                name="inspect",
                description="Beta inspector",
                input_schema={"type": "object", "properties": {}},
                server_name="beta",
                qualified_id="beta::inspect",
            ),
        ]

    async def initialize(self) -> None:
        return None

    def get_all_tools(self) -> list[Any]:
        return self.tools

    async def call_tool(self, qualified_tool_id: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((qualified_tool_id, arguments))
        return {"selected": qualified_tool_id}


@pytest.mark.asyncio
async def test_execute_duplicate_name_tool_uses_requested_qualified_identity(monkeypatch):
    manager = _ManagerStub()
    monkeypatch.setattr(mcp_api, "get_mcp_manager", lambda: manager)

    response = await mcp_api.execute_mcp_tool(
        "inspect",
        {
            "arguments": {"value": 7},
            "serverName": "beta",
            "qualifiedToolId": "beta::inspect",
        },
        object(),
    )
    body = json.loads(response.body)

    assert manager.calls == [("beta::inspect", {"value": 7})]
    assert body["data"]["serverName"] == "beta"
    assert body["data"]["qualifiedId"] == "beta::inspect"


@pytest.mark.asyncio
async def test_legacy_bare_name_execution_remains_compatible(monkeypatch):
    manager = _ManagerStub()
    monkeypatch.setattr(mcp_api, "get_mcp_manager", lambda: manager)

    await mcp_api.execute_mcp_tool("inspect", {"arguments": {}}, object())

    assert manager.calls == [("alpha::inspect", {})]


@pytest.mark.asyncio
async def test_execute_rejects_mismatched_server_and_qualified_identity(monkeypatch):
    manager = _ManagerStub()
    monkeypatch.setattr(mcp_api, "get_mcp_manager", lambda: manager)

    with pytest.raises(HTTPException) as exc_info:
        await mcp_api.execute_mcp_tool(
            "inspect",
            {
                "arguments": {},
                "serverName": "alpha",
                "qualifiedToolId": "beta::inspect",
            },
            object(),
        )

    assert exc_info.value.status_code == 404
    assert manager.calls == []
