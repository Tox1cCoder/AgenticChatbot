from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from client_backend.api import mcp as mcp_api


class _ManagerStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        # Mirrors LocalMCPManager.servers: the enabled servers for this device
        # scope. The scoped routes 404 any name absent from it.
        self.servers = {"alpha": SimpleNamespace(), "beta": SimpleNamespace()}
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

    def get_tools_by_server(self, server_name: str) -> list[Any]:
        return [t for t in self.tools if t.server_name == server_name]

    async def call_tool(
        self,
        qualified_tool_id: str,
        arguments: dict[str, Any],
        timeout: float = 30.0,
    ) -> Any:
        self.calls.append((qualified_tool_id, arguments))
        return {"selected": qualified_tool_id}


def _session():
    return SimpleNamespace(user_id="user-1", device_identifier="device-a")


@pytest.mark.asyncio
async def test_execute_duplicate_name_tool_uses_requested_qualified_identity(monkeypatch):
    manager = _ManagerStub()
    monkeypatch.setattr(mcp_api, "get_mcp_manager", lambda _scope: manager)

    response = await mcp_api.execute_mcp_tool(
        "inspect",
        {
            "arguments": {"value": 7},
            "serverName": "beta",
            "qualifiedToolId": "beta::inspect",
        },
        _session(),
    )
    body = json.loads(response.body)

    assert manager.calls == [("beta::inspect", {"value": 7})]
    assert body["data"]["serverName"] == "beta"
    assert body["data"]["qualifiedId"] == "beta::inspect"


@pytest.mark.asyncio
async def test_ambiguous_bare_name_execution_is_rejected(monkeypatch):
    """Two servers expose ``inspect``; an unqualified execute must not silently
    pick whichever was indexed first.

    Parity with the canonical ``MCPManager.get_tool_by_name`` contract
    (409 AMBIGUOUS_TOOL_NAME). The Streamlit tester always sends
    ``serverName``/``qualifiedToolId``, so only an unqualified caller is
    affected — and for that caller "first wins" was a silent wrong-server
    execution.
    """
    manager = _ManagerStub()
    monkeypatch.setattr(mcp_api, "get_mcp_manager", lambda _scope: manager)

    with pytest.raises(HTTPException) as excinfo:
        await mcp_api.execute_mcp_tool("inspect", {"arguments": {}}, _session())

    assert excinfo.value.status_code == 409
    assert "alpha" in excinfo.value.detail and "beta" in excinfo.value.detail
    assert manager.calls == [], "no tool may run for an ambiguous request"


@pytest.mark.asyncio
async def test_unambiguous_bare_name_execution_still_runs(monkeypatch):
    """A name owned by exactly one server needs no qualification."""
    manager = _ManagerStub()
    manager.tools = [t for t in manager.tools if t.server_name == "beta"]
    monkeypatch.setattr(mcp_api, "get_mcp_manager", lambda _scope: manager)

    await mcp_api.execute_mcp_tool("inspect", {"arguments": {}}, _session())

    assert manager.calls == [("beta::inspect", {})]


def _tools_client(monkeypatch) -> TestClient:
    manager = _ManagerStub()
    monkeypatch.setattr(mcp_api, "get_mcp_manager", lambda _scope: manager)
    app = FastAPI()
    app.include_router(mcp_api.router)
    app.dependency_overrides[mcp_api.require_local_session] = _session
    return TestClient(app)


def test_list_tools_filters_by_camelcase_server_name(monkeypatch):
    """GET /mcp/tools?serverName=beta must return only that server's tools.

    Regression: the query param was declared as ``server_name`` with no alias, so
    the camelCase key the client sends never bound and the filter returned every
    server's tools.
    """
    client = _tools_client(monkeypatch)

    resp = client.get("/mcp/tools", params={"serverName": "beta"})

    assert resp.status_code == 200
    tools = resp.json()["data"]["tools"]
    assert {tool["serverName"] for tool in tools} == {"beta"}


def test_list_tools_without_filter_returns_all(monkeypatch):
    client = _tools_client(monkeypatch)

    resp = client.get("/mcp/tools")

    assert resp.status_code == 200
    tools = resp.json()["data"]["tools"]
    assert {tool["serverName"] for tool in tools} == {"alpha", "beta"}


@pytest.mark.asyncio
async def test_execute_rejects_mismatched_server_and_qualified_identity(monkeypatch):
    manager = _ManagerStub()
    monkeypatch.setattr(mcp_api, "get_mcp_manager", lambda _scope: manager)

    with pytest.raises(HTTPException) as exc_info:
        await mcp_api.execute_mcp_tool(
            "inspect",
            {
                "arguments": {},
                "serverName": "alpha",
                "qualifiedToolId": "beta::inspect",
            },
            _session(),
        )

    assert exc_info.value.status_code == 404
    assert manager.calls == []
