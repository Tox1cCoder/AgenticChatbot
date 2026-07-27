"""Phase-0 characterization: server-scoped MCP tool listing over HTTP.

Drives the mixed-catalog scoping contract through BOTH real apps:

* canonical ``GET /mcp/tools?serverName=...`` (``app.api.mcp.list_tools`` ->
  the real ``MCPService.list_tools`` filter over a mixed catalog), and
* sidecar ``GET /mcp/tools?serverName=...`` (``client_backend.api.mcp`` over a
  real ``LocalMCPManager`` seam).

Findings baked into these tests (see the 2026-07-24 plan, "MCP and capability
architecture findings" #1 and commit ``b55d83a``):

* The ``serverName`` query filter is ALREADY correct on current source, so the
  scoped-query assertions are GREEN contract locks: ``brave_image_search`` must
  never return a ``widgets`` tool. They guard against a regression of the
  query-loss bug that ``b55d83a`` fixed.
* The dedicated scoped route ``GET /mcp/servers/{server_name}/tools`` does NOT
  exist yet (plan T007), so those assertions are RED characterizations.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from dependency_injector import providers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.mcp import router as canonical_mcp_router
from app.core.auth import get_current_user
from app.core.container import Container, setup_auto_injection
from app.services.mcp_service import MCPService
from client_backend.api import mcp as sidecar_mcp
from client_backend.core.auth import require_local_session
from client_backend.core.security import LocalSessionPayload
from client_backend.main import create_app

_BRAVE = "brave_image_search"
_WIDGETS = "widgets"


# ---------------------------------------------------------------------------
# Canonical app harness
# ---------------------------------------------------------------------------

_CANONICAL_CATALOG = [
    {
        "name": "search",
        "description": "Brave image search",
        "args_schema": {"type": "object"},
        "server_name": _BRAVE,
    },
    {
        "name": "render_widget",
        "description": "Render a live widget",
        "args_schema": {"type": "object"},
        "server_name": _WIDGETS,
    },
    {
        "name": "list_widgets",
        "description": "List active widgets",
        "args_schema": {"type": "object"},
        "server_name": _WIDGETS,
    },
]


class _FakeCanonicalManager:
    def __init__(self, catalog: list[dict]):
        self._catalog = catalog

    async def get_all_tools_info(self) -> list[dict]:
        return [dict(t) for t in self._catalog]

    async def list_tool_descriptors(self, server_name: str | None = None) -> list[dict]:
        items = [dict(t) for t in self._catalog]
        if server_name is not None:
            items = [t for t in items if t["server_name"] == server_name]
        return items


def _canonical_service() -> MCPService:
    service = MCPService.__new__(MCPService)
    service.mcp_manager = _FakeCanonicalManager(_CANONICAL_CATALOG)
    return service


def _canonical_app() -> FastAPI:
    app = FastAPI()
    app.include_router(canonical_mcp_router)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id="user-1")
    return app


@pytest.fixture(autouse=True)
def _restore_wiring():
    setup_auto_injection(Container)
    yield
    setup_auto_injection(Container)


def _server_of(tool: dict) -> str | None:
    return tool.get("serverName") or tool.get("server_name")


def test_canonical_scoped_query_returns_only_brave(monkeypatch):
    """Contract lock (GREEN): ``/mcp/tools?serverName=brave_image_search`` never
    returns a ``widgets`` tool. Guards the b55d83a query-loss regression.
    """
    service = _canonical_service()
    with Container.mcp_service.override(providers.Object(service)):
        client = TestClient(_canonical_app())
        resp = client.get("/mcp/tools", params={"serverName": _BRAVE})

    assert resp.status_code == 200, resp.text
    tools = resp.json()["data"]["tools"]
    servers = {_server_of(t) for t in tools}
    assert servers == {_BRAVE}, (
        "scoped canonical request leaked tools from other servers.\n"
        f"servers returned: {sorted(s or '<none>' for s in servers)}\n"
        f"tools: {[t.get('name') for t in tools]}"
    )
    assert not any(_server_of(t) == _WIDGETS for t in tools), (
        f"widgets tool leaked into brave scope: {[t.get('name') for t in tools]}"
    )


def test_canonical_dedicated_scoped_route_exists(monkeypatch):
    """GREEN (plan T007): the dedicated ``GET /mcp/servers/{server_name}/tools``
    route exists and returns ONLY the requested server's tools, with the applied
    scope and a deterministic catalog version.
    """
    service = _canonical_service()
    with Container.mcp_service.override(providers.Object(service)):
        client = TestClient(_canonical_app())
        resp = client.get(f"/mcp/servers/{_BRAVE}/tools")

    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    servers = {_server_of(t) for t in data["tools"]}
    assert servers == {_BRAVE}, (
        f"dedicated scoped route leaked other servers: "
        f"{[t.get('name') for t in data['tools']]}"
    )
    assert not any(_server_of(t) == _WIDGETS for t in data["tools"])
    assert data["scope"]["kind"] == "server"
    assert data["scope"]["serverName"] == _BRAVE
    assert data["serversCount"] == 1
    assert str(data.get("catalogVersion", "")).startswith("sha256:")


# ---------------------------------------------------------------------------
# Sidecar app harness
# ---------------------------------------------------------------------------


def _fake_tool(name: str, server: str):
    return SimpleNamespace(
        name=name,
        description=f"{name} description",
        input_schema={"type": "object"},
        server_name=server,
        qualified_id=f"{server}::{name}",
    )


_SIDECAR_CATALOG = [
    _fake_tool("search", _BRAVE),
    _fake_tool("render_widget", _WIDGETS),
    _fake_tool("list_widgets", _WIDGETS),
]


class _FakeLocalManager:
    def __init__(self, catalog):
        self._catalog = catalog

    async def initialize(self, *args, **kwargs) -> None:
        return None

    def get_all_tools(self):
        return list(self._catalog)

    def get_tools_by_server(self, server_name: str):
        return [t for t in self._catalog if t.server_name == server_name]


def _session() -> LocalSessionPayload:
    now = datetime.now(timezone.utc)
    return LocalSessionPayload(
        user_id="user-1",
        server_user_id="user-1",
        device_id=None,
        device_identifier="dev-abc",
        iat=now,
        exp=now + timedelta(hours=1),
    )


def _sidecar_client(monkeypatch) -> TestClient:
    manager = _FakeLocalManager(_SIDECAR_CATALOG)
    monkeypatch.setattr(sidecar_mcp, "get_mcp_manager", lambda scope=None: manager)
    app = create_app()
    app.dependency_overrides[require_local_session] = lambda: _session()
    return TestClient(app)


def test_sidecar_scoped_query_returns_only_brave(monkeypatch):
    """Contract lock (GREEN): the sidecar ``/mcp/tools?serverName=`` filter
    never returns a ``widgets`` tool for a brave scope."""
    client = _sidecar_client(monkeypatch)
    resp = client.get(
        "/mcp/tools",
        params={"serverName": _BRAVE},
        headers={"Authorization": "Bearer local-session-token"},
    )
    assert resp.status_code == 200, resp.text
    tools = resp.json()["data"]["tools"]
    servers = {_server_of(t) for t in tools}
    assert servers == {_BRAVE}, (
        "scoped sidecar request leaked tools from other servers.\n"
        f"servers returned: {sorted(s or '<none>' for s in servers)}\n"
        f"tools: {[t.get('name') for t in tools]}"
    )
    assert not any(_server_of(t) == _WIDGETS for t in tools), (
        f"widgets tool leaked into brave scope: {[t.get('name') for t in tools]}"
    )


def test_sidecar_dedicated_scoped_route_exists(monkeypatch):
    """GREEN (plan T007): the dedicated sidecar ``GET /mcp/servers/{server_name}/tools``
    route exists and returns ONLY the requested server's tools plus scope +
    catalog version, mirroring the canonical contract."""
    client = _sidecar_client(monkeypatch)
    resp = client.get(
        f"/mcp/servers/{_BRAVE}/tools",
        headers={"Authorization": "Bearer local-session-token"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    servers = {_server_of(t) for t in data["tools"]}
    assert servers == {_BRAVE}, (
        f"sidecar dedicated route leaked other servers: "
        f"{[t.get('name') for t in data['tools']]}"
    )
    assert data["scope"]["kind"] == "server"
    assert data["scope"]["serverName"] == _BRAVE
    assert data["serversCount"] == 1
    assert str(data.get("catalogVersion", "")).startswith("sha256:")
