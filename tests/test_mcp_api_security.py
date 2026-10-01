"""Security boundaries over the real router, bearer auth, service and manager.

Only the users-table read and external MCP transport are replaced. Configuration
persistence, session lifecycle, JWT checks, tools and HTTP responses remain real.
"""

import json
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID

import pytest
from dependency_injector import providers
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.tools import StructuredTool
from pydantic import ValidationError

from app.ai import mcp_integration
from app.api.mcp import router
from app.core import auth
from app.core.config import Settings, get_settings
from app.core.container import Container, setup_auto_injection
from app.services.jwt_service import JwtService
from app.services.mcp_service import MCPService
from app.utils.exception_handler import register_exception_handlers
from tests.token_state_stub import stub_token_states

ADMIN = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
MEMBER = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")


@pytest.fixture
def harness(monkeypatch, tmp_path):
    monkeypatch.delenv("ADMIN_USER_IDS", raising=False)
    monkeypatch.delenv("MCP_ALLOW_API_STDIO", raising=False)
    stub_token_states(monkeypatch)
    now = datetime.now(UTC)

    def lookup(user_id):
        return SimpleNamespace(
            id=user_id,
            username="tester",
            email="tester@example.invalid",
            created_at=now,
            updated_at=now,
            deleted_at=None,
            avatar_url=None,
        )

    monkeypatch.setattr(auth, "get_user_service", lambda: SimpleNamespace(get_by_id=lookup))
    executed = tmp_path / "tool-executed"

    async def echo(value: str) -> str:
        executed.write_text(value)
        return value

    tool = StructuredTool.from_function(coroutine=echo, name="echo", description="Echo input")

    class Transport:
        def __init__(self, connections):
            self.connections = connections

        @asynccontextmanager
        async def session(self, server_name):
            yield object()

    async def load_tools(session):
        return [tool]

    monkeypatch.setattr(mcp_integration, "MultiServerMCPClient", Transport)
    monkeypatch.setattr(mcp_integration, "load_mcp_tools", load_tools)

    @contextmanager
    def open_client(*, allowlist=(), allow_stdio=False, active=False, defaults=False):
        config_path = tmp_path / "mcp.json"
        config_path.write_text(
            json.dumps(
                {
                    "servers": {
                        "remote": {
                            "transport": "http",
                            "url": "https://mcp.example.invalid/mcp?key=secret",
                            "headers": {"Authorization": "secret"},
                            "env": {"TOKEN": "secret"},
                            "enabled": active,
                        },
                        "dormant": {"transport": "stdio", "command": "python", "enabled": False},
                    }
                }
            )
        )
        manager = mcp_integration.MCPManager(str(config_path))
        manager.config = manager._load_config()
        service = MCPService(manager, allow_api_stdio=True) if allow_stdio else MCPService(manager)
        config = Settings(_env_file=None)
        if not defaults:
            config = config.model_copy(update={"admin_user_ids": list(allowlist)})
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_settings] = lambda: config
        register_exception_handlers(app)
        setup_auto_injection(Container)
        jwt_service = JwtService()
        headers = {
            user: {"Authorization": "Bearer " + jwt_service.create_access_token({"sub": str(user)})}
            for user in (ADMIN, MEMBER)
        }
        with Container.mcp_service.override(providers.Object(service)), TestClient(app) as client:
            try:
                yield SimpleNamespace(
                    client=client,
                    manager=manager,
                    service=service,
                    path=config_path,
                    headers=headers,
                    executed=executed,
                )
            finally:
                client.portal.call(manager.cleanup)
        setup_auto_injection(Container)

    return open_client


MUTATIONS = [
    (
        "POST",
        "/mcp/servers",
        {
            "name": "added",
            "transport": "http",
            "url": "https://mcp.example.invalid/mcp",
            "enabled": False,
        },
    ),
    (
        "POST",
        "/mcp/servers/from-url",
        {"name": "added", "url": "https://mcp.example.invalid/mcp", "enabled": False},
    ),
    ("DELETE", "/mcp/servers/remote", None),
    ("PATCH", "/mcp/servers/remote/toggle?enabled=true", None),
    ("POST", "/mcp/tools/echo/execute", {"arguments": {"value": "ran"}, "serverName": "remote"}),
]


@pytest.mark.parametrize("method,path,body", MUTATIONS)
@pytest.mark.parametrize("allowlist", [(), (ADMIN,)])
def test_all_mutations_refuse_non_admin_without_persisting_or_executing(
    harness, method, path, body, allowlist
):
    with harness(allowlist=allowlist, active=True) as h:
        before = h.path.read_bytes()
        response = h.client.request(method, path, json=body, headers=h.headers[MEMBER])
        assert response.status_code == 403, response.text
        assert response.json()["code"] == "ADMIN_REQUIRED"
        assert h.path.read_bytes() == before
        assert not h.executed.exists()
        assert h.manager.client is None


@pytest.mark.parametrize("method,path,body", MUTATIONS)
def test_mutations_require_bearer_authentication(harness, method, path, body):
    with harness(allowlist=(ADMIN,)) as h:
        response = h.client.request(method, path, json=body)
        assert response.status_code in (401, 403)
        assert not h.executed.exists()


def test_default_allowlist_grants_nobody_admin_rights(harness):
    config = Settings(_env_file=None)
    assert config.admin_user_ids == []
    assert config.mcp_allow_api_stdio is False
    with harness(defaults=True) as h:
        response = h.client.delete("/mcp/servers/remote", headers=h.headers[ADMIN])
        assert response.status_code == 403, response.text
        assert "remote" in json.loads(h.path.read_text())["servers"]


@pytest.mark.parametrize("method,path,body", MUTATIONS)
def test_explicit_admin_can_mutate_http_servers_and_test_tools(harness, method, path, body):
    with harness(allowlist=(ADMIN,), active=True) as h:
        response = h.client.request(method, path, json=body, headers=h.headers[ADMIN])
        assert response.status_code == (
            201 if path in ("/mcp/servers", "/mcp/servers/from-url") else 200
        ), response.text
        saved = json.loads(h.path.read_text())["servers"]
        if path == "/mcp/servers/remote":
            assert "remote" not in saved
        elif "toggle" in path:
            assert saved["remote"]["enabledByDefault"] is True
        elif "execute" in path:
            assert h.executed.read_text() == "ran"
            assert response.json()["data"]["result"] == "ran"
        else:
            assert saved["added"]["transport"] in ("http", "streamable_http")


STDIO_REQUESTS = [
    (
        "/mcp/servers",
        {"name": "added", "transport": "stdio", "command": "python", "enabled": False},
    ),
    ("/mcp/servers/from-url", {"name": "added", "url": "npx -y example-package", "enabled": False}),
    ("/mcp/servers/from-url", {"url": "  npx example-package --verbose  ", "enabled": False}),
]


@pytest.mark.parametrize("path,body", STDIO_REQUESTS)
@pytest.mark.parametrize("enabled", [False, True])
def test_stdio_default_refusal_survives_url_parser_and_does_not_persist(
    harness, path, body, enabled
):
    with harness(allowlist=(ADMIN,)) as h:
        before = h.path.read_bytes()
        response = h.client.post(path, json={**body, "enabled": enabled}, headers=h.headers[ADMIN])
        assert response.status_code == 403, response.text
        assert response.json()["code"] == "API_STDIO_DISABLED"
        assert h.path.read_bytes() == before
        assert h.manager.client is None


@pytest.mark.parametrize("path,body", STDIO_REQUESTS)
def test_explicit_stdio_opt_in_allows_admin_command_configuration(harness, path, body):
    with harness(allowlist=(ADMIN,), allow_stdio=True) as h:
        response = h.client.post(path, json=body, headers=h.headers[ADMIN])
        assert response.status_code == 201, response.text
        saved = json.loads(h.path.read_text())["servers"]
        assert any(
            name != "dormant" and config["transport"] == "stdio" for name, config in saved.items()
        )


@pytest.mark.parametrize("transport", ["stdio", "STDIO", " stdio ", "omitted", None, ""])
def test_api_cannot_enable_preconfigured_stdio_without_opt_in(harness, transport):
    with harness(allowlist=(ADMIN,)) as h:
        config = json.loads(h.path.read_text())
        if transport == "omitted":
            config["servers"]["dormant"].pop("transport")
        else:
            config["servers"]["dormant"]["transport"] = transport
        h.path.write_text(json.dumps(config))
        h.manager.config = config
        before = h.path.read_bytes()
        response = h.client.patch(
            "/mcp/servers/dormant/toggle?enabled=true", headers=h.headers[ADMIN]
        )
        assert response.status_code == 403, response.text
        assert response.json()["code"] == "API_STDIO_DISABLED"
        assert h.path.read_bytes() == before
        assert h.manager.client is None


def test_stdio_opt_in_allows_enabling_preconfigured_stdio(harness):
    with harness(allowlist=(ADMIN,), allow_stdio=True) as h:
        response = h.client.patch(
            "/mcp/servers/dormant/toggle?enabled=true", headers=h.headers[ADMIN]
        )
        assert response.status_code == 200, response.text
        assert json.loads(h.path.read_text())["servers"]["dormant"]["enabledByDefault"] is True
        assert h.manager.client is not None


def test_disabling_preconfigured_stdio_is_allowed_without_stdio_opt_in(harness):
    with harness(allowlist=(ADMIN,)) as h:
        response = h.client.patch(
            "/mcp/servers/dormant/toggle?enabled=false", headers=h.headers[ADMIN]
        )
        assert response.status_code == 200, response.text
        assert json.loads(h.path.read_text())["servers"]["dormant"]["enabledByDefault"] is False


@pytest.mark.parametrize(
    "path",
    [
        "/mcp/servers",
        "/mcp/servers/remote",
        "/mcp/tools",
        "/mcp/servers/remote/tools",
        "/mcp/tools/echo",
    ],
)
def test_catalog_reads_are_authenticated_and_available_to_non_admins(harness, path):
    with harness(active=True) as h:
        assert h.client.get(path).status_code in (401, 403)
        response = h.client.get(path, headers=h.headers[MEMBER])
        assert response.status_code == 200, response.text
        assert "secret" not in response.text
        assert "secret" in h.path.read_text()
        if path.startswith("/mcp/servers") and not path.endswith("/tools"):
            assert "***" in response.text


def test_settings_parse_explicit_uuid_allowlist_and_stdio_opt_in(monkeypatch):
    monkeypatch.setenv("ADMIN_USER_IDS", json.dumps([str(ADMIN)]))
    monkeypatch.setenv("MCP_ALLOW_API_STDIO", "true")
    config = Settings(_env_file=None)
    assert getattr(config, "admin_user_ids", None) == [ADMIN]
    assert getattr(config, "mcp_allow_api_stdio", None) is True


def test_settings_refuse_malformed_admin_identity(monkeypatch):
    monkeypatch.setenv("ADMIN_USER_IDS", '["not-a-uuid"]')
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.asyncio
async def test_bundled_manager_stdio_initialization_does_not_need_api_opt_in(tmp_path):
    path = tmp_path / "bundled.json"
    path.write_text(
        json.dumps(
            {
                "servers": {
                    "bundled": {"transport": "stdio", "command": "python", "enabled": True},
                }
            }
        )
    )
    manager = mcp_integration.MCPManager(str(path))
    await manager.initialize()
    assert manager.client is not None
    await manager.cleanup()
