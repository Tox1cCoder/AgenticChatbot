from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from client_backend.api import mcp as mcp_api
from client_backend.schemas.mcp_config import MCPProfileScope
from client_backend.services.local_mcp_manager import LocalMCPManager
from client_backend.services.mcp_config_store import MCPConfigStore


def _registry(tmp_path: Path) -> tuple[Path, Path]:
    application_root = tmp_path / "application"
    registry_path = application_root / "app" / "ai" / "mcp_config.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            {
                "schemaVersion": 2,
                "servers": {
                    "time": {
                        "transport": "stdio",
                        "command": "python",
                        "args": ["app/ai/mcp_servers/time_server.py"],
                        "enabledByDefault": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return application_root, registry_path


def _store(tmp_path: Path, device_identifier: str) -> MCPConfigStore:
    application_root, registry_path = _registry(tmp_path)
    return MCPConfigStore(
        MCPProfileScope(
            user_id="user-1",
            device_identifier=device_identifier,
        ),
        registry_path=registry_path,
        application_root=application_root,
        profile_root=tmp_path / "profiles",
    )


def _session(device_identifier: str):
    return SimpleNamespace(
        user_id="user-1",
        device_identifier=device_identifier,
    )


def test_tool_records_do_not_mutate_foreign_schemas(tmp_path):
    manager = LocalMCPManager(store=_store(tmp_path, "device-a"))
    raw_schema = {
        "$schema": "https://example.com/schema.json",
        "type": "object",
        "properties": {"mode": {"type": "string", "enum": ["safe", 1]}},
    }
    tool = SimpleNamespace(
        name="default_api:inspect",
        description="Inspect",
        args_schema=raw_schema,
    )

    records = manager._tool_records_from_loaded_tools("demo", [tool])

    assert tool.args_schema is raw_schema
    assert records[0].name == "inspect"
    assert "$schema" not in records[0].input_schema
    assert "enum" not in records[0].input_schema["properties"]["mode"]


def test_parse_server_url_payload_normalizes_http_transport():
    name, config = mcp_api._parse_server_url_payload({"url": "https://example.com/mcp"})

    assert name == "mcp"
    assert config["transport"] == "streamable_http"


@pytest.mark.asyncio
async def test_device_scoped_api_mutation_does_not_cross_same_user_devices(
    tmp_path,
    monkeypatch,
):
    stores = {
        "device-a": _store(tmp_path, "device-a"),
        "device-b": _store(tmp_path, "device-b"),
    }

    async def no_reload(_scope):
        return None

    async def no_refresh(_scope):
        return None

    monkeypatch.setattr(
        mcp_api,
        "_store",
        lambda session: stores[session.device_identifier],
    )
    monkeypatch.setattr(mcp_api, "_reload_manager", no_reload)
    monkeypatch.setattr(
        mcp_api,
        "_refresh_runtime_bridge_catalogs_if_connected",
        no_refresh,
    )

    await mcp_api.add_mcp_server(
        {
            "name": "private",
            "transport": "stdio",
            "command": "runner",
            "env": {"API_TOKEN": "device-a-secret"},
        },
        _session("device-a"),
    )

    assert "private" in {server.name for server in stores["device-a"].list_effective_servers()}
    assert "private" not in {server.name for server in stores["device-b"].list_effective_servers()}
    assert stores["device-b"].secret_store.get_for_server("private").env == {}


@pytest.mark.asyncio
async def test_device_scoped_api_rejects_reserved_bundled_name(tmp_path, monkeypatch):
    store = _store(tmp_path, "device-a")
    monkeypatch.setattr(mcp_api, "_store", lambda _session: store)

    with pytest.raises(HTTPException) as exc_info:
        await mcp_api.add_mcp_server(
            {
                "name": "time",
                "transport": "stdio",
                "command": "replacement",
            },
            _session("device-a"),
        )

    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_api_update_preserves_omitted_credentials_and_redacts_server_info(
    tmp_path,
    monkeypatch,
):
    store = _store(tmp_path, "device-a")
    store.save_custom_server(
        "private",
        {"transport": "stdio", "command": "runner"},
        env={"API_TOKEN": "never-return-this"},
        headers={},
    )

    async def no_reload(_scope):
        return None

    async def no_refresh(_scope):
        return None

    monkeypatch.setattr(mcp_api, "_store", lambda _session: store)
    monkeypatch.setattr(mcp_api, "_reload_manager", no_reload)
    monkeypatch.setattr(
        mcp_api,
        "_refresh_runtime_bridge_catalogs_if_connected",
        no_refresh,
    )

    await mcp_api.add_mcp_server(
        {
            "name": "private",
            "transport": "stdio",
            "command": "replacement",
        },
        _session("device-a"),
    )

    assert store.secret_store.get_for_server("private").env == {"API_TOKEN": "never-return-this"}
    server = next(item for item in store.list_effective_servers() if item.name == "private")
    response = mcp_api._server_info(
        server,
        SimpleNamespace(servers={}),
    )
    serialized = json.dumps(response)
    assert response["config"]["envKeys"] == ["API_TOKEN"]
    assert "never-return-this" not in serialized
