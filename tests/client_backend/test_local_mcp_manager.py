import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from client_backend.api import mcp as mcp_api
from client_backend.core.auth import require_local_session
from client_backend.core.config import client_settings
from client_backend.core.paths import get_profile_subdir
from client_backend.core.security import LocalSessionPayload
from client_backend.main import app
from client_backend.services import local_mcp_manager as local_mcp_manager_module
from client_backend.services.local_mcp_manager import LocalMCPManager, shutdown_mcp_manager


async def test_local_mcp_manager_creates_default_config(tmp_path):
    config_path = tmp_path / "mcp" / "mcp_config.json"
    manager = LocalMCPManager(config_path=config_path)

    await manager.initialize()

    assert manager.config_path == config_path.resolve()
    assert manager.config_path.exists()
    assert manager.config_path.read_text(encoding="utf-8").strip() == '{\n  "mcpServers": {}\n}'

    await manager.shutdown()


async def test_local_mcp_manager_loads_real_fastmcp_stdio_server(tmp_path):
    config_path = tmp_path / "mcp" / "mcp_config.json"
    server_script = Path("app/ai/mcp_servers/time_server.py").resolve()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "time-server": {
                        "command": sys.executable,
                        "args": [str(server_script)],
                        "transport": "stdio",
                        "enabled": True,
                    }
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    manager = LocalMCPManager(config_path=config_path)
    await manager.initialize()

    tools = manager.get_all_tools()
    assert any(
        tool.server_name == "time-server" and tool.name == "get_current_time" for tool in tools
    )

    result = await manager.call_tool(
        "time-server::get_current_time",
        {"timezone": "UTC", "format": "%Y-%m-%d"},
        timeout=30,
    )
    serialized_result = json.dumps(result) if not isinstance(result, str) else result
    assert "UTC" in serialized_result
    assert "timezone" in serialized_result

    await manager.shutdown()


async def test_local_mcp_manager_reloads_user_scoped_profile_config(tmp_path, monkeypatch):
    original_mcp_config_path = client_settings.mcp_config_path
    original_profile_root = client_settings.profile_root
    client_settings.mcp_config_path = ""
    client_settings.profile_root = str(tmp_path / "profiles")

    auth_state = SimpleNamespace(current_user_id="user-a")
    monkeypatch.setattr(
        local_mcp_manager_module,
        "get_upstream_auth_service",
        lambda: SimpleNamespace(get_current_user_id=lambda: auth_state.current_user_id),
    )

    user_a_path = get_profile_subdir("user-a", "mcp") / "mcp_config.json"
    user_b_path = get_profile_subdir("user-b", "mcp") / "mcp_config.json"
    user_a_path.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")
    user_b_path.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")

    manager = LocalMCPManager()
    try:
        await manager.initialize()
        assert manager.config_path == user_a_path.resolve()

        auth_state.current_user_id = "user-b"
        await manager.initialize()
        assert manager.config_path == user_b_path.resolve()
    finally:
        await manager.shutdown()
        client_settings.mcp_config_path = original_mcp_config_path
        client_settings.profile_root = original_profile_root


def test_parse_server_url_payload_normalizes_http_transport():
    name, config = mcp_api._parse_server_url_payload({"url": "https://example.com/mcp"})

    assert name == "mcp"
    assert config["transport"] == "streamable_http"
    assert config["url"] == "https://example.com/mcp"


def test_local_mcp_manager_preserves_scoped_npm_package_args(tmp_path):
    config_path = tmp_path / "mcp" / "mcp_config.json"
    manager = LocalMCPManager(config_path=config_path)

    resolved = manager._resolve_config_relative_value(
        "@wonderwhy-er/desktop-commander@latest",
        config_dir=tmp_path,
    )

    assert resolved == "@wonderwhy-er/desktop-commander@latest"


def test_tool_records_from_loaded_tools_do_not_mutate_foreign_schemas(tmp_path):
    manager = LocalMCPManager(config_path=tmp_path / "mcp" / "mcp_config.json")
    raw_schema = {
        "$schema": "https://example.com/schema.json",
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["safe", 1],
            }
        },
    }
    tool = SimpleNamespace(
        name="default_api:inspect",
        description="Inspect",
        args_schema=raw_schema,
    )

    records = manager._tool_records_from_loaded_tools("demo-server", [tool])

    assert tool.args_schema is raw_schema
    assert tool.args_schema["$schema"] == "https://example.com/schema.json"
    assert records[0].name == "inspect"
    assert "$schema" not in records[0].input_schema
    assert "enum" not in records[0].input_schema["properties"]["mode"]


def test_add_mcp_server_endpoint_loads_tools_after_save(tmp_path):
    config_path = tmp_path / "mcp" / "mcp_config.json"
    original_mcp_config_path = client_settings.mcp_config_path
    original_profile_root = client_settings.profile_root
    now = datetime.now(timezone.utc)

    client_settings.mcp_config_path = str(config_path)
    client_settings.profile_root = str(tmp_path / "profiles")
    asyncio.run(shutdown_mcp_manager())

    fake_session = LocalSessionPayload(
        user_id="user-123",
        server_user_id="user-123",
        device_id=None,
        device_identifier="device-abc",
        iat=now,
        exp=now + timedelta(hours=1),
    )
    app.dependency_overrides[require_local_session] = lambda: fake_session

    server_script = Path("app/ai/mcp_servers/time_server.py").resolve()

    try:
        with TestClient(app) as client:
            save_response = client.post(
                "/api/mcp/servers",
                json={
                    "name": "time-server",
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [str(server_script)],
                    "enabled": True,
                },
            )
            assert save_response.status_code == 201, save_response.text

            tools_response = client.get("/api/mcp/tools")
            assert tools_response.status_code == 200, tools_response.text
            payload = tools_response.json()
            assert payload["success"] is True, payload
            assert any(
                tool["serverName"] == "time-server" and tool["name"] == "get_current_time"
                for tool in payload["data"]["tools"]
            )
    finally:
        app.dependency_overrides.clear()
        asyncio.run(shutdown_mcp_manager())
        local_mcp_manager_module._mcp_manager = None
        client_settings.mcp_config_path = original_mcp_config_path
        client_settings.profile_root = original_profile_root
