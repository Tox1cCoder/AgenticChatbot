from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from client_backend.core import paths as profile_paths
from client_backend.core.security import decrypt_local_secret
from client_backend.schemas.mcp_config import (
    MCPProfileDocument,
    MCPProfileScope,
    MCPRegistryDocument,
)
from client_backend.services.local_mcp_manager import LocalMCPManager
from client_backend.services.mcp_config_migration import (
    MCPConfigMigrationConflictError,
    migrate_legacy_mcp_profile,
    prepare_mcp_config_store,
)
from client_backend.services.mcp_config_store import MCPConfigConflictError, MCPConfigStore
from client_backend.services.mcp_secret_store import MCPSecretStore


def test_mcp_profile_rejects_unknown_schema_version():
    with pytest.raises(ValidationError):
        MCPProfileDocument.model_validate(
            {
                "schemaVersion": 1,
                "bundledOverrides": {},
                "customServers": {},
            }
        )


def test_mcp_profile_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        MCPProfileDocument.model_validate(
            {
                "schemaVersion": 2,
                "bundledOverrides": {},
                "customServers": {},
                "legacy": True,
            }
        )


@pytest.mark.parametrize(
    "server",
    [
        {"transport": "stdio", "args": []},
        {"transport": "streamable_http", "enabled": True},
    ],
)
def test_mcp_registry_requires_transport_endpoint(server):
    with pytest.raises(ValidationError):
        MCPRegistryDocument.model_validate(
            {
                "schemaVersion": 2,
                "servers": {"broken": server},
            }
        )


def test_device_profile_paths_are_isolated_for_same_user(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_paths.client_settings, "profile_root", str(tmp_path))
    monkeypatch.setattr(
        profile_paths.client_settings,
        "server_api_base_url",
        "https://server.example",
    )

    first = profile_paths.get_device_profile_subdir("user-1", "device-a", "mcp")
    second = profile_paths.get_device_profile_subdir("user-1", "device-b", "mcp")

    assert first != second
    assert first == (
        Path(tmp_path)
        / hashlib.sha256(b"https://server.example").hexdigest()[:12]
        / "user-1"
        / "devices"
        / "device-a"
        / "mcp"
    )
    assert second.name == "mcp"


@pytest.mark.parametrize("unsafe", ["../other", "a/b", "a\\b", "", ".", ".."])
def test_device_profile_path_rejects_unsafe_components(tmp_path, monkeypatch, unsafe):
    monkeypatch.setattr(profile_paths.client_settings, "profile_root", str(tmp_path))

    with pytest.raises(ValueError):
        profile_paths.get_device_profile_subdir("user-1", unsafe, "mcp")


def test_profile_scope_rejects_empty_identity():
    with pytest.raises(ValidationError):
        MCPProfileScope(user_id="", device_identifier="device-a")


def test_mcp_credentials_are_encrypted_and_device_isolated(tmp_path):
    first = MCPSecretStore(
        MCPProfileScope(user_id="user-1", device_identifier="device-a"),
        profile_root=tmp_path,
    )
    second = MCPSecretStore(
        MCPProfileScope(user_id="user-1", device_identifier="device-b"),
        profile_root=tmp_path,
    )

    first.set_for_server(
        "notion",
        env={"NOTION_TOKEN": "secret-token-value"},
        headers={"Authorization": "Bearer secret-token-value"},
    )

    credentials = first.get_for_server("notion")
    assert credentials.env == {"NOTION_TOKEN": "secret-token-value"}
    assert credentials.headers == {"Authorization": "Bearer secret-token-value"}
    assert first.list_for_server("notion") == {
        "envKeys": ["NOTION_TOKEN"],
        "headerKeys": ["Authorization"],
    }
    assert second.get_for_server("notion").env == {}
    assert second.get_for_server("notion").headers == {}
    assert first.path != second.path
    assert "secret-token-value" not in first.path.read_text(encoding="utf-8")


def test_mcp_credentials_delete_server_binding(tmp_path):
    store = MCPSecretStore(
        MCPProfileScope(user_id="user-1", device_identifier="device-a"),
        profile_root=tmp_path,
    )
    store.set_for_server("demo", env={"API_TOKEN": "value"}, headers={})

    assert store.delete_server("demo") is True
    assert store.delete_server("demo") is False
    assert store.get_for_server("demo").env == {}


def test_mcp_credentials_reject_invalid_environment_name(tmp_path):
    store = MCPSecretStore(
        MCPProfileScope(user_id="user-1", device_identifier="device-a"),
        profile_root=tmp_path,
    )

    with pytest.raises(ValueError):
        store.set_for_server("demo", env={"INVALID-NAME": "value"}, headers={})


def _registry(tmp_path: Path) -> tuple[Path, Path]:
    app_root = tmp_path / "application"
    script = app_root / "app" / "ai" / "mcp_servers" / "time_server.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("print('ok')", encoding="utf-8")
    path = app_root / "app" / "ai" / "mcp_config.json"
    path.write_text(
        json.dumps(
            {
                "schemaVersion": 2,
                "servers": {
                    "time": {
                        "transport": "stdio",
                        "command": "python",
                        "args": ["app/ai/mcp_servers/time_server.py"],
                        "enabledByDefault": True,
                        "description": "Time",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return app_root, path


def _store(tmp_path: Path, device: str) -> MCPConfigStore:
    app_root, registry_path = _registry(tmp_path)
    return MCPConfigStore(
        MCPProfileScope(user_id="user-1", device_identifier=device),
        registry_path=registry_path,
        application_root=app_root,
        profile_root=tmp_path / "profiles",
    )


def test_mcp_config_store_resolves_bundled_server_from_application_root(tmp_path):
    store = _store(tmp_path, "device-a")
    app_root = tmp_path / "application"

    servers = {server.name: server for server in store.list_effective_servers()}

    assert servers["time"].source == "bundled"
    assert servers["time"].command == sys.executable
    assert servers["time"].args == [str((app_root / "app/ai/mcp_servers/time_server.py").resolve())]
    assert servers["time"].cwd == str(app_root.resolve())


def test_mcp_config_store_isolates_same_user_device_mutations(tmp_path):
    first = _store(tmp_path, "device-a")
    second = _store(tmp_path, "device-b")

    first.save_custom_server(
        "custom",
        {
            "transport": "stdio",
            "command": "runner",
            "args": ["tool.py"],
            "enabled": True,
        },
        env={"API_TOKEN": "device-a-token"},
        headers={},
    )
    first.set_enabled("time", False)

    assert {server.name for server in first.list_effective_servers()} == {"time", "custom"}
    assert {server.name for server in second.list_effective_servers()} == {"time"}
    first_time = next(server for server in first.list_effective_servers() if server.name == "time")
    second_time = next(
        server for server in second.list_effective_servers() if server.name == "time"
    )
    assert first_time.enabled is False
    assert second_time.enabled is True


def test_mcp_config_store_rejects_custom_bundled_name_collision(tmp_path):
    store = _store(tmp_path, "device-a")

    with pytest.raises(MCPConfigConflictError):
        store.save_custom_server(
            "time",
            {"transport": "stdio", "command": "other", "args": []},
            env={},
            headers={},
        )


def test_mcp_config_store_delete_disables_bundled_and_removes_custom(tmp_path):
    store = _store(tmp_path, "device-a")
    store.save_custom_server(
        "custom",
        {"transport": "stdio", "command": "runner"},
        env={"API_TOKEN": "secret"},
        headers={},
    )

    assert store.delete_server("time") == "disabled_bundled"
    assert store.delete_server("custom") == "deleted_custom"
    assert (
        next(server for server in store.list_effective_servers() if server.name == "time").enabled
        is False
    )
    assert "custom" not in {server.name for server in store.list_effective_servers()}
    assert store.secret_store.get_for_server("custom").env == {}


def test_mcp_config_store_rolls_back_credentials_when_profile_write_fails(
    tmp_path,
    monkeypatch,
):
    store = _store(tmp_path, "device-a")
    store.save_custom_server(
        "custom",
        {"transport": "stdio", "command": "runner"},
        env={"API_TOKEN": "original"},
        headers={},
    )

    def fail_write(_profile):
        raise OSError("simulated profile write failure")

    monkeypatch.setattr(store, "_write_profile", fail_write)

    with pytest.raises(OSError, match="simulated"):
        store.save_custom_server(
            "custom",
            {"transport": "stdio", "command": "replacement"},
            env={"API_TOKEN": "replacement"},
            headers={},
        )

    assert store.secret_store.get_for_server("custom").env == {"API_TOKEN": "original"}


def test_mcp_migration_preserves_custom_server_and_encrypts_credentials(tmp_path):
    store = _store(tmp_path, "device-a")
    legacy_path = tmp_path / "legacy" / "mcp_config.json"
    legacy_path.parent.mkdir()
    legacy_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "time": {
                        "transport": "stdio",
                        "command": "python",
                        "args": ["app/ai/mcp_servers/time_server.py"],
                        "enabled": False,
                        "description": "Time",
                    },
                    "notion": {
                        "transport": "stdio",
                        "command": "npx",
                        "args": ["-y", "@notionhq/notion-mcp-server"],
                        "env": {"NOTION_TOKEN": "migration-secret"},
                        "enabled": True,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    result = migrate_legacy_mcp_profile(
        store.scope,
        legacy_path=legacy_path,
        store=store,
    )

    assert result.status == "migrated"
    assert result.migrated_servers == ("notion", "time")
    assert result.backup_path is not None and result.backup_path.is_file()
    assert "migration-secret" not in result.backup_path.read_text(encoding="utf-8")
    backup_envelope = json.loads(result.backup_path.read_text(encoding="utf-8"))
    assert decrypt_local_secret(backup_envelope) == legacy_path.read_bytes()
    assert result.receipt_path is not None and result.receipt_path.is_file()
    profile = store.load_profile()
    assert profile.bundled_overrides["time"].enabled is False
    assert "notion" in profile.custom_servers
    assert store.secret_store.get_for_server("notion").env == {"NOTION_TOKEN": "migration-secret"}
    assert "migration-secret" not in result.receipt_path.read_text(encoding="utf-8")
    assert legacy_path.is_file()

    repeated = migrate_legacy_mcp_profile(
        store.scope,
        legacy_path=legacy_path,
        store=store,
    )
    assert repeated.status == "already_v2"


def test_mcp_migration_stops_on_modified_bundled_name_collision(tmp_path):
    store = _store(tmp_path, "device-a")
    legacy_path = tmp_path / "legacy.json"
    original = {
        "mcp_servers": {
            "time": {
                "transport": "stdio",
                "command": "custom-time",
                "args": [],
                "enabled": True,
            }
        }
    }
    legacy_path.write_text(json.dumps(original), encoding="utf-8")

    with pytest.raises(MCPConfigMigrationConflictError):
        migrate_legacy_mcp_profile(
            store.scope,
            legacy_path=legacy_path,
            store=store,
        )

    assert json.loads(legacy_path.read_text(encoding="utf-8")) == original
    assert not store.profile_path.exists()
    assert not store.secret_store.path.exists()


def test_mcp_migration_canonical_key_wins_dual_key_collision(tmp_path):
    store = _store(tmp_path, "device-a")
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(
        json.dumps(
            {
                "mcp_servers": {
                    "custom": {
                        "transport": "stdio",
                        "command": "legacy-command",
                    }
                },
                "mcpServers": {
                    "custom": {
                        "transport": "stdio",
                        "command": "canonical-command",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    migrate_legacy_mcp_profile(
        store.scope,
        legacy_path=legacy_path,
        store=store,
    )

    assert store.load_profile().custom_servers["custom"].command == "canonical-command"


def test_prepare_store_migrates_legacy_before_first_profile_load(tmp_path):
    store = _store(tmp_path, "device-a")
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "custom": {
                        "transport": "stdio",
                        "command": "runner",
                        "env": {"API_TOKEN": "secret"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    prepared, result = prepare_mcp_config_store(
        store.scope,
        store=store,
        legacy_path=legacy_path,
    )

    assert prepared is store
    assert result.status == "migrated"
    assert "custom" in prepared.load_profile().custom_servers
    assert prepared.secret_store.get_for_server("custom").env == {"API_TOKEN": "secret"}


@pytest.mark.asyncio
async def test_v2_manager_discovers_real_bundled_time_server(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    registry_path = tmp_path / "registry.json"
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
                        "description": "Time",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    store = MCPConfigStore(
        MCPProfileScope(user_id="user-1", device_identifier="device-a"),
        registry_path=registry_path,
        application_root=repo_root,
        profile_root=tmp_path / "profiles",
    )
    manager = LocalMCPManager(store=store)

    try:
        await manager.initialize()
        assert manager.servers["time"].is_running()
        assert {tool.name for tool in manager.get_tools_by_server("time")} == {"get_current_time"}
    finally:
        await manager.shutdown()


def test_v2_manager_catalogs_are_device_scoped(tmp_path):
    first = LocalMCPManager(store=_store(tmp_path, "device-a"))
    second = LocalMCPManager(store=_store(tmp_path, "device-b"))

    assert first.scope.device_identifier == "device-a"
    assert second.scope.device_identifier == "device-b"
    assert first.store.profile_path != second.store.profile_path
