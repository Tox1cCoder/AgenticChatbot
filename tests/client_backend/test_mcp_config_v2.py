from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from client_backend.core import paths as profile_paths
from client_backend.schemas.mcp_config import (
    MCPProfileDocument,
    MCPProfileScope,
    MCPRegistryDocument,
)
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
