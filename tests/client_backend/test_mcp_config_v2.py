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
