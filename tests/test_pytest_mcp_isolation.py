"""Contracts preventing pytest from touching live device MCP profiles."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from client_backend.core.config import ClientSettings
from client_backend.schemas.mcp_config import MCPProfileScope
from client_backend.services.mcp_config_store import MCPConfigStore


def test_pytest_profile_root_and_mcp_profiles_are_device_isolated() -> None:
    configured_root = Path(os.environ["CLIENT_PROFILE_ROOT"]).resolve()
    assert configured_root.parent == Path(tempfile.gettempdir()).resolve()
    assert configured_root.name.startswith("sample-chatbot-pytest-")

    settings = ClientSettings()
    first = MCPConfigStore(
        MCPProfileScope(user_id="user-1", device_identifier="device-a"),
        profile_root=Path(settings.profile_root),
    )
    second = MCPConfigStore(
        MCPProfileScope(user_id="user-1", device_identifier="device-b"),
        profile_root=Path(settings.profile_root),
    )

    assert first.profile_path != second.profile_path
    assert configured_root in first.profile_path.parents
    assert configured_root in second.profile_path.parents
