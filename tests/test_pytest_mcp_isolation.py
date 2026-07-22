"""Contracts for hermetic MCP configuration during pytest runs."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

from client_backend.core.config import ClientSettings
from client_backend.services.local_mcp_manager import LocalMCPManager


def test_pytest_bootstrap_uses_an_empty_mcp_config_outside_the_user_profile() -> None:
    configured_path = os.environ.get("CLIENT_MCP_CONFIG_PATH")
    assert configured_path

    config_path = Path(configured_path).resolve()
    assert config_path.parent.parent == Path(tempfile.gettempdir()).resolve()
    assert config_path.parent.name.startswith("sample-chatbot-pytest-")
    assert json.loads(config_path.read_text(encoding="utf-8")) == {"mcpServers": {}}

    settings = ClientSettings()
    assert Path(settings.mcp_config_path).resolve() == config_path

    manager = LocalMCPManager(config_path=config_path)
    assert asyncio.run(manager._load_config()) == []
