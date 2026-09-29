"""Server listings go to every signed-in user, so credentials must not."""

from __future__ import annotations

import pytest

from app.services.mcp_service import MCPService

_CONFIG = {
    "transport": "stdio",
    "command": "npx",
    "env": {"API_TOKEN": "sk-live-secret"},
    "headers": {"Authorization": "Bearer secret-token"},
}


class _Manager:
    config = {"servers": {"tools": _CONFIG}}

    def get_servers_status(self):
        return {"tools": {"enabled": True, "tool_count": 2, "transport": "stdio"}}

    def get_server_info(self, server_name):
        return {
            "name": server_name,
            "enabled": True,
            "tool_count": 2,
            "config": _CONFIG,
            "description": "",
        }


@pytest.mark.asyncio
async def test_listing_masks_env_and_header_values_but_keeps_their_names():
    result = await MCPService(_Manager()).list_servers()

    config = result["servers"][0]["config"]
    assert config["env"] == {"API_TOKEN": "***"}
    assert config["headers"] == {"Authorization": "***"}
    assert config["command"] == "npx"
    assert "secret" not in str(result)


@pytest.mark.asyncio
async def test_server_details_mask_credentials_without_touching_the_live_config():
    details = await MCPService(_Manager()).get_server_details("tools")

    assert "secret" not in str(details)
    assert _CONFIG["env"] == {"API_TOKEN": "sk-live-secret"}
