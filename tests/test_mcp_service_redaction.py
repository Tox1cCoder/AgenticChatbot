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


_HTTP_URL = "https://mcp.example.com/v1/mcp?api_key=sk-live-secret&region=eu&flag"


class _HttpManager(_Manager):
    config = {"servers": {"search": {"transport": "http", "url": _HTTP_URL}}}

    def get_servers_status(self):
        return {"search": {"enabled": True, "tool_count": 1, "transport": "http"}}

    def get_server_info(self, server_name):
        config = dict(self.config["servers"]["search"])
        return {**super().get_server_info(server_name), "config": config}


@pytest.mark.asyncio
async def test_an_http_server_url_masks_query_values_but_keeps_their_names():
    service = MCPService(_HttpManager())

    listed = (await service.list_servers())["servers"][0]["config"]["url"]
    detailed = (await service.get_server_details("search"))["config"]["url"]

    expected = "https://mcp.example.com/v1/mcp?api_key=***&region=***&***"
    assert listed == expected
    assert detailed == expected
    assert _HttpManager.config["servers"]["search"]["url"] == _HTTP_URL


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://mcp.example.com/mcp", "https://mcp.example.com/mcp"),
        ("https://user:hunter2@mcp.example.com/mcp", "https://user:***@mcp.example.com/mcp"),
        ("https://mcp.example.com/mcp?token=abc#frag", "https://mcp.example.com/mcp?token=***#frag"),
        ("https://[::1/mcp?token=abc", "***"),
    ],
    ids=["no_query", "userinfo_password", "fragment_kept", "unparseable"],
)
def test_url_redaction_edges(url, expected):
    from app.services.mcp_service import redact_server_config

    assert redact_server_config({"url": url})["url"] == expected
