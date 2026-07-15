"""
Local MCP management endpoints with the same response envelope as the server.
"""

from __future__ import annotations

import json
import shlex
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from client_backend.api.common import make_api_response
from client_backend.core.auth import require_local_session
from client_backend.core.security import LocalSessionPayload
from client_backend.services.local_mcp_manager import get_mcp_manager, shutdown_mcp_manager
from client_backend.services.runtime_bridge import get_runtime_bridge

router = APIRouter(prefix="/mcp", tags=["mcp"])


def _normalize_transport(transport: Any) -> str:
    normalized = str(transport or "stdio").strip().lower()
    if normalized == "http":
        return "streamable_http"
    return normalized


def _config_path() -> Path:
    manager = get_mcp_manager()
    return manager._resolve_config_path()


def _load_config_document() -> dict[str, Any]:
    path = _config_path()
    if not path.exists():
        return {"mcpServers": {}}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read MCP config: {exc}",
        ) from exc

    if not isinstance(payload, dict):
        return {"mcpServers": {}}
    if "mcpServers" not in payload or not isinstance(payload.get("mcpServers"), dict):
        payload["mcpServers"] = {}
    return payload


def _write_config_document(payload: dict[str, Any]) -> None:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


async def _reload_manager():
    await shutdown_mcp_manager()
    manager = get_mcp_manager()
    await manager.initialize()
    return manager


async def _refresh_runtime_bridge_catalogs_if_connected() -> None:
    bridge = get_runtime_bridge()
    if not bridge.is_connected() or not bridge.get_registered_device_id():
        return
    await bridge.refresh_catalogs()


def _manager_tool_lookup() -> dict[str, list[dict[str, Any]]]:
    manager = get_mcp_manager()
    lookup: dict[str, list[dict[str, Any]]] = {}
    for tool in manager.get_all_tools():
        lookup.setdefault(tool.name, []).append(
            {
                "name": tool.name,
                "description": tool.description,
                "argsSchema": tool.input_schema,
                "serverName": tool.server_name,
                "qualifiedId": tool.qualified_id,
            }
        )
    return lookup


def _resolve_manager_tool(
    tool_name: str,
    *,
    server_name: str | None = None,
    qualified_tool_id: str | None = None,
) -> dict[str, Any] | None:
    """Resolve a tool while preserving the legacy bare-name API.

    Older callers only sent ``tool_name`` and therefore keep the historical
    first-match behavior.  Tester callers can additionally identify the
    selected server (or send the qualified id directly), which prevents a
    same-name tool on another server from being invoked accidentally.
    """
    matches = _manager_tool_lookup().get(tool_name) or []
    normalized_qualified_id = str(qualified_tool_id or "").strip()
    normalized_server_name = str(server_name or "").strip()

    if normalized_qualified_id:
        return next(
            (
                tool
                for tool in matches
                if tool.get("qualifiedId") == normalized_qualified_id
                and (
                    not normalized_server_name
                    or tool.get("serverName") == normalized_server_name
                )
            ),
            None,
        )
    if normalized_server_name:
        return next(
            (
                tool
                for tool in matches
                if tool.get("serverName") == normalized_server_name
            ),
            None,
        )
    return matches[0] if matches else None


def _server_info_from_config(name: str, config: dict[str, Any]) -> dict[str, Any]:
    manager = get_mcp_manager()
    process = manager.servers.get(name)
    tool_count = len(process.tools) if process and process.is_running() else 0
    enabled = bool(config.get("enabled", True))

    return {
        "name": name,
        "transport": str(config.get("transport") or "stdio"),
        "enabled": enabled,
        "description": config.get("description") or "",
        "toolCount": tool_count,
        "config": {
            key: value
            for key, value in config.items()
            if key
            in {
                "transport",
                "command",
                "args",
                "env",
                "cwd",
                "url",
                "headers",
                "enabled",
                "description",
            }
        },
    }


def _parse_server_url_payload(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    raw_value = str(payload.get("url") or "").strip()
    if not raw_value:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="`url` is required",
        )

    name = str(payload.get("name") or "").strip()
    description = str(payload.get("description") or "").strip()
    enabled = bool(payload.get("enabled", True))

    if raw_value.startswith(("http://", "https://")):
        server_name = name or Path(raw_value.rstrip("/")).name or "mcp-server"
        return server_name, {
            "transport": "streamable_http",
            "url": raw_value,
            "enabled": enabled,
            "description": description,
        }

    parts = shlex.split(raw_value, posix=sys.platform != "win32")
    if not parts:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unable to parse MCP command string",
        )

    command, *args = parts
    server_name = name or Path(command).stem or "mcp-server"
    return server_name, {
        "transport": "stdio",
        "command": command,
        "args": args,
        "env": {},
        "enabled": enabled,
        "description": description,
    }


@router.get("/servers")
async def list_mcp_servers(
    _session: LocalSessionPayload = Depends(require_local_session),
):
    """List configured local MCP servers using the server's ApiResponse shape."""
    manager = get_mcp_manager()
    await manager.initialize()
    document = _load_config_document()
    server_configs = document.get("mcpServers", {})
    servers = [
        _server_info_from_config(name, config)
        for name, config in sorted(server_configs.items())
        if isinstance(config, dict)
    ]
    return make_api_response(
        success=True,
        message="MCP servers retrieved successfully",
        data={
            "servers": servers,
            "totalCount": len(servers),
            "enabledCount": sum(1 for server in servers if server["enabled"]),
        },
    )


@router.get("/servers/{server_name}")
async def get_mcp_server(
    server_name: str,
    _session: LocalSessionPayload = Depends(require_local_session),
):
    """Get details for a configured local MCP server."""
    manager = get_mcp_manager()
    await manager.initialize()
    document = _load_config_document()
    config = document.get("mcpServers", {}).get(server_name)
    if not isinstance(config, dict):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="MCP server not found")

    return make_api_response(
        success=True,
        message=f"Server '{server_name}' details retrieved successfully",
        data=_server_info_from_config(server_name, config),
    )


@router.post("/servers", status_code=status.HTTP_201_CREATED)
async def add_mcp_server(
    payload: dict[str, Any],
    _session: LocalSessionPayload = Depends(require_local_session),
):
    """Add or update a local MCP server configuration."""
    name = str(payload.get("name") or "").strip()
    if not name:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="`name` is required",
        )

    document = _load_config_document()
    server_configs = document.setdefault("mcpServers", {})
    server_configs[name] = {
        "transport": _normalize_transport(payload.get("transport")),
        "command": payload.get("command"),
        "args": payload.get("args") or [],
        "env": payload.get("env") or {},
        "cwd": payload.get("cwd"),
        "url": payload.get("url"),
        "headers": payload.get("headers") or {},
        "enabled": bool(payload.get("enabled", True)),
        "description": payload.get("description") or "",
    }
    _write_config_document(document)
    await _reload_manager()
    await _refresh_runtime_bridge_catalogs_if_connected()

    message = f"MCP server '{name}' saved successfully"
    return make_api_response(
        success=True,
        message=message,
        data={"message": message},
        status_code=status.HTTP_201_CREATED,
    )


@router.post("/servers/from-url", status_code=status.HTTP_201_CREATED)
async def add_mcp_server_from_url(
    payload: dict[str, Any],
    _session: LocalSessionPayload = Depends(require_local_session),
):
    """Add a local MCP server from a command string or HTTP URL."""
    name, config = _parse_server_url_payload(payload)
    document = _load_config_document()
    server_configs = document.setdefault("mcpServers", {})
    server_configs[name] = config
    _write_config_document(document)
    await _reload_manager()
    await _refresh_runtime_bridge_catalogs_if_connected()

    message = f"MCP server '{name}' saved successfully"
    return make_api_response(
        success=True,
        message=message,
        data={"message": message},
        status_code=status.HTTP_201_CREATED,
    )


@router.delete("/servers/{server_name}")
async def remove_mcp_server(
    server_name: str,
    _session: LocalSessionPayload = Depends(require_local_session),
):
    """Remove a local MCP server configuration."""
    document = _load_config_document()
    server_configs = document.setdefault("mcpServers", {})
    if server_name not in server_configs:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="MCP server not found")

    del server_configs[server_name]
    _write_config_document(document)
    await _reload_manager()
    await _refresh_runtime_bridge_catalogs_if_connected()

    message = f"MCP server '{server_name}' removed successfully"
    return make_api_response(
        success=True,
        message=message,
        data={"message": message},
    )


@router.patch("/servers/{server_name}/toggle")
async def toggle_mcp_server(
    server_name: str,
    enabled: bool = Query(..., description="True to enable, False to disable"),
    _session: LocalSessionPayload = Depends(require_local_session),
):
    """Enable or disable a local MCP server."""
    document = _load_config_document()
    server_configs = document.setdefault("mcpServers", {})
    config = server_configs.get(server_name)
    if not isinstance(config, dict):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="MCP server not found")

    config["enabled"] = enabled
    _write_config_document(document)
    await _reload_manager()
    await _refresh_runtime_bridge_catalogs_if_connected()

    state = "enabled" if enabled else "disabled"
    message = f"MCP server '{server_name}' {state} successfully"
    return make_api_response(
        success=True,
        message=message,
        data={"message": message},
    )


@router.get("/tools")
async def list_mcp_tools(
    server_name: str | None = Query(None, description="Filter by server name"),
    _session: LocalSessionPayload = Depends(require_local_session),
):
    """List local MCP tools in the server's response shape."""
    manager = get_mcp_manager()
    await manager.initialize()

    tools = []
    for tool in manager.get_all_tools():
        if server_name and tool.server_name != server_name:
            continue
        tools.append(
            {
                "name": tool.name,
                "description": tool.description,
                "argsSchema": tool.input_schema,
                "serverName": tool.server_name,
                "qualifiedId": tool.qualified_id,
            }
        )

    return make_api_response(
        success=True,
        message="MCP tools retrieved successfully",
        data={
            "tools": tools,
            "totalCount": len(tools),
            "serversCount": len({tool["serverName"] for tool in tools}),
        },
    )


@router.get("/tools/{tool_name}")
async def get_mcp_tool(
    tool_name: str,
    _session: LocalSessionPayload = Depends(require_local_session),
):
    """Get details for a local MCP tool."""
    manager = get_mcp_manager()
    await manager.initialize()
    matches = _manager_tool_lookup().get(tool_name) or []
    if not matches:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="MCP tool not found")

    tool = matches[0]
    return make_api_response(
        success=True,
        message=f"Tool '{tool_name}' details retrieved successfully",
        data={
            "name": tool["name"],
            "description": tool["description"],
            "argsSchema": tool["argsSchema"],
            "serverName": tool["serverName"],
        },
    )


@router.post("/tools/{tool_name}/execute")
async def execute_mcp_tool(
    tool_name: str,
    payload: dict[str, Any],
    _session: LocalSessionPayload = Depends(require_local_session),
):
    """Execute a local MCP tool for compatibility/testing."""
    manager = get_mcp_manager()
    await manager.initialize()
    server_name = payload.get("serverName") or payload.get("server_name")
    qualified_tool_id = payload.get("qualifiedToolId") or payload.get("qualified_tool_id")
    tool = _resolve_manager_tool(
        tool_name,
        server_name=server_name,
        qualified_tool_id=qualified_tool_id,
    )
    if tool is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="MCP tool not found")

    started_at = time.perf_counter()
    try:
        result = await manager.call_tool(
            qualified_tool_id=tool["qualifiedId"],
            arguments=payload.get("arguments") or {},
        )
        execution_time = time.perf_counter() - started_at
        data = {
            "success": True,
            "result": result,
            "error": None,
            "executionTime": execution_time,
            "toolName": tool_name,
            "serverName": tool["serverName"],
            "qualifiedId": tool["qualifiedId"],
        }
        return make_api_response(
            success=True,
            message=f"Tool '{tool_name}' executed successfully",
            data=data,
        )
    except Exception as exc:
        execution_time = time.perf_counter() - started_at
        data = {
            "success": False,
            "result": None,
            "error": str(exc),
            "executionTime": execution_time,
            "toolName": tool_name,
            "serverName": tool["serverName"],
            "qualifiedId": tool["qualifiedId"],
        }
        return make_api_response(
            success=False,
            message=f"Tool '{tool_name}' execution failed: {exc}",
            data=data,
        )


@router.post("/reload", include_in_schema=False)
async def reload_mcp(
    _session: LocalSessionPayload = Depends(require_local_session),
):
    """Local-only alias to reload MCP configuration."""
    await _reload_manager()
    await _refresh_runtime_bridge_catalogs_if_connected()
    return make_api_response(
        success=True,
        message="MCP configuration reloaded",
        data={"message": "MCP configuration reloaded"},
    )
