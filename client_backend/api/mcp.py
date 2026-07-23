"""Device-scoped MCP configuration and execution endpoints."""

from __future__ import annotations

import shlex
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import ValidationError

from client_backend.api.common import make_api_response
from client_backend.core.auth import require_local_session
from client_backend.core.security import LocalSessionPayload
from client_backend.schemas.mcp_config import MCPProfileScope
from client_backend.services.local_mcp_manager import (
    LocalMCPManager,
    get_mcp_manager,
    shutdown_mcp_manager,
)
from client_backend.services.mcp_config_migration import prepare_mcp_config_store
from client_backend.services.mcp_config_store import (
    EffectiveMCPServer,
    MCPConfigConflictError,
    MCPConfigStore,
)
from client_backend.services.runtime_bridge import get_runtime_bridge

router = APIRouter(prefix="/mcp", tags=["mcp"])


def _scope(session: LocalSessionPayload) -> MCPProfileScope:
    return MCPProfileScope(
        user_id=session.user_id,
        device_identifier=session.device_identifier,
    )


def _store(session: LocalSessionPayload) -> MCPConfigStore:
    store, _ = prepare_mcp_config_store(_scope(session))
    return store


async def _reload_manager(scope: MCPProfileScope) -> LocalMCPManager:
    await shutdown_mcp_manager(scope)
    manager = get_mcp_manager(scope)
    await manager.initialize()
    return manager


async def _refresh_runtime_bridge_catalogs_if_connected(
    scope: MCPProfileScope,
) -> None:
    bridge = get_runtime_bridge()
    if not bridge.is_connected() or not bridge.get_registered_device_id():
        return
    if bridge.get_device_identifier() != scope.device_identifier:
        return
    bridge_scope = getattr(bridge, "_mcp_scope", None)
    if bridge_scope is not None and bridge_scope != scope:
        return
    await bridge.refresh_catalogs()


def _manager_tool_lookup(manager: LocalMCPManager) -> dict[str, list[dict[str, Any]]]:
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
    manager: LocalMCPManager,
    tool_name: str,
    *,
    server_name: str | None = None,
    qualified_tool_id: str | None = None,
) -> dict[str, Any] | None:
    matches = _manager_tool_lookup(manager).get(tool_name) or []
    normalized_id = str(qualified_tool_id or "").strip()
    normalized_server = str(server_name or "").strip()
    if normalized_id:
        return next(
            (
                tool
                for tool in matches
                if tool["qualifiedId"] == normalized_id
                and (
                    not normalized_server
                    or tool["serverName"] == normalized_server
                )
            ),
            None,
        )
    if normalized_server:
        return next(
            (
                tool
                for tool in matches
                if tool["serverName"] == normalized_server
            ),
            None,
        )
    return matches[0] if matches else None


def _server_info(
    server: EffectiveMCPServer,
    manager: LocalMCPManager,
) -> dict[str, Any]:
    runtime = manager.servers.get(server.name)
    tool_count = len(runtime.tools) if runtime and runtime.is_running() else 0
    config: dict[str, Any] = {
        "transport": server.transport,
        "enabled": server.enabled,
        "description": server.description,
    }
    if server.transport == "stdio":
        config.update(
            {
                "command": server.command,
                "args": server.args,
                "cwd": server.cwd,
                "envKeys": sorted(server.env),
            }
        )
    else:
        config.update(
            {
                "url": server.url,
                "headerKeys": sorted(server.headers),
            }
        )
    return {
        "name": server.name,
        "source": server.source,
        "transport": server.transport,
        "enabled": server.enabled,
        "description": server.description,
        "toolCount": tool_count,
        "config": config,
    }


def _custom_definition(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    transport = str(payload.get("transport") or "stdio").strip().lower()
    if transport == "http":
        transport = "streamable_http"
    env = {str(key): str(value) for key, value in (payload.get("env") or {}).items()}
    headers = {
        str(key): str(value)
        for key, value in (payload.get("headers") or {}).items()
    }
    common = {
        "transport": transport,
        "enabled": bool(payload.get("enabled", True)),
        "description": str(payload.get("description") or ""),
    }
    if transport == "stdio":
        definition = {
            **common,
            "command": payload.get("command"),
            "args": list(payload.get("args") or []),
            "cwd": payload.get("cwd"),
            "envKeys": sorted(env),
        }
    else:
        definition = {
            **common,
            "url": payload.get("url"),
            "headerKeys": sorted(headers),
        }
    return definition, env, headers


def _parse_server_url_payload(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    raw_value = str(payload.get("url") or "").strip()
    if not raw_value:
        raise HTTPException(status_code=422, detail="`url` is required")

    name = str(payload.get("name") or "").strip()
    common = {
        "enabled": bool(payload.get("enabled", True)),
        "description": str(payload.get("description") or ""),
    }
    if raw_value.startswith(("http://", "https://")):
        return name or Path(raw_value.rstrip("/")).name or "mcp-server", {
            **common,
            "transport": "streamable_http",
            "url": raw_value,
        }

    parts = shlex.split(raw_value, posix=sys.platform != "win32")
    if not parts:
        raise HTTPException(status_code=422, detail="Unable to parse MCP command string")
    command, *args = parts
    return name or Path(command).stem or "mcp-server", {
        **common,
        "transport": "stdio",
        "command": command,
        "args": args,
    }


async def _save_custom(
    session: LocalSessionPayload,
    name: str,
    payload: dict[str, Any],
) -> None:
    store = _store(session)
    definition, env, headers = _custom_definition(payload)
    existing_credentials = store.secret_store.get_for_server(name)
    if "env" not in payload:
        env = existing_credentials.env
    if "headers" not in payload:
        headers = existing_credentials.headers
    try:
        store.save_custom_server(
            name,
            definition,
            env=env,
            headers=headers,
        )
    except MCPConfigConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    scope = _scope(session)
    await _reload_manager(scope)
    await _refresh_runtime_bridge_catalogs_if_connected(scope)


@router.get("/servers")
async def list_mcp_servers(
    session: LocalSessionPayload = Depends(require_local_session),
):
    store = _store(session)
    manager = get_mcp_manager(_scope(session))
    await manager.initialize()
    servers = [
        _server_info(server, manager)
        for server in sorted(store.list_effective_servers(), key=lambda item: item.name)
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
    session: LocalSessionPayload = Depends(require_local_session),
):
    store = _store(session)
    manager = get_mcp_manager(_scope(session))
    await manager.initialize()
    server = next(
        (
            candidate
            for candidate in store.list_effective_servers()
            if candidate.name == server_name
        ),
        None,
    )
    if server is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="MCP server not found",
        )
    return make_api_response(
        success=True,
        message=f"Server '{server_name}' details retrieved successfully",
        data=_server_info(server, manager),
    )


@router.post("/servers", status_code=status.HTTP_201_CREATED)
async def add_mcp_server(
    payload: dict[str, Any],
    session: LocalSessionPayload = Depends(require_local_session),
):
    name = str(payload.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=422, detail="`name` is required")
    await _save_custom(session, name, payload)
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
    session: LocalSessionPayload = Depends(require_local_session),
):
    name, definition = _parse_server_url_payload(payload)
    await _save_custom(session, name, definition)
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
    session: LocalSessionPayload = Depends(require_local_session),
):
    store = _store(session)
    try:
        action = store.delete_server(server_name)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="MCP server not found",
        ) from exc
    scope = _scope(session)
    await _reload_manager(scope)
    await _refresh_runtime_bridge_catalogs_if_connected(scope)
    message = (
        f"MCP server '{server_name}' disabled successfully"
        if action == "disabled_bundled"
        else f"MCP server '{server_name}' removed successfully"
    )
    return make_api_response(success=True, message=message, data={"message": message})


@router.patch("/servers/{server_name}/toggle")
async def toggle_mcp_server(
    server_name: str,
    enabled: bool = Query(..., description="True to enable, False to disable"),
    session: LocalSessionPayload = Depends(require_local_session),
):
    store = _store(session)
    try:
        store.set_enabled(server_name, enabled)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="MCP server not found",
        ) from exc
    scope = _scope(session)
    await _reload_manager(scope)
    await _refresh_runtime_bridge_catalogs_if_connected(scope)
    state = "enabled" if enabled else "disabled"
    message = f"MCP server '{server_name}' {state} successfully"
    return make_api_response(success=True, message=message, data={"message": message})


@router.get("/tools")
async def list_mcp_tools(
    server_name: str | None = Query(
        None,
        alias="serverName",
        description="Filter by server name",
    ),
    session: LocalSessionPayload = Depends(require_local_session),
):
    manager = get_mcp_manager(_scope(session))
    await manager.initialize()
    tools = [
        {
            "name": tool.name,
            "description": tool.description,
            "argsSchema": tool.input_schema,
            "serverName": tool.server_name,
            "qualifiedId": tool.qualified_id,
        }
        for tool in manager.get_all_tools()
        if not server_name or tool.server_name == server_name
    ]
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
    session: LocalSessionPayload = Depends(require_local_session),
):
    manager = get_mcp_manager(_scope(session))
    await manager.initialize()
    matches = _manager_tool_lookup(manager).get(tool_name) or []
    if not matches:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="MCP tool not found",
        )
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
    session: LocalSessionPayload = Depends(require_local_session),
):
    manager = get_mcp_manager(_scope(session))
    await manager.initialize()
    tool = _resolve_manager_tool(
        manager,
        tool_name,
        server_name=payload.get("serverName") or payload.get("server_name"),
        qualified_tool_id=(
            payload.get("qualifiedToolId") or payload.get("qualified_tool_id")
        ),
    )
    if tool is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="MCP tool not found",
        )
    started_at = time.perf_counter()
    try:
        result = await manager.call_tool(
            qualified_tool_id=tool["qualifiedId"],
            arguments=payload.get("arguments") or {},
        )
        execution_time = time.perf_counter() - started_at
        return make_api_response(
            success=True,
            message=f"Tool '{tool_name}' executed successfully",
            data={
                "success": True,
                "result": result,
                "error": None,
                "executionTime": execution_time,
                "toolName": tool_name,
                "serverName": tool["serverName"],
                "qualifiedId": tool["qualifiedId"],
            },
        )
    except Exception as exc:
        execution_time = time.perf_counter() - started_at
        return make_api_response(
            success=False,
            message=f"Tool '{tool_name}' execution failed: {exc}",
            data={
                "success": False,
                "result": None,
                "error": str(exc),
                "executionTime": execution_time,
                "toolName": tool_name,
                "serverName": tool["serverName"],
                "qualifiedId": tool["qualifiedId"],
            },
        )


@router.post("/reload", include_in_schema=False)
async def reload_mcp(
    session: LocalSessionPayload = Depends(require_local_session),
):
    scope = _scope(session)
    await _reload_manager(scope)
    await _refresh_runtime_bridge_catalogs_if_connected(scope)
    return make_api_response(
        success=True,
        message="MCP configuration reloaded",
        data={"message": "MCP configuration reloaded"},
    )
