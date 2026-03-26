from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from cachetools import TTLCache
from langchain_core.tools import BaseTool, StructuredTool

from app.core.config import settings
from app.services.client_device_service import ClientDeviceService

from .tool_context import get_tool_context

logger = logging.getLogger(__name__)

_CLIENT_TOOL_CACHE = TTLCache(
    maxsize=512,
    ttl=max(1, settings.client_runtime_catalog_cache_ttl_seconds),
)


@dataclass(frozen=True)
class ClientRuntimeToolSpec:
    """Sanitized description of a client-local tool synced from a device runtime."""

    name: str
    description: str
    origin: str
    server_name: str | None
    qualified_tool_id: str
    input_schema: dict[str, Any]
    exposed_name: str

    @property
    def tool_origin(self) -> str:
        if self.origin == "mcp":
            return "client_mcp"
        return "client_native"


def _sanitize_name_token(value: str | None) -> str:
    token = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value or "").strip()).strip("_").lower()
    return token or "tool"


def _normalize_input_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    normalized = dict(schema)
    normalized.setdefault("type", "object")
    normalized.setdefault("properties", {})
    return normalized


def _build_exposed_name(raw_entry: dict[str, Any], seen: set[str]) -> str:
    origin = str(raw_entry.get("origin") or "native").strip().lower()
    base_name = _sanitize_name_token(raw_entry.get("name"))

    if origin == "mcp":
        server_name = _sanitize_name_token(raw_entry.get("server_name"))
        candidate = f"client__{server_name}__{base_name}"
    else:
        candidate = f"client__{base_name}"

    if candidate not in seen:
        seen.add(candidate)
        return candidate

    suffix = 2
    while f"{candidate}_{suffix}" in seen:
        suffix += 1

    unique_name = f"{candidate}_{suffix}"
    seen.add(unique_name)
    return unique_name


def _parse_tool_specs(catalog: dict[str, Any]) -> list[ClientRuntimeToolSpec]:
    raw_tools = catalog.get("tools", []) if isinstance(catalog, dict) else []
    if not isinstance(raw_tools, list):
        return []

    seen_names: set[str] = set()
    parsed: list[ClientRuntimeToolSpec] = []

    for raw_entry in raw_tools:
        if not isinstance(raw_entry, dict):
            continue

        qualified_tool_id = str(raw_entry.get("qualified_id") or "").strip()
        name = str(raw_entry.get("name") or "").strip()
        if not qualified_tool_id or not name:
            continue

        description = str(raw_entry.get("description") or "").strip() or (
            f"Execute the client-local tool '{name}'."
        )
        exposed_name = _build_exposed_name(raw_entry, seen_names)

        parsed.append(
            ClientRuntimeToolSpec(
                name=name,
                description=description,
                origin=str(raw_entry.get("origin") or "native").strip().lower(),
                server_name=(
                    str(raw_entry.get("server_name")).strip()
                    if raw_entry.get("server_name")
                    else None
                ),
                qualified_tool_id=qualified_tool_id,
                input_schema=_normalize_input_schema(raw_entry.get("input_schema")),
                exposed_name=exposed_name,
            )
        )

    return parsed


def _format_tool_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, indent=2, ensure_ascii=False, default=str)


def _build_tool(
    *,
    spec: ClientRuntimeToolSpec,
    bound_user_id: str,
    bound_device_id: str,
    bound_session_id: str,
) -> BaseTool:
    async def _dispatch_client_tool(**kwargs: Any) -> str:
        ctx = get_tool_context()
        context_device_id = str(ctx.device_id or bound_device_id)

        if context_device_id != bound_device_id:
            raise RuntimeError(
                "Client-local tool was requested for a different device session than the active run."
            )

        session = ClientDeviceService.lookup_active_session(UUID(bound_device_id))
        if session is None or str(session.user_id) != bound_user_id:
            raise RuntimeError("Client device is not connected for this user.")

        if session.session_id != bound_session_id:
            raise RuntimeError(
                "Client device session changed after tool binding. Retry from the active device."
            )

        gateway = session.websocket
        if gateway is None or not hasattr(gateway, "dispatch_tool_call"):
            raise RuntimeError("Client runtime gateway is not available for tool dispatch.")

        response = await gateway.dispatch_tool_call(
            request_id=str(uuid4()),
            tool_name=spec.name,
            qualified_tool_id=spec.qualified_tool_id,
            arguments=kwargs,
            timeout_seconds=settings.client_runtime_ws_timeout_seconds,
        )

        if not response.get("success", False):
            error_message = response.get("error") or "Unknown client-local tool error"
            raise RuntimeError(error_message)

        return _format_tool_result(response.get("result"))

    description = (
        f"[Client device tool] {spec.description} "
        f"(origin={spec.tool_origin}, qualified_id={spec.qualified_tool_id})"
    )

    return StructuredTool.from_function(
        coroutine=_dispatch_client_tool,
        name=spec.exposed_name,
        description=description,
        args_schema=spec.input_schema,
        infer_schema=False,
        metadata={
            "client_runtime": True,
            "device_id": bound_device_id,
            "tool_origin": spec.tool_origin,
            "server_name": spec.server_name,
            "qualified_tool_id": spec.qualified_tool_id,
            "source_tool_name": spec.name,
        },
    )


def get_client_runtime_tools(
    *,
    user_id: str | None,
    device_id: str | None,
) -> list[BaseTool]:
    """
    Return the client-local tools currently available for a specific request device.

    The returned tool list is cached per `(user_id, device_id, session_id, catalog_version)`
    so the graph can rebuild bindings cheaply while still invalidating when the client
    runtime reconnects or resyncs its catalog.
    """
    if not settings.enable_client_runtime_bridge or not user_id or not device_id:
        return []

    try:
        device_uuid = UUID(str(device_id))
    except Exception:
        logger.warning("Invalid device_id passed to client runtime tool lookup: %s", device_id)
        return []

    session = ClientDeviceService.lookup_active_session(device_uuid)
    if session is None or str(session.user_id) != str(user_id):
        return []

    if not session.tool_catalog:
        return []

    cache_key = session.get_tool_cache_key()
    cached = _CLIENT_TOOL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    tool_specs = _parse_tool_specs(session.tool_catalog)
    tools = [
        _build_tool(
            spec=spec,
            bound_user_id=str(session.user_id),
            bound_device_id=str(session.device_id),
            bound_session_id=session.session_id,
        )
        for spec in tool_specs
    ]

    _CLIENT_TOOL_CACHE[cache_key] = tools
    return tools
