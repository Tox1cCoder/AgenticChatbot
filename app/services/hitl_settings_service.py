"""Per-user HITL approval settings service."""

from __future__ import annotations

from uuid import UUID

from app.ai.hitl_config import get_tools_requiring_approval, is_hitl_enabled
from app.repositories.tool_approval_setting import ToolApprovalSettingRepository


def _lookup_device_session(user_id: UUID, device_id: str):
    """Return the caller-owned active runtime session for a device, if any."""
    try:
        device_uuid = UUID(str(device_id))
    except (ValueError, AttributeError, TypeError):
        return None

    from app.services.client_device_service import ClientDeviceService

    session = ClientDeviceService.lookup_active_session(device_uuid)
    if session is None or str(session.user_id) != str(user_id):
        return None
    return session


def _add_server_side_capabilities(servers: set[str], tools: set[str]) -> None:
    try:
        from app.ai.mcp_registry import MCPRegistry

        manager = MCPRegistry.get_manager_sync()
    except Exception:  # pragma: no cover - defensive; no MCP runtime in scope
        return
    if manager is None:
        return
    for server_name, server_tools in getattr(manager, "_server_tools", {}).items():
        server = str(server_name).strip()
        if server:
            servers.add(server)
        for tool in server_tools or []:
            name = str(getattr(tool, "name", "") or "").strip()
            if name:
                tools.add(name)
                if server:
                    tools.add(f"{server}::{name}")


def _build_capability_index(user_id: UUID, device_id: str) -> tuple[set[str], set[str]]:
    """Server names and tool ids reachable from this request's device context."""
    servers: set[str] = set()
    tools: set[str] = set()

    session = _lookup_device_session(user_id, device_id)
    if session is not None:
        entries = (session.tool_catalog or {}).get("tools", []) or []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            server = str(entry.get("server_name") or "").strip()
            qualified_id = str(entry.get("qualified_id") or "").strip()
            name = str(entry.get("name") or "").strip()
            if server:
                servers.add(server)
            if qualified_id:
                tools.add(qualified_id)
            if name:
                tools.add(name)

    _add_server_side_capabilities(servers, tools)
    return servers, tools


class HitlSettingsService:
    def __init__(self, repository: ToolApprovalSettingRepository):
        self.repository = repository

    def get_settings(self, user_id: UUID, device_id: str | None = None) -> dict:
        rows = self.repository.list_by_user(user_id)
        capability_index = _build_capability_index(user_id, device_id) if device_id else None

        def rule(row) -> dict:
            item = {
                "scope_type": row.scope_type,
                "scope_value": row.scope_value,
                "require_approval": bool(row.require_approval),
            }
            if capability_index is not None:
                known_servers, known_tools = capability_index
                known = known_servers if row.scope_type == "server" else known_tools
                item["available"] = row.scope_value in known
            return item

        return {
            "master_enabled": is_hitl_enabled(),
            "global_tools": list(get_tools_requiring_approval()),
            "servers": [rule(r) for r in rows if r.scope_type == "server"],
            "tools": [rule(r) for r in rows if r.scope_type == "tool"],
        }

    def apply(self, user_id: UUID, items: list[dict]) -> dict:
        self.repository.bulk_set(user_id, items)
        return self.get_settings(user_id)

    def clear(self, user_id: UUID, scope_type: str, scope_value: str) -> dict:
        self.repository.delete(user_id, scope_type, scope_value)
        return self.get_settings(user_id)

    def build_turn_policy(self, user_id: UUID) -> dict:
        """Full policy dict consumed by the graph gate (checkpoint-safe)."""
        grouped = self.repository.build_policy(user_id)
        return {
            "master_enabled": is_hitl_enabled(),
            "servers": grouped["servers"],
            "tools": grouped["tools"],
            "global_tools": list(get_tools_requiring_approval()),
        }
