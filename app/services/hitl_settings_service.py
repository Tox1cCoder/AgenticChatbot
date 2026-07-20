"""Per-user HITL approval settings service."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select

from app.ai.hitl_config import get_tools_requiring_approval, is_hitl_enabled
from app.core.exceptions import CustomHTTPException
from app.models.client_device import ClientDevice
from app.repositories.tool_approval_setting import ToolApprovalSettingRepository

_EDITABLE_ORIGINS = {"client_mcp", "client_skill"}


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


def _device_belongs_to_user(user_id: UUID, device_id: UUID, session_factory) -> bool:
    with session_factory() as session:
        stmt = select(ClientDevice.id).where(
            ClientDevice.id == device_id,
            ClientDevice.user_id == user_id,
        )
        return session.execute(stmt).scalar_one_or_none() is not None


def _build_capability_index(user_id: UUID, device_id: UUID) -> dict[str, dict[str, set[str]]]:
    """Index editable targets reachable from one active client device."""
    index = {
        origin: {"servers": set(), "tools": set()} for origin in sorted(_EDITABLE_ORIGINS)
    }

    session = _lookup_device_session(user_id, str(device_id))
    if session is not None:
        entries = (session.tool_catalog or {}).get("tools", []) or []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            origin = str(entry.get("origin") or "").strip().lower()
            if origin not in index:
                continue
            origin_index = index[origin]
            server = str(entry.get("server_name") or "").strip()
            qualified_id = str(entry.get("qualified_id") or "").strip()
            name = str(entry.get("name") or "").strip()
            if server:
                origin_index["servers"].add(server)
            if qualified_id:
                origin_index["tools"].add(qualified_id)
            if name:
                origin_index["tools"].add(name)
    return index


class HitlSettingsService:
    def __init__(self, repository: ToolApprovalSettingRepository):
        self.repository = repository

    def _resolve_device(self, user_id: UUID, device_id: str | None) -> UUID:
        if not device_id:
            raise CustomHTTPException(
                422,
                "A registered client device is required for HITL settings.",
                "HITL_DEVICE_REQUIRED",
            )
        try:
            resolved = UUID(str(device_id))
        except (TypeError, ValueError, AttributeError) as exc:
            raise CustomHTTPException(
                404,
                "Client device not found.",
                "HITL_DEVICE_NOT_FOUND",
            ) from exc
        if not _device_belongs_to_user(user_id, resolved, self.repository.session_factory):
            raise CustomHTTPException(
                404,
                "Client device not found.",
                "HITL_DEVICE_NOT_FOUND",
            )
        return resolved

    @staticmethod
    def _validate_origin(tool_origin: str) -> str:
        normalized = str(tool_origin or "").strip().lower()
        if normalized not in _EDITABLE_ORIGINS:
            raise CustomHTTPException(
                422,
                "toolOrigin must be 'client_mcp' or 'client_skill'.",
                "HITL_TOOL_ORIGIN_INVALID",
            )
        return normalized

    def get_settings(self, user_id: UUID, device_id: str | None = None) -> dict:
        resolved_device = self._resolve_device(user_id, device_id)
        rows = self.repository.list_by_device(user_id, resolved_device)

        def rule(row) -> dict:
            return {
                "scope_type": row.scope_type,
                "scope_value": row.scope_value,
                "tool_origin": row.tool_origin,
                "require_approval": bool(row.require_approval),
            }

        return {
            "device_id": resolved_device,
            "master_enabled": is_hitl_enabled(),
            "global_tools": list(get_tools_requiring_approval()),
            "servers": [rule(r) for r in rows if r.scope_type == "server"],
            "tools": [rule(r) for r in rows if r.scope_type == "tool"],
        }

    def apply(self, user_id: UUID, device_id: str | None, items: list[dict]) -> dict:
        resolved_device = self._resolve_device(user_id, device_id)
        session = _lookup_device_session(user_id, str(resolved_device))
        if session is None:
            raise CustomHTTPException(
                409,
                "The client device runtime is not connected.",
                "HITL_DEVICE_RUNTIME_UNAVAILABLE",
            )
        capabilities = _build_capability_index(user_id, resolved_device)
        normalized_items = []
        for item in items:
            normalized = dict(item)
            origin = self._validate_origin(normalized.get("tool_origin"))
            normalized["tool_origin"] = origin
            scope_type = str(normalized.get("scope_type") or "").strip()
            scope_value = str(normalized.get("scope_value") or "").strip()
            target_group = "servers" if scope_type == "server" else "tools"
            if scope_value not in capabilities[origin][target_group]:
                raise CustomHTTPException(
                    409,
                    "The requested HITL target is not in the active device catalog.",
                    "HITL_TARGET_UNAVAILABLE",
                )
            normalized_items.append(normalized)
        self.repository.bulk_set(user_id, resolved_device, normalized_items)
        return self.get_settings(user_id, str(resolved_device))

    def clear(
        self,
        user_id: UUID,
        device_id: str | None,
        tool_origin: str,
        scope_type: str,
        scope_value: str,
    ) -> dict:
        resolved_device = self._resolve_device(user_id, device_id)
        origin = self._validate_origin(tool_origin)
        self.repository.delete(
            user_id,
            resolved_device,
            origin,
            scope_type,
            scope_value,
        )
        return self.get_settings(user_id, str(resolved_device))

    def build_turn_policy(self, user_id: UUID, device_id: UUID | None) -> dict:
        """Full policy dict consumed by the graph gate (checkpoint-safe)."""
        grouped = (
            self.repository.build_policy(user_id, device_id)
            if device_id is not None
            else {
                "client_mcp": {"servers": {}, "tools": {}},
                "client_skill": {"servers": {}, "tools": {}},
            }
        )
        return {
            "master_enabled": is_hitl_enabled(),
            "client_rules": grouped,
            "global_tools": list(get_tools_requiring_approval()),
        }
