"""Session-factory backed repository for per-user HITL approval settings."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import UUID

from sqlalchemy import select

from app.models.tool_approval_setting import ToolApprovalSetting

VALID_SCOPE_TYPES = {"server", "tool"}
VALID_TOOL_ORIGINS = {"client_mcp", "client_skill"}


class ToolApprovalSettingRepository:
    """Sync, session-factory style repository (mirrors CustomAgentRepository)."""

    def __init__(self, session_factory: Callable[[], Any]):
        self.session_factory = session_factory

    @staticmethod
    def _validate_scope(
        tool_origin: str, scope_type: str, scope_value: str
    ) -> tuple[str, str, str]:
        normalized_origin = str(tool_origin or "").strip().lower()
        normalized_type = str(scope_type or "").strip()
        normalized_value = str(scope_value or "").strip()
        if normalized_origin not in VALID_TOOL_ORIGINS:
            raise ValueError("tool_origin must be 'client_mcp' or 'client_skill'")
        if normalized_type not in VALID_SCOPE_TYPES:
            raise ValueError("scope_type must be 'server' or 'tool'")
        if not normalized_value:
            raise ValueError("scope_value is required")
        return normalized_origin, normalized_type, normalized_value

    def list_by_device(self, user_id: UUID, device_id: UUID) -> list[ToolApprovalSetting]:
        with self.session_factory() as session:
            stmt = select(ToolApprovalSetting).where(
                ToolApprovalSetting.user_id == user_id,
                ToolApprovalSetting.device_id == device_id,
            )
            return list(session.execute(stmt).scalars().all())

    def set(
        self,
        user_id: UUID,
        device_id: UUID,
        tool_origin: str,
        scope_type: str,
        scope_value: str,
        require_approval: bool,
    ) -> ToolApprovalSetting:
        tool_origin, scope_type, scope_value = self._validate_scope(
            tool_origin, scope_type, scope_value
        )
        with self.session_factory() as session:
            stmt = select(ToolApprovalSetting).where(
                ToolApprovalSetting.user_id == user_id,
                ToolApprovalSetting.device_id == device_id,
                ToolApprovalSetting.tool_origin == tool_origin,
                ToolApprovalSetting.scope_type == scope_type,
                ToolApprovalSetting.scope_value == scope_value,
            )
            setting = session.execute(stmt).scalar_one_or_none()
            if setting is None:
                setting = ToolApprovalSetting(
                    user_id=user_id,
                    device_id=device_id,
                    tool_origin=tool_origin,
                    scope_type=scope_type,
                    scope_value=scope_value,
                    require_approval=require_approval,
                )
                session.add(setting)
            else:
                setting.require_approval = require_approval
            session.commit()
            session.refresh(setting)
            session.expunge(setting)
            return setting

    def bulk_set(
        self, user_id: UUID, device_id: UUID, items: list[dict[str, Any]]
    ) -> list[ToolApprovalSetting]:
        return [
            self.set(
                user_id,
                device_id,
                str(item["tool_origin"]),
                str(item["scope_type"]),
                str(item["scope_value"]),
                bool(item["require_approval"]),
            )
            for item in items
        ]

    def delete(
        self,
        user_id: UUID,
        device_id: UUID,
        tool_origin: str,
        scope_type: str,
        scope_value: str,
    ) -> bool:
        tool_origin, scope_type, scope_value = self._validate_scope(
            tool_origin, scope_type, scope_value
        )
        with self.session_factory() as session:
            stmt = select(ToolApprovalSetting).where(
                ToolApprovalSetting.user_id == user_id,
                ToolApprovalSetting.device_id == device_id,
                ToolApprovalSetting.tool_origin == tool_origin,
                ToolApprovalSetting.scope_type == scope_type,
                ToolApprovalSetting.scope_value == scope_value,
            )
            setting = session.execute(stmt).scalar_one_or_none()
            if setting is None:
                return False
            session.delete(setting)
            session.commit()
            return True

    def build_policy(self, user_id: UUID, device_id: UUID) -> dict[str, dict[str, dict[str, bool]]]:
        grouped: dict[str, dict[str, dict[str, bool]]] = {
            origin: {"servers": {}, "tools": {}} for origin in sorted(VALID_TOOL_ORIGINS)
        }
        for row in self.list_by_device(user_id, device_id):
            origin_rules = grouped.get(row.tool_origin)
            if origin_rules is None:
                continue
            if row.scope_type == "server":
                origin_rules["servers"][row.scope_value] = bool(row.require_approval)
            elif row.scope_type == "tool":
                origin_rules["tools"][row.scope_value] = bool(row.require_approval)
        return grouped
