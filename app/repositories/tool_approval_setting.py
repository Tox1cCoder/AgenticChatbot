"""Session-factory backed repository for per-user HITL approval settings."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import UUID

from sqlalchemy import select

from app.models.tool_approval_setting import ToolApprovalSetting

VALID_SCOPE_TYPES = {"server", "tool"}


class ToolApprovalSettingRepository:
    """Sync, session-factory style repository (mirrors CustomAgentRepository)."""

    def __init__(self, session_factory: Callable[[], Any]):
        self.session_factory = session_factory

    @staticmethod
    def _validate_scope(scope_type: str, scope_value: str) -> tuple[str, str]:
        normalized_type = str(scope_type or "").strip()
        normalized_value = str(scope_value or "").strip()
        if normalized_type not in VALID_SCOPE_TYPES:
            raise ValueError("scope_type must be 'server' or 'tool'")
        if not normalized_value:
            raise ValueError("scope_value is required")
        return normalized_type, normalized_value

    def list_by_user(self, user_id: UUID) -> list[ToolApprovalSetting]:
        with self.session_factory() as session:
            stmt = select(ToolApprovalSetting).where(ToolApprovalSetting.user_id == user_id)
            return list(session.execute(stmt).scalars().all())

    def set(
        self, user_id: UUID, scope_type: str, scope_value: str, require_approval: bool
    ) -> ToolApprovalSetting:
        scope_type, scope_value = self._validate_scope(scope_type, scope_value)
        with self.session_factory() as session:
            stmt = select(ToolApprovalSetting).where(
                ToolApprovalSetting.user_id == user_id,
                ToolApprovalSetting.scope_type == scope_type,
                ToolApprovalSetting.scope_value == scope_value,
            )
            setting = session.execute(stmt).scalar_one_or_none()
            if setting is None:
                setting = ToolApprovalSetting(
                    user_id=user_id,
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

    def bulk_set(self, user_id: UUID, items: list[dict[str, Any]]) -> list[ToolApprovalSetting]:
        return [
            self.set(
                user_id,
                str(item["scope_type"]),
                str(item["scope_value"]),
                bool(item["require_approval"]),
            )
            for item in items
        ]

    def delete(self, user_id: UUID, scope_type: str, scope_value: str) -> bool:
        scope_type, scope_value = self._validate_scope(scope_type, scope_value)
        with self.session_factory() as session:
            stmt = select(ToolApprovalSetting).where(
                ToolApprovalSetting.user_id == user_id,
                ToolApprovalSetting.scope_type == scope_type,
                ToolApprovalSetting.scope_value == scope_value,
            )
            setting = session.execute(stmt).scalar_one_or_none()
            if setting is None:
                return False
            session.delete(setting)
            session.commit()
            return True

    def build_policy(self, user_id: UUID) -> dict[str, dict[str, bool]]:
        servers: dict[str, bool] = {}
        tools: dict[str, bool] = {}
        for row in self.list_by_user(user_id):
            if row.scope_type == "server":
                servers[row.scope_value] = bool(row.require_approval)
            elif row.scope_type == "tool":
                tools[row.scope_value] = bool(row.require_approval)
        return {"servers": servers, "tools": tools}
