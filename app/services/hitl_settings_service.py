"""Per-user HITL approval settings service."""

from __future__ import annotations

from uuid import UUID

from app.ai.hitl_config import get_tools_requiring_approval, is_hitl_enabled
from app.repositories.tool_approval_setting import ToolApprovalSettingRepository


class HitlSettingsService:
    def __init__(self, repository: ToolApprovalSettingRepository):
        self.repository = repository

    def get_settings(self, user_id: UUID) -> dict:
        rows = self.repository.list_by_user(user_id)
        return {
            "master_enabled": is_hitl_enabled(),
            "global_tools": list(get_tools_requiring_approval()),
            "servers": [
                {"scope_type": "server", "scope_value": r.scope_value,
                 "require_approval": bool(r.require_approval)}
                for r in rows if r.scope_type == "server"
            ],
            "tools": [
                {"scope_type": "tool", "scope_value": r.scope_value,
                 "require_approval": bool(r.require_approval)}
                for r in rows if r.scope_type == "tool"
            ],
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
