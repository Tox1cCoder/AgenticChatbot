"""Schemas for the per-user HITL approval settings API."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.utils.case_conversion import to_camel_case as to_camel


class _CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class HitlScopeRule(_CamelModel):
    scope_type: Literal["server", "tool"] = Field(..., description='"server" or "tool"')
    scope_value: str = Field(..., description="server name or qualified tool id")
    tool_origin: str = Field(..., description='"client_mcp" or "client_skill"')
    require_approval: bool


class HitlScopeRuleState(HitlScopeRule):
    """A device-scoped editable rule returned by the settings API."""


class HitlSettingsUpdate(_CamelModel):
    items: list[HitlScopeRule]


class HitlSettingsResponse(_CamelModel):
    device_id: UUID
    master_enabled: bool
    global_tools: list[str]
    servers: list[HitlScopeRuleState]
    tools: list[HitlScopeRuleState]


class HitlInterruptStateResponse(_CamelModel):
    interrupt_id: str
    conversation_id: UUID
    status: Literal["pending", "resolving", "resolved", "failed", "expired"]
    expires_at: datetime
    updated_at: datetime
