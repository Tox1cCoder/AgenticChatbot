"""Schemas for the per-user HITL approval settings API."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_serializer

from app.utils.case_conversion import to_camel_case as to_camel


class _CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class HitlScopeRule(_CamelModel):
    scope_type: Literal["server", "tool"] = Field(..., description='"server" or "tool"')
    scope_value: str = Field(..., description="server name or qualified tool id")
    require_approval: bool


class HitlScopeRuleState(HitlScopeRule):
    """A stored rule as returned by the settings API.

    ``available`` is present only when the request supplied a ``deviceId``:
    it reports whether the rule's target currently exists in that device's
    live tool catalog or in the server-side MCP registry. Rules are
    account-wide policy; availability is per-device context.
    """

    available: bool | None = Field(
        None,
        description=(
            "Whether the rule's target exists on the requested device or the "
            "server. Present only when the request passed deviceId."
        ),
    )

    @model_serializer(mode="wrap")
    def _omit_absent_available(self, handler):
        data = handler(self)
        if self.available is None:
            data.pop("available", None)
        return data


class HitlSettingsUpdate(_CamelModel):
    items: list[HitlScopeRule]


class HitlSettingsResponse(_CamelModel):
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
