"""Schemas for the per-user HITL approval settings API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.utils.case_conversion import to_camel_case as to_camel


class _CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class HitlScopeRule(_CamelModel):
    scope_type: Literal["server", "tool"] = Field(..., description='"server" or "tool"')
    scope_value: str = Field(..., description="server name or qualified tool id")
    require_approval: bool


class HitlSettingsUpdate(_CamelModel):
    items: list[HitlScopeRule]


class HitlSettingsResponse(_CamelModel):
    master_enabled: bool
    global_tools: list[str]
    servers: list[HitlScopeRule]
    tools: list[HitlScopeRule]
