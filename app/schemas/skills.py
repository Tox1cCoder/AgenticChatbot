"""Pydantic schemas for the Skills system."""

from pydantic import BaseModel, ConfigDict

from app.utils.case_conversion import to_camel_case as to_camel


class SkillInfo(BaseModel):
    """Summary info for a skill (used in list responses)."""

    name: str
    description: str
    enabled: bool
    folder_path: str

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class SkillDetail(SkillInfo):
    """Full detail for a single skill, including Markdown body."""

    content: str


class SkillListResponse(BaseModel):
    """Response for listing all skills."""

    skills: list[SkillInfo]
    total_count: int
    enabled_count: int

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class SkillToggleRequest(BaseModel):
    """Request body for toggling a skill (unused if using query param)."""

    enabled: bool


class SkillOperationResponse(BaseModel):
    """Generic operation response with a message."""

    message: str
