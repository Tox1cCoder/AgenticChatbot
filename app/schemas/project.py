"""Request and response schemas for projects."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.custom_agent import CustomAgentRead
from app.utils.case_conversion import to_camel_case as to_camel


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Project name")
    description: str | None = Field(None, max_length=2000, description="Short description")
    instructions: str | None = Field(
        None,
        max_length=8000,
        description="Instructions applied to every conversation in this project",
    )

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ProjectUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = Field(None, max_length=2000)
    instructions: str | None = Field(None, max_length=8000)

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ProjectRead(BaseModel):
    model_config = ConfigDict(
        from_attributes=True, alias_generator=to_camel, populate_by_name=True
    )

    id: UUID
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None
    owner_id: UUID
    name: str
    description: str | None = None
    instructions: str | None = None
    conversation_count: int = Field(
        default=0, description="Live conversations currently in this project"
    )
    custom_agents: list[CustomAgentRead] | None = Field(
        default=None,
        description="Ordered default agents (when requested)",
    )


class ProjectCustomAgentsUpdate(BaseModel):
    """Replace the ordered set of default custom agents for a project."""

    custom_agent_ids: list[UUID] = Field(default_factory=list)

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    @field_validator("custom_agent_ids")
    @classmethod
    def _no_duplicates(cls, value: list[UUID]) -> list[UUID]:
        if len(value) != len(set(value)):
            raise ValueError("custom_agent_ids must not contain duplicates")
        return value
