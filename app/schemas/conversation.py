from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import PlanLifecycle
from app.schemas.message import MessageRead
from app.utils.case_conversion import to_camel_case as to_camel


class ConversationCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255, description="Conversation title")
    persona_prompt: str | None = Field(
        None,
        max_length=8000,
        description="Custom persona/system instruction for this conversation",
    )
    planning_mode_enabled: bool | None = Field(
        default=False,
        description="Enable planning mode for this conversation",
    )

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ConversationUpdate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=255, description="Conversation title")
    persona_prompt: str | None = Field(
        None,
        max_length=8000,
        description="Custom persona/system instruction for this conversation",
    )
    planning_mode_enabled: bool | None = Field(
        None,
        description="Toggle planning mode",
    )

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ConversationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, alias_generator=to_camel, populate_by_name=True)

    id: UUID
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None
    owner_id: UUID
    title: str = Field(..., min_length=1, max_length=255, description="Conversation title")
    persona_prompt: str | None = Field(
        None,
        max_length=8000,
        description="Custom persona/system instruction for this conversation",
    )
    planning_mode_enabled: bool = Field(
        default=False,
        description="Whether planning mode is enabled",
    )
    plan_lifecycle: PlanLifecycle | None = Field(
        default=None,
        description="Explicit lifecycle state of the plan",
    )
    message_count: int | None = Field(
        default=None, description="Total number of messages in the conversation"
    )
    messages: list["MessageRead"] | None = Field(
        default=None, description="Recent messages in the conversation (when requested)"
    )


class ConversationInDB(BaseModel):
    model_config = ConfigDict(from_attributes=True, alias_generator=to_camel, populate_by_name=True)

    id: UUID
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None
    owner_id: UUID
    title: str = Field(..., min_length=1, max_length=255, description="Conversation title")
    persona_prompt: str | None = Field(
        None,
        max_length=8000,
        description="Custom persona/system instruction for this conversation",
    )
    planning_mode_enabled: bool = Field(
        default=False,
        description="Whether planning mode is enabled",
    )
    plan_lifecycle: PlanLifecycle | None = Field(
        default=None,
        description="Explicit lifecycle state of the plan",
    )


# Update forward reference
ConversationRead.model_rebuild()
