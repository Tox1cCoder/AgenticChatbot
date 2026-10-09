from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from app.models.enums import PlanLifecycle
from app.schemas.custom_agent import CustomAgentRead
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
    project_id: UUID | None = Field(
        None, description="Project this conversation belongs to, if any"
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

    @field_validator("title", "planning_mode_enabled")
    @classmethod
    def _not_nullable(cls, value: Any, info: ValidationInfo) -> Any:
        # Both columns are NOT NULL. Updates apply ``exclude_unset``, so omitting
        # a field is a no-op, but an explicit null reached the column and failed
        # as an IntegrityError (500). Validators skip omitted fields.
        # ``persona_prompt`` stays nullable: null clears it.
        if value is None:
            raise ValueError(f"{info.field_name} cannot be null; omit it to leave it unchanged")
        return value


class ConversationForkRequest(BaseModel):
    message_id: UUID = Field(..., description="Assistant message to fork through (inclusive)")
    request_id: UUID = Field(..., description="Client-generated ID making a retry idempotent")
    title: str | None = Field(
        default=None, min_length=1, max_length=255, description="Optional child title"
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
    custom_agents: list["CustomAgentRead"] | None = Field(
        default=None,
        description="Custom agents attached to this conversation (when requested)",
    )
    project_id: UUID | None = Field(
        default=None, description="Project this conversation belongs to, if any"
    )
    forked_from_conversation_id: UUID | None = Field(
        default=None, description="Conversation this one was forked from, if any"
    )
    forked_from_message_id: UUID | None = Field(
        default=None, description="Source message this fork was cut at, if any"
    )
    forked_through_sequence: int | None = Field(
        default=None, description="Inclusive source sequence copied into this fork"
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
    project_id: UUID | None = Field(
        default=None, description="Project this conversation belongs to, if any"
    )
    forked_from_conversation_id: UUID | None = Field(
        default=None, description="Conversation this one was forked from, if any"
    )
    forked_from_message_id: UUID | None = Field(
        default=None, description="Source message this fork was cut at, if any"
    )
    forked_through_sequence: int | None = Field(
        default=None, description="Inclusive source sequence copied into this fork"
    )


# Update forward reference
ConversationRead.model_rebuild()
