from datetime import datetime
from typing import Optional, List
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict
from app.schemas.message import MessageRead
from app.utils.case_conversion import to_camel_case as to_camel


class ConversationCreate(BaseModel):
    title: str = Field(
        ..., min_length=1, max_length=255, description="Conversation title"
    )
    persona_prompt: Optional[str] = Field(
        None,
        max_length=8000,
        description="Custom persona/system instruction for this conversation",
    )

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ConversationUpdate(BaseModel):
    title: Optional[str] = Field(
        None, min_length=1, max_length=255, description="Conversation title"
    )
    persona_prompt: Optional[str] = Field(
        None,
        max_length=8000,
        description="Custom persona/system instruction for this conversation",
    )

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ConversationRead(BaseModel):
    model_config = ConfigDict(
        from_attributes=True, alias_generator=to_camel, populate_by_name=True
    )

    id: UUID
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    owner_id: UUID
    title: str = Field(
        ..., min_length=1, max_length=255, description="Conversation title"
    )
    persona_prompt: Optional[str] = Field(
        None,
        max_length=8000,
        description="Custom persona/system instruction for this conversation",
    )
    message_count: Optional[int] = Field(
        default=None, description="Total number of messages in the conversation"
    )
    messages: Optional[List["MessageRead"]] = Field(
        default=None, description="Recent messages in the conversation (when requested)"
    )


class ConversationInDB(BaseModel):
    model_config = ConfigDict(
        from_attributes=True, alias_generator=to_camel, populate_by_name=True
    )

    id: UUID
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    owner_id: UUID
    title: str = Field(
        ..., min_length=1, max_length=255, description="Conversation title"
    )
    persona_prompt: Optional[str] = Field(
        None,
        max_length=8000,
        description="Custom persona/system instruction for this conversation",
    )


# Update forward reference
ConversationRead.model_rebuild()
