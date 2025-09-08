from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict

from app.models.enums import MessageRole


class MessageBase(BaseModel):
    content: str = Field(..., min_length=1, description="Message content")
    role: MessageRole = Field(
        ..., description="Message role: user, assistant, or system"
    )


class MessageCreate(MessageBase):
    conversation_id: UUID = Field(
        ..., description="Conversation ID this message belongs to"
    )
    parent_message_id: Optional[UUID] = Field(
        None, description="Parent message ID for threaded conversations"
    )


class MessageUpdate(BaseModel):
    content: Optional[str] = Field(None, min_length=1, description="Message content")


class MessageRead(MessageBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    conversation_id: UUID
    parent_message_id: Optional[UUID] = None
    created_at: datetime


class MessageInDB(MessageRead):
    pass
