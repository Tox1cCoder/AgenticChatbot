from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class MessageBase(BaseModel):
    content: str = Field(..., min_length=1, description="Message content")
    sender: int = Field(
        ..., description="Message sender: 1=user, 2=assistant, 3=system"
    )


class MessageCreate(MessageBase):
    conversation_id: UUID = Field(
        ..., description="Conversation ID this message belongs to"
    )


class MessageUpdate(BaseModel):
    content: Optional[str] = Field(None, min_length=1, description="Message content")


class MessageRead(MessageBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    conversation_id: UUID
    created_at: datetime


class MessageInDB(MessageRead):
    pass
