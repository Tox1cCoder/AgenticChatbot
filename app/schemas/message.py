from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class MessageCreate(BaseModel):
    conversation_id: UUID = Field(
        ..., description="Conversation ID this message belongs to"
    )
    content: str = Field(..., min_length=1, description="Message content")
    sender: int = Field(
        ..., description="Message sender: 1=user, 2=assistant, 3=system"
    )


class MessageUpdate(BaseModel):
    content: Optional[str] = Field(None, min_length=1, description="Message content")


class MessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    conversation_id: UUID
    sender: int = Field(
        ..., description="Message sender: 1=user, 2=assistant, 3=system"
    )
    content: str = Field(..., min_length=1, description="Message content")


class MessageInDB(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    conversation_id: UUID
    sender: int = Field(
        ..., description="Message sender: 1=user, 2=assistant, 3=system"
    )
    content: str = Field(..., min_length=1, description="Message content")
