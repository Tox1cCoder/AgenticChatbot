from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict
from app.models.enums import MessageRole


def to_camel(string: str) -> str:
    parts = string.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


class MessageCreate(BaseModel):
    conversation_id: UUID = Field(
        ..., description="Conversation ID this message belongs to"
    )
    content: str = Field(..., min_length=1, description="Message content")
    role: MessageRole = Field(
        default=MessageRole.user, description="Message role: user=1, assistant=2"
    )

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class MessageUpdate(BaseModel):
    content: Optional[str] = Field(None, min_length=1, description="Message content")

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class MessageRead(BaseModel):
    model_config = ConfigDict(
        from_attributes=True, alias_generator=to_camel, populate_by_name=True
    )

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
    model_config = ConfigDict(
        from_attributes=True, alias_generator=to_camel, populate_by_name=True
    )

    id: UUID
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    conversation_id: UUID
    sender: int = Field(
        ..., description="Message sender: 1=user, 2=assistant, 3=system"
    )
    content: str = Field(..., min_length=1, description="Message content")
