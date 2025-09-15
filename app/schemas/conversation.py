from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class ConversationCreate(BaseModel):
    title: str = Field(
        ..., min_length=1, max_length=255, description="Conversation title"
    )


class ConversationUpdate(BaseModel):
    title: Optional[str] = Field(
        None, min_length=1, max_length=255, description="Conversation title"
    )


class ConversationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    owner_id: UUID
    title: str = Field(
        ..., min_length=1, max_length=255, description="Conversation title"
    )


class ConversationInDB(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    owner_id: UUID
    title: str = Field(
        ..., min_length=1, max_length=255, description="Conversation title"
    )
