from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class ConversationBase(BaseModel):
    title: str = Field(
        ..., min_length=1, max_length=255, description="Conversation title"
    )


class ConversationCreate(ConversationBase):
    pass


class ConversationUpdate(BaseModel):
    title: Optional[str] = Field(
        None, min_length=1, max_length=255, description="Conversation title"
    )


class ConversationRead(ConversationBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    owner_id: UUID
    created_at: datetime
    updated_at: datetime


class ConversationInDB(ConversationRead):
    pass
