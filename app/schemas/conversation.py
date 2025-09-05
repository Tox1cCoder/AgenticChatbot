from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class ConversationBase(BaseModel):
    title: Optional[str] = Field(None, max_length=255, description="Conversation title")


class ConversationCreate(ConversationBase):
    user_id: int = Field(..., description="User ID who owns this conversation")


class ConversationUpdate(BaseModel):
    title: Optional[str] = Field(None, max_length=255, description="Conversation title")


class ConversationRead(ConversationBase):
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime


class ConversationInDB(ConversationRead):
    pass
