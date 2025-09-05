from datetime import datetime
from typing import Optional, Literal

from pydantic import BaseModel, Field, ConfigDict


class MessageBase(BaseModel):
    content: str = Field(..., min_length=1, description="Message content")
    role: Literal["user", "assistant"] = Field(..., description="Message role: user or assistant")


class MessageCreate(MessageBase):
    conversation_id: int = Field(..., description="Conversation ID this message belongs to")
    user_id: int = Field(..., description="User ID who sent this message")


class MessageUpdate(BaseModel):
    content: Optional[str] = Field(None, min_length=1, description="Message content")


class MessageRead(MessageBase):
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    conversation_id: int
    user_id: int
    created_at: datetime


class MessageInDB(MessageRead):
    pass
