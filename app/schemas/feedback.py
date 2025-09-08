from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class FeedbackBase(BaseModel):
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    comment: Optional[str] = Field(None, description="Optional comment")


class FeedbackCreate(FeedbackBase):
    message_id: UUID = Field(..., description="Message ID this feedback is for")
    user_id: UUID = Field(..., description="User ID who provides this feedback")


class FeedbackUpdate(BaseModel):
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating from 1 to 5")
    comment: Optional[str] = Field(None, description="Optional comment")


class FeedbackRead(FeedbackBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    message_id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime


class FeedbackInDB(FeedbackRead):
    pass
