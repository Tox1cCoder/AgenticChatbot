from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class FeedbackCreate(BaseModel):
    message_id: UUID = Field(..., description="Message ID this feedback is for")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    comment: Optional[str] = Field(None, description="Optional comment")


class FeedbackUpdate(BaseModel):
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating from 1 to 5")
    comment: Optional[str] = Field(None, description="Optional comment")


class FeedbackRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    message_id: UUID
    user_id: UUID
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    comment: Optional[str] = Field(None, description="Optional comment")


class FeedbackInDB(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    message_id: UUID
    user_id: UUID
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    comment: Optional[str] = Field(None, description="Optional comment")
