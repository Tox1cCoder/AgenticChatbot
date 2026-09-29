from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.utils.case_conversion import to_camel_case as to_camel


class FeedbackCreate(BaseModel):
    message_id: UUID = Field(..., description="Message ID this feedback is for")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    comment: str | None = Field(None, description="Optional comment")

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class FeedbackUpdate(BaseModel):
    rating: int | None = Field(None, ge=1, le=5, description="Rating from 1 to 5")
    comment: str | None = Field(None, description="Optional comment")

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    @field_validator("rating")
    @classmethod
    def _rating_is_not_nullable(cls, value: int | None) -> int:
        # ``rating`` is NOT NULL. Updates apply ``exclude_unset``, so omitting it
        # is a no-op, but an explicit null reached the column and failed as an
        # IntegrityError (500). Validators skip omitted fields.
        if value is None:
            raise ValueError("rating cannot be null; omit it to leave it unchanged")
        return value


class FeedbackRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, alias_generator=to_camel, populate_by_name=True)

    id: UUID
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None
    message_id: UUID
    user_id: UUID
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    comment: str | None = Field(None, description="Optional comment")
