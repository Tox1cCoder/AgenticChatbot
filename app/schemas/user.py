from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.utils.case_conversion import to_camel_case as to_camel


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="User password")
    avatar_url: str | None = Field(None, max_length=2048, description="Avatar URL")

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class UserUpdate(BaseModel):
    username: str | None = Field(None, min_length=3, max_length=50, description="Username")
    email: EmailStr | None = Field(None, description="User email address")
    avatar_url: str | None = Field(None, max_length=2048, description="Avatar URL")

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, alias_generator=to_camel, populate_by_name=True)

    id: UUID
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: EmailStr = Field(..., description="User email address")
    avatar_url: str | None = None


class UserInDB(BaseModel):
    model_config = ConfigDict(from_attributes=True, alias_generator=to_camel, populate_by_name=True)

    id: UUID
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: EmailStr = Field(..., description="User email address")
    avatar_url: str | None = None
    password_hash: str
