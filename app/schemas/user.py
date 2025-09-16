from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, EmailStr, ConfigDict


def to_camel(string: str) -> str:
    parts = string.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="User password")
    avatar_url: Optional[str] = Field(None, max_length=2048, description="Avatar URL")

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class UserUpdate(BaseModel):
    username: Optional[str] = Field(
        None, min_length=3, max_length=50, description="Username"
    )
    email: Optional[EmailStr] = Field(None, description="User email address")
    avatar_url: Optional[str] = Field(None, max_length=2048, description="Avatar URL")

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class UserRead(BaseModel):
    model_config = ConfigDict(
        from_attributes=True, alias_generator=to_camel, populate_by_name=True
    )

    id: UUID
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: EmailStr = Field(..., description="User email address")
    avatar_url: Optional[str] = None


class UserInDB(BaseModel):
    model_config = ConfigDict(
        from_attributes=True, alias_generator=to_camel, populate_by_name=True
    )

    id: UUID
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: EmailStr = Field(..., description="User email address")
    avatar_url: Optional[str] = None
    password_hash: str
