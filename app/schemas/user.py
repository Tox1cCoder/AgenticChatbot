from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, EmailStr, ConfigDict


class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: EmailStr = Field(..., description="User email address")


class UserCreate(UserBase):
    password: str = Field(..., min_length=8, description="User password")
    avatar_url: Optional[str] = Field(None, max_length=2048, description="Avatar URL")


class UserUpdate(BaseModel):
    username: Optional[str] = Field(
        None, min_length=3, max_length=50, description="Username"
    )
    email: Optional[EmailStr] = Field(None, description="User email address")
    avatar_url: Optional[str] = Field(None, max_length=2048, description="Avatar URL")


class UserRead(UserBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    avatar_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class UserInDB(UserRead):
    password_hash: str
