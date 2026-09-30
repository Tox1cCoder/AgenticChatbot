from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.core.security.password import BCRYPT_MAX_PASSWORD_BYTES
from app.utils.case_conversion import to_camel_case as to_camel


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="User password")
    avatar_url: str | None = Field(None, max_length=2048, description="Avatar URL")

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    @field_validator("password")
    @classmethod
    def _fits_bcrypt(cls, value: str) -> str:
        # A byte limit, not a character one: bcrypt hashes UTF-8 bytes, and a
        # longer password made hash_password raise, which surfaced as a 500.
        if len(value.encode("utf-8")) > BCRYPT_MAX_PASSWORD_BYTES:
            raise ValueError(
                f"password must be at most {BCRYPT_MAX_PASSWORD_BYTES} bytes when UTF-8 encoded"
            )
        return value


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
    token_version: int = 0
