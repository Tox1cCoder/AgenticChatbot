"""
Token Response Schema
"""

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.core.config import settings
from app.utils.case_conversion import to_camel_case as to_camel


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Token Response Schema"""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = settings.access_token_expire_minutes * 60
    user_id: str

    model_config = ConfigDict(alias_generator=to_camel, validate_by_name=True)


class RefreshTokenResponse(BaseModel):
    accessToken: str = Field(alias="access_token")
    tokenType: str = Field(default="bearer", alias="token_type")
    expiresIn: int = Field(default=settings.access_token_expire_minutes * 60, alias="expires_in")

    model_config = ConfigDict(alias_generator=to_camel, validate_by_name=True)
