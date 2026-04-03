"""
Error Response Schema
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.utils.case_conversion import to_camel_case as to_camel


class ErrorResponse(BaseModel):
    """Error Response Schema"""

    success: bool = Field(False, description="Indicates the request failed")
    message: str = Field(..., description="Error message")
    code: str | None = Field(None, description="Error code")
    details: dict[str, Any] | None = Field(None, description="Additional error details")

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
