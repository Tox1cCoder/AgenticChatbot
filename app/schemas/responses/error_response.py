"""
Error Response Schema
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


def to_camel(string: str) -> str:
    parts = string.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


class ErrorResponse(BaseModel):
    """Error Response Schema"""

    success: bool = Field(False, description="Indicates the request failed")
    message: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, alias="errorCode", description="Error code")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
