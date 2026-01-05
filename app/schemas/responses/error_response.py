"""
Error Response Schema
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from app.utils.case_conversion import to_camel_case as to_camel


class ErrorResponse(BaseModel):
    """Error Response Schema"""

    success: bool = Field(False, description="Indicates the request failed")
    message: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
