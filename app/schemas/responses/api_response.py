"""
Generic API response wrapper
"""

from typing import Generic, TypeVar, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from app.utils.case_conversion import to_camel_case as to_camel

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """Generic API response wrapper"""

    success: bool = Field(..., description="Indicates if the request was successful")
    message: str = Field(..., description="Response message")
    data: Optional[T] = Field(None, description="Response data")
    error: Optional[Dict[str, Any]] = Field(None, description="Error details")

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
