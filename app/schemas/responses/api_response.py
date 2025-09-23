"""
Generic API response wrapper
"""

from typing import Generic, TypeVar, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

T = TypeVar("T")


def to_camel(string: str) -> str:
    parts = string.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


class ApiResponse(BaseModel, Generic[T]):
    """Generic API response wrapper"""

    success: bool = Field(..., description="Indicates if the request was successful")
    message: str = Field(..., description="Response message")
    data: Optional[T] = Field(None, description="Response data")
    error: Optional[Dict[str, Any]] = Field(None, description="Error details")

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
