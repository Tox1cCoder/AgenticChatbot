"""
Generic API response wrapper
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_serializer

from app.utils.case_conversion import to_camel_case as to_camel

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """Generic API response wrapper"""

    success: bool = Field(..., description="Indicates if the request was successful")
    code: str | None = Field(None, description="Stable machine-readable error code")
    message: str = Field(..., description="Response message")
    data: T | None = Field(None, description="Response data")
    error: dict[str, Any] | None = Field(None, description="Error details")

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    @model_serializer(mode="wrap")
    def _omit_absent_code(self, handler):
        data = handler(self)
        if self.code is None:
            data.pop("code", None)
        return data
