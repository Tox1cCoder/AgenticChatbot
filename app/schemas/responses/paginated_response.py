"""
Paginated API response schemas
"""

from typing import Generic, TypeVar, Dict, Any
from pydantic import BaseModel

from app.repositories.utils.pagination import PaginationMeta

T = TypeVar("T")


class PaginatedData(BaseModel, Generic[T]):
    """Container for paginated data with metadata"""

    meta: PaginationMeta
    items: list[T]


class PaginatedApiResponse(BaseModel, Generic[T]):
    """API response format for paginated data"""

    success: bool = True
    data: PaginatedData[T]
    message: str = ""

    @classmethod
    def from_paginator(cls, paginator, message: str = "") -> "PaginatedApiResponse[T]":
        """Create response from Paginator object"""
        return cls(
            success=True,
            data=PaginatedData(meta=paginator.meta, items=paginator.items),
            message=message,
        )
