"""
Pagination utilities for repository layer
"""

import math
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from app.utils.case_conversion import to_camel_case as to_camel

T = TypeVar("T")


class PaginationMeta(BaseModel):
    """Holds pagination metadata"""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    total: int
    per_page: int = Field(alias="perPage")
    current_page: int = Field(alias="currentPage")
    last_page: int = Field(alias="lastPage")

    @classmethod
    def calculate(cls, total: int, per_page: int, current_page: int) -> "PaginationMeta":
        """Calculate pagination metadata"""
        last_page = math.ceil(total / per_page) if per_page > 0 else 1
        return cls(
            total=total,
            per_page=per_page,
            current_page=current_page,
            last_page=max(last_page, 1),
        )


class Paginator(BaseModel, Generic[T]):
    """Handles pagination logic, returns items and meta"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    items: list[T]
    meta: PaginationMeta

    @classmethod
    def create(cls, items: list[T], total: int, current_page: int, per_page: int) -> "Paginator[T]":
        """Create paginator instance with items and calculated metadata"""
        meta = PaginationMeta.calculate(total, per_page, current_page)
        return cls(items=items, meta=meta)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "meta": self.meta.model_dump(),
            "items": [
                item.model_dump() if hasattr(item, "model_dump") else item for item in self.items
            ],
        }
