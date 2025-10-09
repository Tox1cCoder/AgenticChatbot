from enum import Enum
from typing import TypeVar, Generic, List, Optional
from pydantic import BaseModel, Field
from app.utils.case_conversion import to_camel_case as to_camel


class OrderDirection(str, Enum):
    """Enum for order direction values"""

    ASC = "asc"
    DESC = "desc"


class ConversationOrderBy(str, Enum):
    """Enum for conversation order by fields"""

    CREATED_AT = "createdAt"
    UPDATED_AT = "updatedAt"

    def to_snake_case(self) -> str:
        mapping = {"createdAt": "created_at", "updatedAt": "updated_at"}
        return mapping.get(self.value, self.value)


class MessageOrderBy(str, Enum):
    """Enum for message order by fields"""

    CREATED_AT = "createdAt"
    UPDATED_AT = "updatedAt"

    def to_snake_case(self) -> str:
        mapping = {"createdAt": "created_at", "updatedAt": "updated_at"}
        return mapping.get(self.value, self.value)


class PaginationParams(BaseModel):
    """Base pagination parameters schema"""

    page: int = Field(default=1, ge=1, description="Page number (1-based)")
    limit: int = Field(default=10, ge=1, le=100, description="Number of items per page")
    order_direction: OrderDirection = Field(
        default=OrderDirection.DESC, alias="orderDirection"
    )
    model_config = {"populate_by_name": True}


class ConversationPaginationParams(PaginationParams):
    """Pagination parameters for conversations"""

    order_by: ConversationOrderBy = Field(
        default=ConversationOrderBy.UPDATED_AT, alias="orderBy"
    )


class MessagePaginationParams(PaginationParams):
    """Pagination parameters for messages"""

    order_by: MessageOrderBy = Field(default=MessageOrderBy.CREATED_AT, alias="orderBy")
