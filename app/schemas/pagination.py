from enum import Enum
from typing import TypeVar, Generic, List, Optional
from pydantic import BaseModel, Field


class OrderDirection(str, Enum):
    """Enum for order direction values"""

    ASC = "asc"
    DESC = "desc"


class ConversationOrderBy(str, Enum):
    """Enum for conversation order by fields"""

    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"
    TITLE = "title"


class MessageOrderBy(str, Enum):
    """Enum for message order by fields"""

    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"


class PaginationParams(BaseModel):
    """Base pagination parameters schema with order_by and order_direction restored"""

    page: int = Field(default=1, ge=1, description="Page number (1-based)")
    limit: int = Field(default=10, ge=1, le=100, description="Number of items per page")
    order_direction: OrderDirection = Field(
        default=OrderDirection.ASC, alias="orderDirection" 
    )

    class Config:
        populate_by_name = True


class ConversationPaginationParams(PaginationParams):
    """Pagination parameters for conversations with order_by restored"""

    order_by: ConversationOrderBy = Field(
        default=ConversationOrderBy.UPDATED_AT, alias="orderBy"
    )


class MessagePaginationParams(PaginationParams):
    """Pagination parameters for messages with order_by restored"""

    order_by: MessageOrderBy = Field(default=MessageOrderBy.CREATED_AT, alias="orderBy")

