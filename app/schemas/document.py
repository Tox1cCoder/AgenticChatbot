"""
Document-related Pydantic schemas for API requests and responses.
"""

from datetime import datetime
from enum import IntEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class DocumentStatus(IntEnum):
    """Document processing status"""

    PROCESSING = 1
    READY = 2
    FAILED = 3


class DocumentBase(BaseModel):
    """Base schema for documents"""

    filename: str
    file_type: str
    status: int = DocumentStatus.PROCESSING.value


class DocumentCreate(DocumentBase):
    """Schema for creating a new document"""

    conversation_id: UUID


class DocumentUpdate(BaseModel):
    """Schema for updating a document"""

    filename: str | None = None
    file_type: str | None = None
    status: int | None = None


class DocumentResponse(DocumentBase):
    """Schema for document responses"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    conversation_id: UUID
    upload_time: datetime


class DocumentListResponse(BaseModel):
    """Schema for paginated document lists"""

    documents: list[DocumentResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
