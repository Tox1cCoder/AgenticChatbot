from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict
from app.utils.case_conversion import to_camel_case as to_camel


class DocumentImageBase(BaseModel):
    document_id: UUID
    chunk_id: Optional[UUID] = None
    image_path: str
    image_caption: Optional[str] = None
    page_number: Optional[int] = None
    mime_type: str

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class DocumentImageCreate(DocumentImageBase):
    pass


class DocumentImageRead(DocumentImageBase):
    model_config = ConfigDict(
        from_attributes=True, alias_generator=to_camel, populate_by_name=True
    )

    id: UUID
    created_at: datetime


class DocumentImageUpdate(BaseModel):
    image_caption: Optional[str] = None
    chunk_id: Optional[UUID] = None

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
