from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.utils.case_conversion import to_camel_case as to_camel


class DocumentImageBase(BaseModel):
    document_id: UUID
    chunk_id: UUID | None = None
    image_path: str
    image_caption: str | None = None
    page_number: int | None = None
    mime_type: str

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class DocumentImageCreate(DocumentImageBase):
    pass


class DocumentImageRead(DocumentImageBase):
    model_config = ConfigDict(from_attributes=True, alias_generator=to_camel, populate_by_name=True)

    id: UUID
    created_at: datetime


class DocumentImageUpdate(BaseModel):
    image_caption: str | None = None
    chunk_id: UUID | None = None

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
