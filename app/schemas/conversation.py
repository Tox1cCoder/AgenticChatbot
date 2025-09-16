from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


def to_camel(string: str) -> str:
    parts = string.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


class ConversationCreate(BaseModel):
    title: str = Field(
        ..., min_length=1, max_length=255, description="Conversation title"
    )

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ConversationUpdate(BaseModel):
    title: Optional[str] = Field(
        None, min_length=1, max_length=255, description="Conversation title"
    )

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ConversationRead(BaseModel):
    model_config = ConfigDict(
        from_attributes=True, alias_generator=to_camel, populate_by_name=True
    )

    id: UUID
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    owner_id: UUID
    title: str = Field(
        ..., min_length=1, max_length=255, description="Conversation title"
    )


class ConversationInDB(BaseModel):
    model_config = ConfigDict(
        from_attributes=True, alias_generator=to_camel, populate_by_name=True
    )

    id: UUID
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    owner_id: UUID
    title: str = Field(
        ..., min_length=1, max_length=255, description="Conversation title"
    )
