"""Repository for offloaded tool result blob records."""

from typing import Any
from uuid import UUID

from sqlalchemy import select

from app.models.tool_result_blob import ToolResultBlob


class ToolResultBlobRepository:
    """Repository for ToolResultBlob persistence operations."""

    def __init__(self, session_factory):
        self.session_factory = session_factory

    def create(self, data: dict[str, Any]) -> ToolResultBlob:
        with self.session_factory() as db:  # type: Session
            record = ToolResultBlob(**data)
            db.add(record)
            db.commit()
            db.refresh(record)
            return record

    def get_for_user(self, blob_id: UUID, user_id: UUID) -> ToolResultBlob | None:
        with self.session_factory() as db:  # type: Session
            statement = select(ToolResultBlob).where(
                ToolResultBlob.id == blob_id,
                ToolResultBlob.user_id == user_id,
                ToolResultBlob.deleted_at.is_(None),
            )
            return db.execute(statement).scalars().first()
