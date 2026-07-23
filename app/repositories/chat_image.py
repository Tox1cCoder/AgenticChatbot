"""Repository for stored chat image records."""

from typing import Any
from uuid import UUID

from sqlalchemy import select

from app.models.chat_image import ChatImage


class ChatImageRepository:
    """Persistence for ChatImage rows."""

    def __init__(self, session_factory):
        self.session_factory = session_factory

    def create(self, data: dict[str, Any]) -> ChatImage:
        with self.session_factory() as db:
            record = ChatImage(**data)
            db.add(record)
            db.commit()
            db.refresh(record)
            return record

    def get_for_user(self, image_id: UUID, user_id: UUID) -> ChatImage | None:
        with self.session_factory() as db:
            statement = select(ChatImage).where(
                ChatImage.id == image_id,
                ChatImage.user_id == user_id,
                ChatImage.deleted_at.is_(None),
            )
            return db.execute(statement).scalars().first()
