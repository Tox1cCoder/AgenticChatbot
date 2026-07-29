"""Async persistence for selected web-image references."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.web_image_reference import WebImageReference
from app.repositories.session_transport import RepositorySessionMixin


class WebImageReferenceRepository(RepositorySessionMixin):
    """Persist and authorize opaque references without blocking async callers."""

    def __init__(
        self,
        session_factory: Callable[[], Any],
        async_session_factory: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__(
            session_factory=session_factory,
            async_session_factory=async_session_factory,
        )

    async def acreate(self, data: dict[str, Any]) -> WebImageReference:
        def work(db: Session) -> WebImageReference:
            record = WebImageReference(**data)
            db.add(record)
            db.commit()
            db.refresh(record)
            return record

        return await self._arun(work)

    async def aget_for_user(
        self,
        image_id: UUID,
        user_id: UUID,
    ) -> WebImageReference | None:
        def work(db: Session) -> WebImageReference | None:
            statement = select(WebImageReference).where(
                WebImageReference.id == image_id,
                WebImageReference.user_id == user_id,
                WebImageReference.deleted_at.is_(None),
            )
            return db.execute(statement).scalars().first()

        return await self._arun(work)
