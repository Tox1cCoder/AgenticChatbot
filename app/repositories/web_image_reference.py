"""Async persistence for selected web-image references."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import func, select, update
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

    async def aget_pending_for_user(
        self,
        image_id: UUID,
        user_id: UUID,
        *,
        conversation_id: UUID,
    ) -> WebImageReference | None:
        def work(db: Session) -> WebImageReference | None:
            statement = select(WebImageReference).where(
                WebImageReference.id == image_id,
                WebImageReference.user_id == user_id,
                WebImageReference.conversation_id == conversation_id,
                WebImageReference.lifecycle_state == "pending",
                WebImageReference.deleted_at.is_(None),
            )
            return db.execute(statement).scalars().first()

        return await self._arun(work)

    async def amark_selected(
        self,
        image_ids: list[UUID] | tuple[UUID, ...],
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> int:
        return await self._aupdate_lifecycle(
            image_ids,
            user_id=user_id,
            conversation_id=conversation_id,
            state="selected",
        )

    async def arelease_many(
        self,
        image_ids: list[UUID] | tuple[UUID, ...],
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> int:
        return await self._aupdate_lifecycle(
            image_ids,
            user_id=user_id,
            conversation_id=conversation_id,
            state="released",
        )

    async def _aupdate_lifecycle(
        self,
        image_ids: list[UUID] | tuple[UUID, ...],
        *,
        user_id: UUID,
        conversation_id: UUID,
        state: str,
    ) -> int:
        if not image_ids:
            return 0

        def work(db: Session) -> int:
            values: dict[str, Any] = {
                "lifecycle_state": state,
                "expires_at": None,
            }
            if state == "released":
                values["deleted_at"] = func.now()
            statement = (
                update(WebImageReference)
                .where(
                    WebImageReference.id.in_(image_ids),
                    WebImageReference.user_id == user_id,
                    WebImageReference.conversation_id == conversation_id,
                    WebImageReference.lifecycle_state == "pending",
                    WebImageReference.deleted_at.is_(None),
                )
                .values(**values)
            )
            result = db.execute(statement)
            db.commit()
            return int(result.rowcount or 0)

        return await self._arun(work)

    async def arelease_expired(self, now: datetime) -> int:
        def work(db: Session) -> int:
            statement = (
                update(WebImageReference)
                .where(
                    WebImageReference.lifecycle_state == "pending",
                    WebImageReference.expires_at.is_not(None),
                    WebImageReference.expires_at <= now,
                    WebImageReference.deleted_at.is_(None),
                )
                .values(
                    lifecycle_state="released",
                    expires_at=None,
                    deleted_at=func.now(),
                )
            )
            result = db.execute(statement)
            db.commit()
            return int(result.rowcount or 0)

        return await self._arun(work)
