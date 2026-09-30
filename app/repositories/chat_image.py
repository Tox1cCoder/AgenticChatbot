"""Repository for stored chat image records."""

from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.models.chat_image import ChatImage

_DIALECT_INSERTS = {"postgresql": postgresql_insert, "sqlite": sqlite_insert}


class ChatImageRepository:
    """Persistence for ChatImage rows."""

    def __init__(self, session_factory):
        self.session_factory = session_factory

    def create(self, data: dict[str, Any]) -> ChatImage:
        """Insert a row, or return the live row that already owns this content.

        ``uq_chat_images_user_sha256_live`` admits one live row per
        ``(user_id, sha256)``. Two workers persisting the same image can both
        miss ``get_by_user_and_sha`` and both get here; the insert skips on
        that index instead of raising, and the loser gets the winner's row.
        """
        with self.session_factory() as db:
            dialect = db.get_bind().dialect.name
            insert = _DIALECT_INSERTS.get(dialect)
            if insert is None:
                raise NotImplementedError(f"chat image insert has no {dialect!r} conflict clause")
            statement = (
                insert(ChatImage)
                .values(**data)
                .on_conflict_do_nothing(
                    index_elements=["user_id", "sha256"],
                    index_where=ChatImage.deleted_at.is_(None),
                )
                .returning(ChatImage.id)
            )
            inserted_id = db.execute(statement).scalar_one_or_none()
            db.commit()
            if inserted_id is not None:
                return db.get(ChatImage, inserted_id)
            existing = self._live_row(db, data["user_id"], data["sha256"])
            if existing is None:
                # The conflicting row was soft-deleted between the insert and
                # this read; the caller's retry will insert cleanly.
                raise RuntimeError("chat image conflict row disappeared before it could be read")
            return existing

    def get_for_user(self, image_id: UUID, user_id: UUID) -> ChatImage | None:
        with self.session_factory() as db:
            statement = select(ChatImage).where(
                ChatImage.id == image_id,
                ChatImage.user_id == user_id,
                ChatImage.deleted_at.is_(None),
            )
            return db.execute(statement).scalars().first()

    def get_by_user_and_sha(self, user_id: UUID, sha256: str) -> ChatImage | None:
        """Earliest non-deleted row owned by ``user_id`` for this content hash.

        Backs the storage layer's row-level idempotency: identical content
        re-stored by the same owner (e.g. a resumed run re-persisting a
        generated image) reuses this row instead of inserting a duplicate."""
        with self.session_factory() as db:
            return self._live_row(db, user_id, sha256)

    @staticmethod
    def _live_row(db, user_id: UUID, sha256: str) -> ChatImage | None:
        statement = (
            select(ChatImage)
            .where(
                ChatImage.user_id == user_id,
                ChatImage.sha256 == sha256,
                ChatImage.deleted_at.is_(None),
            )
            .order_by(ChatImage.created_at.asc())
        )
        return db.execute(statement).scalars().first()
