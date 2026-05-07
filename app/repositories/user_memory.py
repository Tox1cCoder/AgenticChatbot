"""Repository for agent-editable user memory items."""

from sqlalchemy import desc, select

from app.models.user_memory import UserMemory


class UserMemoryRepository:
    """Persistence layer for ``UserMemory`` rows."""

    def __init__(self, session_factory):
        self.session_factory = session_factory

    def create(self, *, user_id: str, content: str, source: str) -> UserMemory:
        with self.session_factory() as db:
            record = UserMemory(user_id=user_id, content=content, source=source)
            db.add(record)
            db.commit()
            db.refresh(record)
            return record

    def list_for_user(self, user_id: str, limit: int = 20) -> list[UserMemory]:
        with self.session_factory() as db:
            statement = (
                select(UserMemory)
                .where(
                    UserMemory.user_id == user_id,
                    UserMemory.deleted_at.is_(None),
                )
                .order_by(desc(UserMemory.created_at))
                .limit(limit)
            )
            return list(db.execute(statement).scalars().all())

    def delete_for_user(self, memory_id: str, user_id: str) -> bool:
        with self.session_factory() as db:
            statement = select(UserMemory).where(
                UserMemory.id == memory_id,
                UserMemory.user_id == user_id,
                UserMemory.deleted_at.is_(None),
            )
            record = db.execute(statement).scalars().first()
            if record is None:
                return False
            from datetime import datetime, timezone

            record.deleted_at = datetime.now(timezone.utc)
            db.commit()
            return True
