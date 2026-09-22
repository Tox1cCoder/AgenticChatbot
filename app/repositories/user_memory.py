"""Repository for agent-editable user memory items.

Memory is scoped to a project. ``project_id IS NULL`` means "global" - saved
from a conversation that belongs to no project - and global memories are
visible from every project. A conversation with no project sees only the
global ones, because ``project_id = NULL`` is never true in SQL.

Reads have async twins. Recall runs while a turn is being assembled, before
the first token, where a sync engine checkout would block the event loop.
"""

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import desc, or_, select
from sqlalchemy.orm import Session

from app.models.conversation import Conversation
from app.models.user_memory import UserMemory
from app.repositories.session_transport import RepositorySessionMixin


class UserMemoryRepository(RepositorySessionMixin):
    """Persistence layer for ``UserMemory`` rows."""

    def __init__(
        self,
        session_factory: Callable[[], Any],
        async_session_factory: Callable[[], Any] | None = None,
    ):
        super().__init__(
            session_factory=session_factory,
            async_session_factory=async_session_factory,
        )

    # ------------------------------------------------------------------
    # Query bodies - written once, run over either transport
    # ------------------------------------------------------------------

    @staticmethod
    def _list_work(
        user_id: str,
        limit: int,
        project_id: str | None,
    ) -> Callable[[Session], list[UserMemory]]:
        scope = (
            or_(UserMemory.project_id == project_id, UserMemory.project_id.is_(None))
            if project_id is not None
            else UserMemory.project_id.is_(None)
        )
        statement = (
            select(UserMemory)
            .where(
                UserMemory.user_id == user_id,
                UserMemory.deleted_at.is_(None),
                scope,
            )
            .order_by(desc(UserMemory.created_at))
            .limit(limit)
        )

        def work(session: Session) -> list[UserMemory]:
            rows = list(session.execute(statement).scalars().all())
            for row in rows:
                session.expunge(row)
            return rows

        return work

    @staticmethod
    def _resolve_project_work(
        user_id: str,
        conversation_id: str,
    ) -> Callable[[Session], str | None]:
        statement = select(Conversation.project_id).where(
            Conversation.id == conversation_id,
            Conversation.owner_id == user_id,
        )

        def work(session: Session) -> str | None:
            project_id = session.execute(statement).scalars().first()
            return str(project_id) if project_id else None

        return work

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def list_for_user(
        self,
        user_id: str,
        limit: int = 20,
        project_id: str | None = None,
    ) -> list[UserMemory]:
        """Memories visible from ``project_id``, most recent first.

        Passing a project returns that project's memories plus the global
        ones. Passing None returns only the global ones - a conversation
        outside any project must not see project work.
        """
        return self._run(self._list_work(user_id, limit, project_id))

    async def alist_for_user(
        self,
        user_id: str,
        limit: int = 20,
        project_id: str | None = None,
    ) -> list[UserMemory]:
        """Async twin of :meth:`list_for_user`."""
        return await self._arun(self._list_work(user_id, limit, project_id))

    def resolve_project_id(self, user_id: str, conversation_id: str | None) -> str | None:
        """The project a conversation belongs to, or None.

        Ownership-checked, so a conversation id the user does not own resolves
        to None (global) rather than to somebody else's project. Resolved at
        write time rather than at bind time so that moving a conversation
        between projects takes effect on the next save.
        """
        if not conversation_id:
            return None
        return self._run(self._resolve_project_work(user_id, conversation_id))

    async def aresolve_project_id(self, user_id: str, conversation_id: str | None) -> str | None:
        """Async twin of :meth:`resolve_project_id`."""
        if not conversation_id:
            return None
        return await self._arun(self._resolve_project_work(user_id, conversation_id))

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def create(
        self,
        *,
        user_id: str,
        content: str,
        source: str,
        project_id: str | None = None,
    ) -> UserMemory:
        with self.session_factory() as db:
            record = UserMemory(
                user_id=user_id,
                content=content,
                source=source,
                project_id=project_id,
            )
            db.add(record)
            db.commit()
            db.refresh(record)
            db.expunge(record)
            return record

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
            record.deleted_at = datetime.now(timezone.utc)
            db.commit()
            return True
