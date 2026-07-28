"""Repository for custom agents and their per-conversation attachments."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.models.custom_agent import ConversationCustomAgent, CustomAgent
from app.repositories.session_transport import RepositorySessionMixin


class CustomAgentRepository(RepositorySessionMixin):
    """Session-factory backed repository for custom agents."""

    def __init__(
        self,
        session_factory: Callable[[], Any],
        async_session_factory: Callable[[], Any] | None = None,
    ):
        super().__init__(
            session_factory=session_factory,
            async_session_factory=async_session_factory,
        )

    # ----------------------------------------------------------------- reads

    def list_by_owner(self, owner_id: UUID) -> list[CustomAgent]:
        """All live (non-deleted) custom agents for an owner, newest first."""
        with self.session_factory() as session:
            stmt = (
                select(CustomAgent)
                .where(CustomAgent.owner_id == owner_id, CustomAgent.deleted_at.is_(None))
                .order_by(CustomAgent.created_at.desc())
            )
            return list(session.execute(stmt).scalars().all())

    def get_owned(self, owner_id: UUID, custom_agent_id: UUID) -> CustomAgent | None:
        """A single live agent if it exists and is owned by ``owner_id``."""
        with self.session_factory() as session:
            return self._get_owned(session, owner_id, custom_agent_id)

    def get_any(self, custom_agent_id: UUID) -> CustomAgent | None:
        """A single live agent regardless of owner (caller enforces ownership)."""
        with self.session_factory() as session:
            stmt = select(CustomAgent).where(
                CustomAgent.id == custom_agent_id, CustomAgent.deleted_at.is_(None)
            )
            return session.execute(stmt).scalar_one_or_none()

    def find_live_by_slug(
        self, owner_id: UUID, slug: str, *, exclude_id: UUID | None = None
    ) -> CustomAgent | None:
        """Find a live agent with the given slug for duplicate-name pre-checks."""
        with self.session_factory() as session:
            stmt = select(CustomAgent).where(
                CustomAgent.owner_id == owner_id,
                CustomAgent.slug == slug,
                CustomAgent.deleted_at.is_(None),
            )
            if exclude_id is not None:
                stmt = stmt.where(CustomAgent.id != exclude_id)
            return session.execute(stmt).scalar_one_or_none()

    # --------------------------------------------------------------- writes

    def create(self, owner_id: UUID, fields: dict[str, Any]) -> CustomAgent:
        with self.session_factory() as session:
            agent = CustomAgent(owner_id=owner_id, **fields)
            session.add(agent)
            session.commit()
            session.refresh(agent)
            session.expunge(agent)
            return agent

    def update(
        self, owner_id: UUID, custom_agent_id: UUID, fields: dict[str, Any]
    ) -> CustomAgent | None:
        with self.session_factory() as session:
            agent = self._get_owned(session, owner_id, custom_agent_id)
            if agent is None:
                return None
            for key, value in fields.items():
                setattr(agent, key, value)
            session.commit()
            session.refresh(agent)
            session.expunge(agent)
            return agent

    def delete_with_detach(self, owner_id: UUID, custom_agent_id: UUID) -> bool:
        """Detach from all conversations and soft-delete, in one transaction."""
        with self.session_factory() as session:
            agent = self._get_owned(session, owner_id, custom_agent_id)
            if agent is None:
                return False
            session.execute(
                delete(ConversationCustomAgent).where(
                    ConversationCustomAgent.custom_agent_id == custom_agent_id
                )
            )
            agent.deleted_at = func.now()
            session.commit()
            return True

    # ----------------------------------------------------------- attachments

    @staticmethod
    def _list_attachments_in_session(
        session: Session, owner_id: UUID, conversation_id: UUID
    ) -> list[tuple[ConversationCustomAgent, CustomAgent]]:
        """Attachment-listing body shared by both transports."""
        stmt = (
            select(ConversationCustomAgent, CustomAgent)
            .join(CustomAgent, CustomAgent.id == ConversationCustomAgent.custom_agent_id)
            .where(
                ConversationCustomAgent.conversation_id == conversation_id,
                ConversationCustomAgent.owner_id == owner_id,
                CustomAgent.deleted_at.is_(None),
            )
            .order_by(ConversationCustomAgent.agent_order.asc())
        )
        rows = session.execute(stmt).all()
        for _attachment, agent in rows:
            session.expunge(agent)
        return [(attachment, agent) for attachment, agent in rows]

    def list_attachments(
        self, owner_id: UUID, conversation_id: UUID
    ) -> list[tuple[ConversationCustomAgent, CustomAgent]]:
        """Ordered (attachment, agent) pairs for a conversation, live agents only."""
        return self._run(
            lambda session: self._list_attachments_in_session(session, owner_id, conversation_id)
        )

    async def alist_attachments(
        self, owner_id: UUID, conversation_id: UUID
    ) -> list[tuple[ConversationCustomAgent, CustomAgent]]:
        """Async twin of :meth:`list_attachments`."""
        return await self._arun(
            lambda session: self._list_attachments_in_session(session, owner_id, conversation_id)
        )

    def replace_attachments(
        self, owner_id: UUID, conversation_id: UUID, custom_agent_ids: list[UUID]
    ) -> None:
        """Replace the conversation's attachment set with the given ordered ids."""
        with self.session_factory() as session:
            session.execute(
                delete(ConversationCustomAgent).where(
                    ConversationCustomAgent.conversation_id == conversation_id,
                    ConversationCustomAgent.owner_id == owner_id,
                )
            )
            for order, custom_agent_id in enumerate(custom_agent_ids):
                session.add(
                    ConversationCustomAgent(
                        owner_id=owner_id,
                        conversation_id=conversation_id,
                        custom_agent_id=custom_agent_id,
                        agent_order=order,
                    )
                )
            session.commit()

    def list_attached_conversation_ids(self, custom_agent_id: UUID) -> list[UUID]:
        with self.session_factory() as session:
            stmt = select(ConversationCustomAgent.conversation_id).where(
                ConversationCustomAgent.custom_agent_id == custom_agent_id
            )
            return list(session.execute(stmt).scalars().all())

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _get_owned(session: Session, owner_id: UUID, custom_agent_id: UUID) -> CustomAgent | None:
        stmt = select(CustomAgent).where(
            CustomAgent.id == custom_agent_id,
            CustomAgent.owner_id == owner_id,
            CustomAgent.deleted_at.is_(None),
        )
        return session.execute(stmt).scalar_one_or_none()
