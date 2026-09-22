"""Repository for projects, their default agents, and conversation membership."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import UUID

from sqlalchemy import delete, func, select, update

from app.models.conversation import Conversation
from app.models.custom_agent import ConversationCustomAgent, CustomAgent
from app.models.project import Project, ProjectCustomAgent
from app.repositories.session_transport import RepositorySessionMixin


class ProjectRepository(RepositorySessionMixin):
    """Session-factory backed repository for projects."""

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

    def list_by_owner(self, owner_id: UUID) -> list[Project]:
        """All live projects for an owner, newest first."""
        with self.session_factory() as session:
            stmt = (
                select(Project)
                .where(Project.owner_id == owner_id, Project.deleted_at.is_(None))
                .order_by(Project.created_at.desc())
            )
            projects = list(session.execute(stmt).scalars().all())
            for project in projects:
                session.expunge(project)
            return projects

    @staticmethod
    def _get_owned_work(owner_id: UUID, project_id: UUID):
        statement = select(Project).where(
            Project.id == project_id,
            Project.owner_id == owner_id,
            Project.deleted_at.is_(None),
        )

        def work(session: Any) -> Project | None:
            project = session.execute(statement).scalar_one_or_none()
            if project is not None:
                session.expunge(project)
            return project

        return work

    def get_owned(self, owner_id: UUID, project_id: UUID) -> Project | None:
        """A live project if it exists and belongs to ``owner_id``."""
        return self._run(self._get_owned_work(owner_id, project_id))

    async def aget_owned(self, owner_id: UUID, project_id: UUID) -> Project | None:
        """Async twin of :meth:`get_owned`.

        Resolving a conversation's system instruction happens before the first
        token, where a sync engine checkout would block the event loop.
        """
        return await self._arun(self._get_owned_work(owner_id, project_id))

    def get_live(self, project_id: UUID) -> Project | None:
        """A live project regardless of owner.

        Used by ownership checks that need to tell "missing" apart from
        "someone else's" (see ``ProjectService.require_owned``).
        """
        with self.session_factory() as session:
            project = session.execute(
                select(Project).where(Project.id == project_id, Project.deleted_at.is_(None))
            ).scalar_one_or_none()
            if project is not None:
                session.expunge(project)
            return project

    def conversation_counts(self, owner_id: UUID) -> dict[UUID, int]:
        """Live conversation count per project, for the owner's project list."""
        with self.session_factory() as session:
            stmt = (
                select(Conversation.project_id, func.count(Conversation.id))
                .where(
                    Conversation.owner_id == owner_id,
                    Conversation.project_id.is_not(None),
                    Conversation.deleted_at.is_(None),
                )
                .group_by(Conversation.project_id)
            )
            return {project_id: count for project_id, count in session.execute(stmt).all()}

    def list_agents(self, owner_id: UUID, project_id: UUID) -> list[CustomAgent]:
        """Live custom agents in the project's default set, in attachment order.

        Returns the agent rows rather than ids so the service can convert them
        to ``CustomAgentRead``, matching the conversation agent route.
        """
        with self.session_factory() as session:
            stmt = (
                select(CustomAgent)
                .join(
                    ProjectCustomAgent,
                    ProjectCustomAgent.custom_agent_id == CustomAgent.id,
                )
                .where(
                    ProjectCustomAgent.project_id == project_id,
                    ProjectCustomAgent.owner_id == owner_id,
                    CustomAgent.deleted_at.is_(None),
                )
                .order_by(ProjectCustomAgent.agent_order.asc())
            )
            agents = list(session.execute(stmt).scalars().all())
            for agent in agents:
                session.expunge(agent)
            return agents

    # ---------------------------------------------------------------- writes

    def create(self, owner_id: UUID, fields: dict[str, Any]) -> Project:
        with self.session_factory() as session:
            project = Project(owner_id=owner_id, **fields)
            session.add(project)
            session.commit()
            session.refresh(project)
            session.expunge(project)
            return project

    def update(self, owner_id: UUID, project_id: UUID, fields: dict[str, Any]) -> Project | None:
        with self.session_factory() as session:
            project = session.execute(
                select(Project).where(
                    Project.id == project_id,
                    Project.owner_id == owner_id,
                    Project.deleted_at.is_(None),
                )
            ).scalar_one_or_none()
            if project is None:
                return None
            for key, value in fields.items():
                setattr(project, key, value)
            session.commit()
            session.refresh(project)
            session.expunge(project)
            return project

    def soft_delete_and_detach(self, owner_id: UUID, project_id: UUID) -> bool:
        """Soft-delete the project and release its conversations.

        Conversations survive as loose conversations. The detach is a hard
        write while the delete is soft, so restoring a project would restore
        it empty; there is no restore in this slice.
        """
        with self.session_factory() as session:
            project = session.execute(
                select(Project).where(
                    Project.id == project_id,
                    Project.owner_id == owner_id,
                    Project.deleted_at.is_(None),
                )
            ).scalar_one_or_none()
            if project is None:
                return False
            session.execute(
                update(Conversation)
                .where(Conversation.project_id == project_id)
                .values(project_id=None)
            )
            project.deleted_at = func.now()
            session.commit()
            return True

    def replace_agents(
        self, owner_id: UUID, project_id: UUID, custom_agent_ids: list[UUID]
    ) -> None:
        """Replace the project's default set with the given ordered ids.

        This does not reach into conversations already in the project; seeding
        happens at create and attach time only.
        """
        with self.session_factory() as session:
            session.execute(
                delete(ProjectCustomAgent).where(
                    ProjectCustomAgent.project_id == project_id,
                    ProjectCustomAgent.owner_id == owner_id,
                )
            )
            for order, custom_agent_id in enumerate(custom_agent_ids):
                session.add(
                    ProjectCustomAgent(
                        owner_id=owner_id,
                        project_id=project_id,
                        custom_agent_id=custom_agent_id,
                        agent_order=order,
                    )
                )
            session.commit()

    def seed_conversation_agents(
        self, owner_id: UUID, project_id: UUID, conversation_id: UUID
    ) -> int:
        """Copy the project's default agents onto a conversation, insert-if-absent.

        Returns the number of rows inserted. Never removes or reorders an
        existing attachment; new rows continue the conversation's ordering.
        Shared by conversation creation and by attach.
        """
        with self.session_factory() as session:
            project_agent_ids = list(
                session.execute(
                    select(ProjectCustomAgent.custom_agent_id)
                    .join(
                        CustomAgent,
                        CustomAgent.id == ProjectCustomAgent.custom_agent_id,
                    )
                    .where(
                        ProjectCustomAgent.project_id == project_id,
                        ProjectCustomAgent.owner_id == owner_id,
                        CustomAgent.deleted_at.is_(None),
                    )
                    .order_by(ProjectCustomAgent.agent_order.asc())
                )
                .scalars()
                .all()
            )
            if not project_agent_ids:
                return 0

            existing = set(
                session.execute(
                    select(ConversationCustomAgent.custom_agent_id).where(
                        ConversationCustomAgent.conversation_id == conversation_id
                    )
                )
                .scalars()
                .all()
            )
            next_order = (
                session.execute(
                    select(
                        func.coalesce(func.max(ConversationCustomAgent.agent_order), -1)
                    ).where(ConversationCustomAgent.conversation_id == conversation_id)
                ).scalar_one()
                + 1
            )

            inserted = 0
            for custom_agent_id in project_agent_ids:
                if custom_agent_id in existing:
                    continue
                session.add(
                    ConversationCustomAgent(
                        owner_id=owner_id,
                        conversation_id=conversation_id,
                        custom_agent_id=custom_agent_id,
                        agent_order=next_order,
                    )
                )
                next_order += 1
                inserted += 1
            session.commit()
            return inserted

    def set_conversation_project(self, conversation_id: UUID, project_id: UUID | None) -> None:
        """Point a conversation at a project, or release it when ``None``."""
        with self.session_factory() as session:
            session.execute(
                update(Conversation)
                .where(Conversation.id == conversation_id)
                .values(project_id=project_id)
            )
            session.commit()

    def conversation_project_id(self, conversation_id: UUID) -> UUID | None:
        """The project a conversation currently belongs to, if any."""
        with self.session_factory() as session:
            return session.execute(
                select(Conversation.project_id).where(Conversation.id == conversation_id)
            ).scalar_one_or_none()
