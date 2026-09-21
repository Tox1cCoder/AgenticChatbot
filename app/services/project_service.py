"""Service layer for projects: CRUD, ownership, default agents, membership."""

from __future__ import annotations

import logging
from uuid import UUID

from app.core.exceptions.project import (
    ProjectConversationNotFoundError,
    ProjectForbiddenError,
    ProjectNotFoundError,
)
from app.models.project import Project
from app.repositories.custom_agent import CustomAgentRepository
from app.repositories.project import ProjectRepository
from app.schemas.custom_agent import CustomAgentRead
from app.schemas.project import ProjectCreate, ProjectRead, ProjectUpdate
from app.utils.validation.conversation_validation import ConversationValidationUtils

logger = logging.getLogger(__name__)


class ProjectService:
    """Owner-scoped project operations."""

    def __init__(
        self,
        repository: ProjectRepository,
        custom_agent_repository: CustomAgentRepository,
        conversation_validation_utils: ConversationValidationUtils,
    ):
        self.repository = repository
        self.custom_agent_repository = custom_agent_repository
        self.conversation_validation_utils = conversation_validation_utils

    # ------------------------------------------------------------ ownership

    def require_owned(self, owner_id: UUID, project_id: UUID) -> Project:
        """Return the project, or raise 404 if missing and 403 if not the caller's.

        Distinguishing the two is deliberate and matches ``custom_agents``.
        """
        project = self.repository.get_live(project_id)
        if project is None:
            raise ProjectNotFoundError()
        if project.owner_id != owner_id:
            raise ProjectForbiddenError()
        return project

    # ----------------------------------------------------------------- CRUD

    def list_projects(self, owner_id: UUID) -> list[ProjectRead]:
        counts = self.repository.conversation_counts(owner_id)
        return [
            self._to_read(project, counts.get(project.id, 0))
            for project in self.repository.list_by_owner(owner_id)
        ]

    def create_project(self, owner_id: UUID, payload: ProjectCreate) -> ProjectRead:
        project = self.repository.create(
            owner_id,
            {
                "name": payload.name,
                "description": payload.description,
                "instructions": payload.instructions,
            },
        )
        return self._to_read(project, 0)

    def get_project(
        self, owner_id: UUID, project_id: UUID, *, include_agents: bool = False
    ) -> ProjectRead:
        project = self.require_owned(owner_id, project_id)
        counts = self.repository.conversation_counts(owner_id)
        agents = (
            [
                CustomAgentRead.model_validate(agent)
                for agent in self.repository.list_agents(owner_id, project_id)
            ]
            if include_agents
            else None
        )
        return self._to_read(project, counts.get(project.id, 0), agents)

    def update_project(
        self, owner_id: UUID, project_id: UUID, payload: ProjectUpdate
    ) -> ProjectRead:
        self.require_owned(owner_id, project_id)
        fields = payload.model_dump(exclude_unset=True)
        project = self.repository.update(owner_id, project_id, fields)
        if project is None:
            raise ProjectNotFoundError()
        counts = self.repository.conversation_counts(owner_id)
        return self._to_read(project, counts.get(project.id, 0))

    def delete_project(self, owner_id: UUID, project_id: UUID) -> None:
        self.require_owned(owner_id, project_id)
        if not self.repository.soft_delete_and_detach(owner_id, project_id):
            raise ProjectNotFoundError()

    # --------------------------------------------------------------- agents

    def list_agents(self, owner_id: UUID, project_id: UUID) -> list[CustomAgentRead]:
        self.require_owned(owner_id, project_id)
        return [
            CustomAgentRead.model_validate(agent)
            for agent in self.repository.list_agents(owner_id, project_id)
        ]

    def set_agents(
        self, owner_id: UUID, project_id: UUID, custom_agent_ids: list[UUID]
    ) -> list[CustomAgentRead]:
        """Replace the default set. Does not touch conversations already in the project."""
        self.require_owned(owner_id, project_id)
        for custom_agent_id in custom_agent_ids:
            if self.custom_agent_repository.get_owned(owner_id, custom_agent_id) is None:
                raise ProjectForbiddenError(
                    detail=f"Custom agent {custom_agent_id} is not available to this user"
                )
        self.repository.replace_agents(owner_id, project_id, custom_agent_ids)
        return self.list_agents(owner_id, project_id)

    # ----------------------------------------------------------- membership

    def attach_conversation(self, owner_id: UUID, project_id: UUID, conversation_id: UUID) -> None:
        """Move a conversation into the project and seed the project's agents.

        Attaching a conversation that already belongs to another project is a
        move, not an error.
        """
        self.require_owned(owner_id, project_id)
        self.conversation_validation_utils.validate_conversation_access(owner_id, conversation_id)
        self.repository.set_conversation_project(conversation_id, project_id)
        self.repository.seed_conversation_agents(owner_id, project_id, conversation_id)

    def detach_conversation(self, owner_id: UUID, project_id: UUID, conversation_id: UUID) -> None:
        """Release a conversation from the project, keeping its seeded agents."""
        self.require_owned(owner_id, project_id)
        self.conversation_validation_utils.validate_conversation_access(owner_id, conversation_id)
        if self.repository.conversation_project_id(conversation_id) != project_id:
            raise ProjectConversationNotFoundError()
        self.repository.set_conversation_project(conversation_id, None)

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _to_read(
        project: Project,
        conversation_count: int,
        agents: list[CustomAgentRead] | None = None,
    ) -> ProjectRead:
        return ProjectRead(
            id=project.id,
            created_at=project.created_at,
            updated_at=project.updated_at,
            deleted_at=project.deleted_at,
            owner_id=project.owner_id,
            name=project.name,
            description=project.description,
            instructions=project.instructions,
            conversation_count=conversation_count,
            custom_agents=agents,
        )
