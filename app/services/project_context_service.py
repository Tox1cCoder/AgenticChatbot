"""Resolves the system instruction a conversation runs with.

This is the single place project-derived prompt context is assembled. It reads
the project ownership-checked against the conversation's own owner, so editing
a project's instructions takes effect on the next turn of every conversation
in it, and a conversation can never inherit a project it does not own.
"""

from __future__ import annotations

import logging
from typing import Any

from app.repositories.project import ProjectRepository
from app.utils.text_processing import compose_system_instruction

logger = logging.getLogger(__name__)


class ProjectContextService:
    """Combines a conversation's project instructions with its own persona."""

    def __init__(self, project_repository: ProjectRepository):
        self.project_repository = project_repository

    def resolve_system_instruction(self, conversation: Any) -> str | None:
        """The sanitized system instruction for ``conversation``, or ``None``.

        A conversation with no project costs no extra query and returns exactly
        what the pre-project code returned.
        """
        if conversation is None:
            return None

        persona = getattr(conversation, "persona_prompt", None)
        project_id = getattr(conversation, "project_id", None)
        if project_id is None:
            return compose_system_instruction(None, persona)

        owner_id = getattr(conversation, "owner_id", None)
        project_instructions = None
        try:
            project = self.project_repository.get_owned(owner_id, project_id)
            if project is not None:
                project_instructions = project.instructions
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to resolve project instructions: %s", exc)

        return compose_system_instruction(project_instructions, persona)
