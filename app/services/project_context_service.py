"""Resolves the system instruction a conversation runs with.

This is the single place project-derived prompt context is assembled. It reads
the project ownership-checked against the conversation's own owner, so editing
a project's instructions takes effect on the next turn of every conversation
in it, and a conversation can never inherit a project it does not own.

Long-term user memory is assembled here too, for the same reason: it is scoped
by the conversation's project, and every caller that needs a system
instruction already goes through this resolver. Injecting here is what makes
recall deterministic - the model does not have to decide to call a tool to
learn what it was told in an earlier conversation.
"""

from __future__ import annotations

import logging
from typing import Any

from app.ai.user_memory_tools import format_memories_for_prompt
from app.core.config import settings
from app.repositories.project import ProjectRepository
from app.utils.text_processing import compose_system_instruction

logger = logging.getLogger(__name__)


class ProjectContextService:
    """Combines project instructions, a conversation's persona, and memory."""

    def __init__(
        self,
        project_repository: ProjectRepository,
        user_memory_repository: Any | None = None,
    ):
        self.project_repository = project_repository
        self.user_memory_repository = user_memory_repository

    def resolve_system_instruction(self, conversation: Any) -> str | None:
        """The sanitized system instruction for ``conversation``, or ``None``.

        A conversation with no project and no memory costs no extra query and
        returns exactly what the pre-project code returned.

        Synchronous, for callers that are not on the event loop. The streaming
        turn uses :meth:`aresolve_system_instruction` instead.
        """
        if conversation is None:
            return None

        persona, project_id, owner_id = self._parts(conversation)

        project_instructions = None
        if project_id is not None:
            try:
                project = self.project_repository.get_owned(owner_id, project_id)
                if project is not None:
                    project_instructions = project.instructions
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to resolve project instructions: %s", exc)

        memory = None
        if self._memory_limit() > 0 and owner_id:
            try:
                memory = self.user_memory_repository.list_for_user(
                    str(owner_id),
                    limit=self._memory_limit(),
                    project_id=str(project_id) if project_id else None,
                )
            except Exception as exc:
                logger.warning("Failed to load user memory for prompt: %s", exc)

        return compose_system_instruction(
            project_instructions,
            persona,
            format_memories_for_prompt(memory or []) or None,
        )

    async def aresolve_system_instruction(self, conversation: Any) -> str | None:
        """Async twin of :meth:`resolve_system_instruction`.

        The streaming turn assembles its request on the event loop before the
        first token, where a sync engine checkout would block every other
        request in the process. Both lookups here have async twins for that
        reason; see ``tests/test_preflight_has_no_blocking_db.py``.
        """
        if conversation is None:
            return None

        persona, project_id, owner_id = self._parts(conversation)

        project_instructions = None
        if project_id is not None:
            try:
                project = await self.project_repository.aget_owned(owner_id, project_id)
                if project is not None:
                    project_instructions = project.instructions
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to resolve project instructions: %s", exc)

        memory = None
        if self._memory_limit() > 0 and owner_id:
            try:
                memory = await self.user_memory_repository.alist_for_user(
                    str(owner_id),
                    limit=self._memory_limit(),
                    project_id=str(project_id) if project_id else None,
                )
            except Exception as exc:
                logger.warning("Failed to load user memory for prompt: %s", exc)

        return compose_system_instruction(
            project_instructions,
            persona,
            format_memories_for_prompt(memory or []) or None,
        )

    @staticmethod
    def _parts(conversation: Any) -> tuple[Any, Any, Any]:
        return (
            getattr(conversation, "persona_prompt", None),
            getattr(conversation, "project_id", None),
            getattr(conversation, "owner_id", None),
        )

    def _memory_limit(self) -> int:
        """How many memories to inject, or 0 when recall is off."""
        if self.user_memory_repository is None:
            return 0
        if not getattr(settings, "enable_user_memory_tools", False):
            return 0
        return max(0, int(getattr(settings, "user_memory_max_prompt_items", 0) or 0))
