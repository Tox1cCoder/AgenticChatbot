"""Service layer for Skills operations."""

import logging
from typing import Dict, Any

from app.ai.skills_registry import SkillsRegistry
from app.core.exceptions.skills import SkillNotFoundError

logger = logging.getLogger(__name__)


class SkillsService:
    """Business logic for skill management — mirrors MCPService."""

    def __init__(self, registry: SkillsRegistry):
        self.registry = registry

    async def list_skills(self) -> Dict[str, Any]:
        """Returns all skills with enabled state and metadata."""
        all_skills = self.registry.get_all_skills()

        skills_list = [
            {
                "name": s.name,
                "description": s.description,
                "enabled": s.enabled,
                "folder_path": s.folder_path,
            }
            for s in all_skills
        ]

        enabled_count = sum(1 for s in all_skills if s.enabled)

        return {
            "skills": skills_list,
            "total_count": len(skills_list),
            "enabled_count": enabled_count,
        }

    async def get_skill(self, name: str) -> Dict[str, Any]:
        """Returns full detail for one skill, raises SkillNotFoundError."""
        try:
            skill = self.registry.get_skill(name)
        except KeyError:
            raise SkillNotFoundError(name)

        return {
            "name": skill.name,
            "description": skill.description,
            "enabled": skill.enabled,
            "folder_path": skill.folder_path,
            "content": skill.content,
        }

    async def toggle_skill(self, name: str, enabled: bool) -> Dict[str, Any]:
        """Enable or disable a skill. Returns confirmation message."""
        try:
            self.registry.toggle_skill(name, enabled)
        except KeyError:
            raise SkillNotFoundError(name)

        action = "enabled" if enabled else "disabled"
        return {"message": f"Skill '{name}' {action}"}

    async def reload_skills(self) -> Dict[str, Any]:
        """Trigger a disk rescan (hot-reload after manually adding a skill)."""
        self.registry.reload()
        all_skills = self.registry.get_all_skills()
        enabled_count = sum(1 for s in all_skills if s.enabled)
        return {
            "message": f"Skills reloaded: {len(all_skills)} found ({enabled_count} enabled)"
        }
