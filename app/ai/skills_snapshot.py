"""Read-only helpers for inspecting installed server repo skills."""

from __future__ import annotations

import logging
from typing import Any

from .skills_registry import get_server_skills_registry

logger = logging.getLogger(__name__)


def _serialize_skill(skill: Any) -> dict[str, Any]:
    return {
        "name": skill.name,
        "description": skill.description,
        "enabled": skill.enabled,
        "folderPath": skill.folder_path,
    }


def list_repo_skills_for_demo() -> dict[str, Any]:
    """Return a demo-friendly snapshot of installed server repo skills."""
    skills = sorted(
        get_server_skills_registry().get_all_skills(),
        key=lambda item: item.name.lower(),
    )
    return {
        "skills": [_serialize_skill(skill) for skill in skills],
        "totalCount": len(skills),
        "enabledCount": sum(1 for skill in skills if skill.enabled),
    }


def get_repo_skill_detail_for_demo(name: str) -> dict[str, Any] | None:
    """Return one installed server repo skill, including its activation content."""
    try:
        skill = get_server_skills_registry().get_skill(name)
    except KeyError:
        return None
    except Exception as exc:
        logger.warning("Failed to load repo skill detail for %s: %s", name, exc)
        return None

    detail = _serialize_skill(skill)
    detail["content"] = skill.content
    return detail


def reload_repo_skills_for_demo() -> dict[str, Any] | None:
    """Rescan the repo skills directory for the local demo process."""
    try:
        registry = get_server_skills_registry()
        registry.reload()
        skills = registry.get_all_skills()
    except Exception as exc:
        logger.warning("Failed to reload repo skills for demo: %s", exc)
        return None

    enabled_count = sum(1 for skill in skills if skill.enabled)
    return {"message": f"Repo skills reloaded: {len(skills)} found ({enabled_count} enabled)"}
