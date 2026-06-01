"""Internal server skill source resolution for prompt binding and activation."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any
from uuid import UUID

from app.services.client_device_service import ClientDeviceService

from .skills_registry import get_server_skills_registry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedSkill:
    """A source-aware skill resolved for a specific execution scope."""

    name: str
    description: str
    source: str
    lookup_name: str
    category: str | None = None
    tags: tuple[str, ...] = ()
    content_length: int | None = None
    bound_user_id: str | None = None
    bound_device_id: str | None = None
    bound_session_id: str | None = None

    def to_summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "lookup_name": self.lookup_name,
            "description": self.description,
            "category": self.category,
            "tags": list(self.tags),
            "content_length": self.content_length,
            "source": self.source,
        }


def get_bound_device_session(*, user_id: str | None, device_id: str | None):
    """Return the active sidecar session for the request scope, if any."""
    if not user_id or not device_id:
        return None

    try:
        device_uuid = UUID(str(device_id))
    except Exception:
        logger.warning("Invalid device_id passed to client skill lookup: %s", device_id)
        return None

    session = ClientDeviceService.lookup_active_session(device_uuid)
    if session is None or str(session.user_id) != str(user_id):
        return None

    return session


def _list_server_skills() -> list[ResolvedSkill]:
    try:
        active_skills = get_server_skills_registry().get_active_skills()
    except Exception as exc:
        logger.warning("Failed to load server-owned repo skills: %s", exc)
        return []

    return [
        ResolvedSkill(
            name=skill.name,
            description=skill.description,
            source="server",
            lookup_name=skill.name,
            category=getattr(skill, "category", None),
            tags=tuple(getattr(skill, "tags", ()) or ()),
            content_length=len(skill.content) if skill.content else 0,
        )
        for skill in active_skills
    ]


def _list_client_skills(*, user_id: str | None, device_id: str | None) -> list[ResolvedSkill]:
    session = get_bound_device_session(user_id=user_id, device_id=device_id)
    if session is None or not session.skill_catalog:
        return []

    raw_skills = session.skill_catalog.get("skills", [])
    if not isinstance(raw_skills, list):
        return []

    skills: list[ResolvedSkill] = []
    for raw_skill in raw_skills:
        if not isinstance(raw_skill, dict):
            continue

        name = str(raw_skill.get("name") or "").strip()
        if not name or not bool(raw_skill.get("enabled", True)):
            continue

        tags = raw_skill.get("tags") or []
        if not isinstance(tags, list):
            tags = []

        skills.append(
            ResolvedSkill(
                name=name,
                description=str(raw_skill.get("description") or "").strip(),
                source="client",
                lookup_name=name,
                category=raw_skill.get("category"),
                tags=tuple(str(tag) for tag in tags if tag),
                content_length=raw_skill.get("content_length"),
                bound_user_id=str(user_id or ""),
                bound_device_id=str(device_id or ""),
                bound_session_id=session.session_id,
            )
        )

    return skills


def _normalize_lookup_names(skills: list[ResolvedSkill]) -> list[ResolvedSkill]:
    counts = Counter(skill.name for skill in skills)
    normalized: list[ResolvedSkill] = []
    for skill in skills:
        lookup_name = skill.name
        if skill.name and counts[skill.name] > 1:
            lookup_name = f"{skill.source}:{skill.name}"
        normalized.append(replace(skill, lookup_name=lookup_name))
    return normalized


def _ref_matches_skill(ref: dict[str, Any], skill: ResolvedSkill) -> bool:
    if str(ref.get("source") or "") != skill.source:
        return False
    lookup_name = ref.get("lookup_name")
    name = ref.get("name")
    return skill.lookup_name in (lookup_name, name) or skill.name in (lookup_name, name)


def filter_skills_by_refs(
    skills: list[ResolvedSkill],
    allowed_skill_refs: list[dict[str, Any]] | None,
) -> list[ResolvedSkill]:
    """Restrict skills to a custom agent's selected refs (no-op when None)."""
    if allowed_skill_refs is None:
        return skills
    return [s for s in skills if any(_ref_matches_skill(ref, s) for ref in allowed_skill_refs)]


def list_resolved_skills(
    *,
    user_id: str | None,
    device_id: str | None,
    allowed_skill_refs: list[dict[str, Any]] | None = None,
) -> list[ResolvedSkill]:
    """Return all runtime-visible skills for this execution scope.

    When ``allowed_skill_refs`` is provided (custom agents), the result is
    restricted to skills matching one of the selected refs.
    """
    combined = _list_server_skills()
    combined.extend(_list_client_skills(user_id=user_id, device_id=device_id))
    normalized = _normalize_lookup_names(combined)
    restricted = filter_skills_by_refs(normalized, allowed_skill_refs)
    return sorted(restricted, key=lambda item: item.lookup_name.lower())


def get_available_skill_summaries(
    *,
    user_id: str | None,
    device_id: str | None,
    allowed_skill_refs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return prompt-safe skill summaries for the active request scope."""
    return [
        skill.to_summary()
        for skill in list_resolved_skills(
            user_id=user_id, device_id=device_id, allowed_skill_refs=allowed_skill_refs
        )
    ]


def resolve_skill_reference(
    *,
    skill_name: str,
    user_id: str | None,
    device_id: str | None,
    allowed_skill_refs: list[dict[str, Any]] | None = None,
) -> tuple[ResolvedSkill | None, str | None]:
    """Resolve a model-selected skill name to one explicit skill source."""
    available_skills = list_resolved_skills(
        user_id=user_id, device_id=device_id, allowed_skill_refs=allowed_skill_refs
    )
    available_names = [skill.lookup_name for skill in available_skills]

    exact_lookup_match = next(
        (skill for skill in available_skills if skill.lookup_name == skill_name), None
    )
    if exact_lookup_match is not None:
        return exact_lookup_match, None

    raw_matches = [skill for skill in available_skills if skill.name == skill_name]
    if len(raw_matches) == 1:
        return raw_matches[0], None

    if len(raw_matches) > 1:
        options = ", ".join(skill.lookup_name for skill in raw_matches)
        return None, f"Error: skill '{skill_name}' is ambiguous. Use one of: {options}"

    return (
        None,
        f"Error: skill '{skill_name}' not found. "
        f"Available skills: {', '.join(available_names) if available_names else '(none)'}",
    )
