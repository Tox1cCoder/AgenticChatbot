"""
Skills-related schemas for the client backend.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class SkillMetadata(BaseModel):
    """Metadata for a local skill."""

    name: str
    description: str
    version: str | None = None
    author: str | None = None
    tags: list[str] = Field(default_factory=list)


class LocalSkill(BaseModel):
    """Representation of a locally loaded skill."""

    name: str
    source_path: str
    metadata: SkillMetadata
    content: str  # The SKILL.md content
    enabled: bool = True
    loaded_at: datetime


class SkillSummary(BaseModel):
    """Summary of a skill for syncing to server."""

    name: str
    description: str
    enabled: bool
    source_hash: str  # Hash of the skill content for versioning


class SkillCatalog(BaseModel):
    """Catalog of locally available skills."""

    skills: list[LocalSkill]
    generated_at: datetime
    version: str


class SkillListResponse(BaseModel):
    """Response for listing skills."""

    skills: list[LocalSkill]
    total: int


class SkillToggleRequest(BaseModel):
    """Request to enable/disable a skill."""

    skill_name: str
    enabled: bool


class SkillToggleResponse(BaseModel):
    """Response for toggling a skill."""

    skill_name: str
    enabled: bool
    synced_to_server: bool = False


class SkillInstallRequest(BaseModel):
    """Request to install a local skill bundle directory.

    Provider-neutral by design: a bundle is just a directory containing
    ``SKILL.md`` and an optional ``skill.json``, so this request never names
    a specific integration (e.g. a calendar provider).
    """

    source_path: str


class SkillUninstallRequest(BaseModel):
    """Request to uninstall a previously installed local skill bundle."""

    name: str
