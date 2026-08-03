"""
Skills-related schemas for the client backend.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from client_backend.schemas.skill_installation import CamelModel


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


# The mutation request models below inherit CamelModel so one client can speak
# camelCase while the existing path-based tooling keeps its snake_case bodies
# (`populate_by_name`). `extra="forbid"` matters more than the aliases: these
# requests authorize running local code, so a misspelled `approveSetup` must be a
# 422 rather than a silently ignored field that leaves approval at its default.
class SkillInstallRequest(CamelModel):
    """Request to install a local skill bundle directory.

    A bundle is a directory containing exactly one discoverable ``SKILL.md``
    and any executable assets that skill owns.
    """

    source_path: str
    expected_source_hash: str | None = None
    approve_setup: bool = False
    replace_source_hash: str | None = None


class SkillInstallPreviewRequest(CamelModel):
    """Request a hash-bound, path-safe installation preview."""

    source_path: str


class SkillSetupRequest(CamelModel):
    """Approve preparation of a Python runtime for one discovered skill."""

    expected_source_hash: str
    approve_setup: bool = False


class SkillUninstallRequest(CamelModel):
    """Request to uninstall a previously installed local skill bundle."""

    name: str


class SkillSecretSetRequest(CamelModel):
    """Set one environment binding for the skill named in the route."""

    name: str
    value: str
