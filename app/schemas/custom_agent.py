"""Pydantic schemas for per-user custom agents.

Field names are snake_case (matching the API payload examples in the plan).
A camelCase alias generator with ``populate_by_name=True`` is applied so both
snake_case and camelCase request bodies are accepted, mirroring the existing
conversation schemas.
"""

from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.utils.case_conversion import to_camel_case as to_camel

CUSTOM_AGENT_PREFIX = "custom_agent:"
CUSTOM_MODEL_AGENT_KEY = "custom"


def runtime_agent_id_for(custom_agent_id: UUID | str) -> str:
    """Build the runtime agent id (``custom_agent:<uuid>``) for a custom agent."""
    return f"{CUSTOM_AGENT_PREFIX}{custom_agent_id}"


class _CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


# --------------------------------------------------------------------------- #
# Tool / skill references
# --------------------------------------------------------------------------- #


class ServerDefaultToolRef(_CamelModel):
    """Legacy reference to a backend MCP tool from the server-default catalog."""

    type: Literal["server_default"] = "server_default"
    qualified_tool_id: str = Field(..., min_length=1)
    display_metadata: dict[str, Any] | None = None


class ServerMcpToolRef(_CamelModel):
    """Exact reference to a backend MCP server tool."""

    type: Literal["server_mcp"] = "server_mcp"
    server_name: str = Field(..., min_length=1)
    tool_name: str = Field(..., min_length=1)
    qualified_tool_id: str = Field(..., min_length=1)
    display_metadata: dict[str, Any] | None = None


class ClientToolRef(_CamelModel):
    """Exact reference to a tool exposed by a specific client device/session catalog."""

    type: Literal["client"] = "client"
    device_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    catalog_version: str = Field(..., min_length=1)
    tool_instance_id: str = Field(..., min_length=1)
    server_name: str = Field(..., min_length=1)
    qualified_tool_id: str = Field(..., min_length=1)
    tool_name: str = Field(..., min_length=1)
    display_metadata: dict[str, Any] | None = None

    @field_validator("catalog_version", mode="before")
    @classmethod
    def _coerce_catalog_version(cls, value: Any) -> str:
        return str(value)


CustomAgentToolRef = Annotated[
    ServerDefaultToolRef | ServerMcpToolRef | ClientToolRef,
    Field(discriminator="type"),
]


class CustomAgentSkillRef(_CamelModel):
    """Exact reference to a selectable skill (server or active client skill)."""

    source: Literal["server", "client"]
    lookup_name: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    display_metadata: dict[str, Any] | None = None


# --------------------------------------------------------------------------- #
# Create / update / read
# --------------------------------------------------------------------------- #


def _require_non_empty(value: str | None, field_name: str) -> str | None:
    if value is None:
        return value
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(f"{field_name} must not be empty")
    return trimmed


class CustomAgentCreate(_CamelModel):
    name: str = Field(..., max_length=255)
    description: str | None = Field(default=None, max_length=2000)
    prompt: str
    provider_type: str = Field(..., min_length=1, max_length=64)
    model: str = Field(..., min_length=1, max_length=255)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    reasoning_effort: str | None = Field(default=None, max_length=32)
    tool_refs: list[CustomAgentToolRef] = Field(default_factory=list)
    skill_refs: list[CustomAgentSkillRef] = Field(default_factory=list)
    enabled: bool = True

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        return _require_non_empty(value, "name")  # type: ignore[return-value]

    @field_validator("prompt")
    @classmethod
    def _validate_prompt(cls, value: str) -> str:
        return _require_non_empty(value, "prompt")  # type: ignore[return-value]


class CustomAgentUpdate(_CamelModel):
    """Partial update. Only fields explicitly set are applied (``exclude_unset``)."""

    name: str | None = Field(default=None, max_length=255)
    description: str | None = Field(default=None, max_length=2000)
    prompt: str | None = None
    provider_type: str | None = Field(default=None, max_length=64)
    model: str | None = Field(default=None, max_length=255)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    reasoning_effort: str | None = Field(default=None, max_length=32)
    tool_refs: list[CustomAgentToolRef] | None = None
    skill_refs: list[CustomAgentSkillRef] | None = None
    enabled: bool | None = None

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str | None) -> str | None:
        return _require_non_empty(value, "name")

    @field_validator("prompt")
    @classmethod
    def _validate_prompt(cls, value: str | None) -> str | None:
        return _require_non_empty(value, "prompt")


class CustomAgentRead(_CamelModel):
    model_config = ConfigDict(from_attributes=True, alias_generator=to_camel, populate_by_name=True)

    id: UUID
    created_at: datetime
    updated_at: datetime
    owner_id: UUID
    name: str
    slug: str
    description: str | None = None
    prompt: str
    provider_type: str
    model: str
    temperature: float | None = None
    reasoning_effort: str | None = None
    tool_refs: list[dict[str, Any]] = Field(default_factory=list)
    skill_refs: list[dict[str, Any]] = Field(default_factory=list)
    enabled: bool = True
    runtime_agent_id: str | None = None

    @model_validator(mode="after")
    def _default_runtime_id(self) -> "CustomAgentRead":
        # Derive the runtime id from the row id when not supplied.
        if not self.runtime_agent_id and self.id:
            self.runtime_agent_id = runtime_agent_id_for(self.id)
        return self


# --------------------------------------------------------------------------- #
# Conversation attachment
# --------------------------------------------------------------------------- #


class ConversationCustomAgentsUpdate(_CamelModel):
    """Replace the ordered set of custom agents attached to a conversation."""

    custom_agent_ids: list[UUID] = Field(default_factory=list)

    @field_validator("custom_agent_ids")
    @classmethod
    def _no_duplicates(cls, value: list[UUID]) -> list[UUID]:
        if len(value) != len(set(value)):
            raise ValueError("custom_agent_ids must not contain duplicates")
        return value


class ConversationCustomAgentRead(_CamelModel):
    model_config = ConfigDict(from_attributes=True, alias_generator=to_camel, populate_by_name=True)

    custom_agent_id: UUID
    agent_order: int
    custom_agent: CustomAgentRead | None = None


# --------------------------------------------------------------------------- #
# Options (selectable providers/tools/skills for the current request context)
# --------------------------------------------------------------------------- #


class CustomAgentOptions(_CamelModel):
    """Selectable model providers, backend MCP tools, client tools, and skills."""

    providers: list[dict[str, Any]] = Field(default_factory=list)
    server_default_tools: list[dict[str, Any]] = Field(default_factory=list)
    server_tools: list[dict[str, Any]] = Field(default_factory=list)
    client_tools: list[dict[str, Any]] = Field(default_factory=list)
    skills: list[dict[str, Any]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Runtime workflow state entry
# --------------------------------------------------------------------------- #


class CustomAgentModelRequest(_CamelModel):
    provider_type: str
    model: str
    temperature: float | None = None
    reasoning_effort: str | None = None


class CustomAgentState(_CamelModel):
    """Per-agent entry inside ``GraphState.custom_agents`` (keyed by runtime id)."""

    id: str
    runtime_agent_id: str
    model_agent_key: str = CUSTOM_MODEL_AGENT_KEY
    name: str
    description: str | None = None
    prompt: str
    model_request: CustomAgentModelRequest
    tool_refs: list[dict[str, Any]] = Field(default_factory=list)
    skill_refs: list[dict[str, Any]] = Field(default_factory=list)
    agent_order: int = 0
