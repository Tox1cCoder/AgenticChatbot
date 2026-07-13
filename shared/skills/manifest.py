"""Pydantic models and validation for the ``skill.json`` manifest format.

A ``skill.json`` file declares how a skill bundle's capabilities are executed:
runtime type, dependencies, required secrets, permissions, and a typed list of
capabilities with JSON-Schema-ish input schemas. This module defines the
first-slice, provider-neutral contract. It does not execute anything, resolve
secrets, or touch the filesystem.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator, model_validator

# First-slice runtime types the execution engine can actually dispatch to.
SUPPORTED_RUNTIME_TYPES = frozenset({"python_module", "python_script", "binary"})

# Reserved for future slices; accepted in the schema shape but rejected today
# so we never silently accept metadata we cannot execute safely.
RESERVED_RUNTIME_TYPES = frozenset({"node_package", "mcp_server", "shell"})

SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0"})

# Capability names become part of a `skill::<skill>::<capability>` qualified
# id and a `client__..._<capability>` model-facing tool name, so they must be
# safe identifiers: start with a letter, then letters/digits/underscores only.
_CAPABILITY_NAME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")

# Skill names may contain hyphens (e.g. "example-calendar").
_SKILL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")


class SkillRuntimeSpec(BaseModel):
    """How a skill's capabilities are invoked."""

    type: str
    module: str | None = None
    entrypoint: str | None = None
    script: str | None = None
    command: str | None = None

    @field_validator("type")
    @classmethod
    def _validate_type(cls, value: str) -> str:
        if value in SUPPORTED_RUNTIME_TYPES:
            return value
        if value in RESERVED_RUNTIME_TYPES:
            raise ValueError(f"runtime type '{value}' is reserved for a future slice")
        raise ValueError(
            f"unknown runtime type '{value}'; supported types are: "
            f"{sorted(SUPPORTED_RUNTIME_TYPES)}"
        )


class SkillDependencySpec(BaseModel):
    """Declared dependencies a skill needs to be ready to execute."""

    python: list[str] = Field(default_factory=list)
    node: list[str] = Field(default_factory=list)
    system: list[str] = Field(default_factory=list)


class SkillSecretSpec(BaseModel):
    """A named secret a skill (or one of its capabilities) requires."""

    name: str
    required: bool = True
    description: str | None = None

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("secret name must not be empty")
        return value


class SkillCapabilityExecutionSpec(BaseModel):
    """The concrete invocation contract for one capability."""

    argv: list[str]
    json_output: bool = False


class SkillCapabilitySpec(BaseModel):
    """One typed, callable operation exposed by a skill."""

    name: str
    description: str
    input_schema: dict
    execution: SkillCapabilityExecutionSpec
    permissions: list[str] = Field(default_factory=list)
    secrets: list[str] = Field(default_factory=list)
    mutation: bool = False

    def is_mutation(self) -> bool:
        """Whether this capability mutates state.

        A capability may declare mutation via the ``mutation`` flag OR by
        listing the literal ``"mutation"`` permission token. This is the single
        source of truth for that definition so the readiness/permission
        evaluator, the HITL catalog signal, and any future consumer never drift
        apart on what counts as a mutation.
        """
        return self.mutation or "mutation" in self.permissions

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not _CAPABILITY_NAME_PATTERN.match(value):
            raise ValueError(
                f"capability name '{value}' is not a safe tool identifier; it must "
                "match ^[a-zA-Z][a-zA-Z0-9_]*$ (start with a letter, then letters, "
                "digits, or underscores only)"
            )
        return value

    @field_validator("description")
    @classmethod
    def _validate_description(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("capability description must not be empty")
        return value

    @field_validator("input_schema")
    @classmethod
    def _validate_input_schema(cls, value: dict) -> dict:
        if not value:
            raise ValueError("capability input_schema must be a non-empty object")
        return value


class SkillManifest(BaseModel):
    """The full, validated contents of a ``skill.json`` file."""

    schema_version: str
    name: str
    display_name: str | None = None
    description: str
    runtime: SkillRuntimeSpec
    dependencies: SkillDependencySpec = Field(default_factory=SkillDependencySpec)
    secrets: list[SkillSecretSpec] = Field(default_factory=list)
    permissions: list[str] = Field(default_factory=list)
    capabilities: list[SkillCapabilitySpec]

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: str) -> str:
        if value not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported schema_version '{value}'; supported versions are: "
                f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )
        return value

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not _SKILL_NAME_PATTERN.match(value):
            raise ValueError(
                f"skill name '{value}' must match ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ "
                "(letters, digits, underscore, hyphen only)"
            )
        return value

    @field_validator("description")
    @classmethod
    def _validate_description(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("manifest description must not be empty")
        return value

    @model_validator(mode="after")
    def _validate_unique_capability_names(self) -> SkillManifest:
        seen: set[str] = set()
        for capability in self.capabilities:
            if capability.name in seen:
                raise ValueError(
                    f"duplicate capability name '{capability.name}' in manifest; "
                    "capability names must be unique within a skill"
                )
            seen.add(capability.name)
        return self


def load_manifest(data: dict) -> SkillManifest:
    """Construct and validate a :class:`SkillManifest` from a parsed-JSON dict.

    This is the one entry point later tasks (registry, readiness manager,
    execution engine) should use to turn a loaded ``skill.json`` payload into a
    validated manifest. Raises ``pydantic.ValidationError`` on any malformed
    or unsafe manifest; callers are expected to let it propagate or catch it
    to build normalized runtime error codes (a later task).
    """
    return SkillManifest.model_validate(data)
