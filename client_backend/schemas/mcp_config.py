"""Strict schema-v2 models for device-local MCP configuration."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class MCPProfileScope(_StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str = Field(min_length=1)
    device_identifier: str = Field(min_length=1)


class BundledStdioServer(_StrictModel):
    transport: Literal["stdio"]
    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    cwd: str | None = None
    enabled_by_default: bool = Field(True, alias="enabledByDefault")
    description: str = ""


class BundledHTTPServer(_StrictModel):
    transport: Literal["streamable_http", "sse"]
    url: str = Field(min_length=1)
    enabled_by_default: bool = Field(True, alias="enabledByDefault")
    description: str = ""


BundledServerDefinition = Annotated[
    BundledStdioServer | BundledHTTPServer,
    Field(discriminator="transport"),
]


class CustomStdioServer(_StrictModel):
    transport: Literal["stdio"]
    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    cwd: str | None = None
    env_keys: list[str] = Field(default_factory=list, alias="envKeys")
    enabled: bool = True
    description: str = ""


class CustomHTTPServer(_StrictModel):
    transport: Literal["streamable_http", "sse"]
    url: str = Field(min_length=1)
    header_keys: list[str] = Field(default_factory=list, alias="headerKeys")
    enabled: bool = True
    description: str = ""


CustomServerDefinition = Annotated[
    CustomStdioServer | CustomHTTPServer,
    Field(discriminator="transport"),
]


class BundledOverride(_StrictModel):
    enabled: bool


class MCPRegistryDocument(_StrictModel):
    schema_version: Literal[2] = Field(alias="schemaVersion")
    servers: dict[str, BundledServerDefinition] = Field(default_factory=dict)


class MCPProfileDocument(_StrictModel):
    schema_version: Literal[2] = Field(alias="schemaVersion")
    bundled_overrides: dict[str, BundledOverride] = Field(
        default_factory=dict,
        alias="bundledOverrides",
    )
    custom_servers: dict[str, CustomServerDefinition] = Field(
        default_factory=dict,
        alias="customServers",
    )
