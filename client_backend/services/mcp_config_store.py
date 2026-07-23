"""Canonical schema-v2 MCP registry/profile persistence."""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import TypeAdapter

from client_backend.core.paths import get_device_profile_subdir
from client_backend.schemas.mcp_config import (
    BundledOverride,
    CustomServerDefinition,
    CustomStdioServer,
    MCPProfileDocument,
    MCPProfileScope,
    MCPRegistryDocument,
)
from client_backend.services.mcp_secret_store import MCPSecretStore


class MCPConfigConflictError(ValueError):
    """Raised when a custom server uses a reserved bundled name."""


@dataclass(frozen=True)
class EffectiveMCPServer:
    name: str
    source: Literal["bundled", "custom"]
    transport: str
    enabled: bool
    description: str = ""
    command: str | None = None
    args: list[str] = field(default_factory=list)
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


_CUSTOM_ADAPTER = TypeAdapter(CustomServerDefinition)


class MCPConfigStore:
    def __init__(
        self,
        scope: MCPProfileScope,
        *,
        registry_path: Path | None = None,
        application_root: Path | None = None,
        profile_root: Path | None = None,
    ) -> None:
        self.scope = scope
        self.application_root = (
            Path(application_root).resolve()
            if application_root is not None
            else Path(__file__).resolve().parents[2]
        )
        self.registry_path = (
            Path(registry_path).resolve()
            if registry_path is not None
            else self.application_root / "app" / "ai" / "mcp_config.json"
        )
        if profile_root is None:
            directory = get_device_profile_subdir(
                scope.user_id, scope.device_identifier, "mcp"
            )
        else:
            directory = (
                Path(profile_root)
                / scope.user_id
                / "devices"
                / scope.device_identifier
                / "mcp"
            )
            directory.mkdir(parents=True, exist_ok=True)
        self.profile_path = directory / "config.v2.json"
        self.secret_store = MCPSecretStore(scope, profile_root=profile_root)

    def load_registry(self) -> MCPRegistryDocument:
        return MCPRegistryDocument.model_validate_json(
            self.registry_path.read_text(encoding="utf-8")
        )

    def load_profile(self) -> MCPProfileDocument:
        if not self.profile_path.is_file():
            profile = MCPProfileDocument(schemaVersion=2)
            self._write_profile(profile)
            return profile
        return MCPProfileDocument.model_validate_json(
            self.profile_path.read_text(encoding="utf-8")
        )

    def list_effective_servers(self) -> list[EffectiveMCPServer]:
        registry = self.load_registry()
        profile = self.load_profile()
        effective: list[EffectiveMCPServer] = []
        for name, definition in registry.servers.items():
            override = profile.bundled_overrides.get(name)
            enabled = override.enabled if override is not None else definition.enabled_by_default
            payload = definition.model_dump(by_alias=True)
            command = payload.get("command")
            args = list(payload.get("args") or [])
            cwd = payload.get("cwd")
            if definition.transport == "stdio":
                if str(command).lower() in {"python", "python3"}:
                    command = sys.executable
                args = [self._resolve_bundled_value(value) for value in args]
                cwd = (
                    self._resolve_bundled_value(cwd)
                    if cwd
                    else str(self.application_root)
                )
            effective.append(
                EffectiveMCPServer(
                    name=name,
                    source="bundled",
                    transport=definition.transport,
                    enabled=enabled,
                    description=definition.description,
                    command=command,
                    args=args,
                    cwd=cwd,
                    url=payload.get("url"),
                )
            )
        for name, definition in profile.custom_servers.items():
            credentials = self.secret_store.get_for_server(name)
            payload = definition.model_dump(by_alias=True)
            effective.append(
                EffectiveMCPServer(
                    name=name,
                    source="custom",
                    transport=definition.transport,
                    enabled=definition.enabled,
                    description=definition.description,
                    command=payload.get("command"),
                    args=[
                        self._resolve_custom_value(value)
                        for value in payload.get("args") or []
                    ],
                    cwd=(
                        self._resolve_custom_value(payload["cwd"])
                        if payload.get("cwd")
                        else None
                    ),
                    env=credentials.env,
                    url=payload.get("url"),
                    headers=credentials.headers,
                )
            )
        return effective

    def save_custom_server(
        self,
        name: str,
        definition: CustomServerDefinition | dict[str, Any],
        *,
        env: dict[str, str],
        headers: dict[str, str],
    ) -> None:
        registry = self.load_registry()
        if name in registry.servers:
            raise MCPConfigConflictError(f"'{name}' is a reserved bundled server name")
        parsed = _CUSTOM_ADAPTER.validate_python(definition)
        if isinstance(parsed, CustomStdioServer):
            parsed = parsed.model_copy(update={"env_keys": sorted(env)})
        else:
            parsed = parsed.model_copy(update={"header_keys": sorted(headers)})
        profile = self.load_profile()
        candidate = profile.model_copy(
            update={"custom_servers": {**profile.custom_servers, name: parsed}}
        )
        self.secret_store.set_for_server(name, env=env, headers=headers)
        self._write_profile(candidate)

    def set_enabled(self, name: str, enabled: bool) -> None:
        registry = self.load_registry()
        profile = self.load_profile()
        if name in registry.servers:
            overrides = dict(profile.bundled_overrides)
            overrides[name] = BundledOverride(enabled=enabled)
            self._write_profile(profile.model_copy(update={"bundled_overrides": overrides}))
            return
        custom = profile.custom_servers.get(name)
        if custom is None:
            raise KeyError(name)
        servers = dict(profile.custom_servers)
        servers[name] = custom.model_copy(update={"enabled": enabled})
        self._write_profile(profile.model_copy(update={"custom_servers": servers}))

    def _resolve_bundled_value(self, value: str) -> str:
        path = Path(value)
        return str(path if path.is_absolute() else (self.application_root / path).resolve())

    def _resolve_custom_value(self, value: str) -> str:
        if value.startswith(("-", "@")) or re.match(
            r"^[A-Za-z][A-Za-z0-9+.-]*://", value
        ):
            return value
        path = Path(value)
        if path.is_absolute():
            return str(path)
        if "/" not in value and "\\" not in value and not path.suffix:
            return value
        return str((self.profile_path.parent / path).resolve())

    def _write_profile(self, profile: MCPProfileDocument) -> None:
        validated = MCPProfileDocument.model_validate(profile.model_dump(by_alias=True))
        temporary = self.profile_path.with_name(
            f".{self.profile_path.name}.{uuid4().hex}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(validated.model_dump(by_alias=True), indent=2),
                encoding="utf-8",
            )
            os.replace(temporary, self.profile_path)
        finally:
            with contextlib.suppress(OSError):
                temporary.unlink()
