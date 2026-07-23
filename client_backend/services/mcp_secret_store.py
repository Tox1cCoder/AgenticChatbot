"""Encrypted MCP credentials isolated by user and installation identity."""

from __future__ import annotations

import contextlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from client_backend.core.paths import get_device_profile_subdir
from client_backend.core.security import decrypt_local_secret, encrypt_local_secret
from client_backend.schemas.mcp_config import MCPProfileScope
from client_backend.services.mcp_file_lock import mcp_path_lock

_STORAGE_VERSION = 1
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class MCPServerCredentials:
    env: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)


class MCPSecretStore:
    """Persist encrypted environment and header values for one local device."""

    def __init__(
        self,
        scope: MCPProfileScope,
        *,
        profile_root: Path | None = None,
    ) -> None:
        self.scope = scope
        if profile_root is None:
            directory = get_device_profile_subdir(
                scope.user_id,
                scope.device_identifier,
                "mcp",
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
        self.path = directory / "credentials.json"

    def get_for_server(self, server_name: str) -> MCPServerCredentials:
        values = self._read().get(self._validate_server_name(server_name), {})
        return MCPServerCredentials(
            env={str(key): str(value) for key, value in (values.get("env") or {}).items()},
            headers={
                str(key): str(value) for key, value in (values.get("headers") or {}).items()
            },
        )

    def list_for_server(self, server_name: str) -> dict[str, list[str]]:
        credentials = self.get_for_server(server_name)
        return {
            "envKeys": sorted(credentials.env),
            "headerKeys": sorted(credentials.headers),
        }

    def set_for_server(
        self,
        server_name: str,
        *,
        env: dict[str, str],
        headers: dict[str, str],
    ) -> None:
        with mcp_path_lock(self.path.parent):
            self._set_for_server(server_name, env=env, headers=headers)

    def _set_for_server(
        self,
        server_name: str,
        *,
        env: dict[str, str],
        headers: dict[str, str],
    ) -> None:
        name = self._validate_server_name(server_name)
        normalized_env: dict[str, str] = {}
        for key, value in env.items():
            if not _ENV_NAME.fullmatch(str(key)):
                raise ValueError(f"invalid environment variable name: {key}")
            normalized_env[str(key)] = str(value)
        normalized_headers: dict[str, str] = {}
        for key, value in headers.items():
            header_name = str(key).strip()
            if not header_name or "\r" in header_name or "\n" in header_name:
                raise ValueError("invalid HTTP header name")
            normalized_headers[header_name] = str(value)

        bindings = self._read()
        bindings[name] = {"env": normalized_env, "headers": normalized_headers}
        self._write(bindings)

    def delete_server(self, server_name: str) -> bool:
        with mcp_path_lock(self.path.parent):
            name = self._validate_server_name(server_name)
            bindings = self._read()
            if name not in bindings:
                return False
            del bindings[name]
            self._write(bindings)
            return True

    @staticmethod
    def _validate_server_name(server_name: str) -> str:
        name = str(server_name or "").strip()
        if not name:
            raise ValueError("server name must not be empty")
        return name

    def _read(self) -> dict[str, dict[str, dict[str, str]]]:
        if not self.path.is_file():
            return {}
        envelope = json.loads(self.path.read_text(encoding="utf-8"))
        plaintext = decrypt_local_secret(envelope)
        payload = json.loads(plaintext.decode("utf-8"))
        if payload.get("version") != _STORAGE_VERSION:
            raise ValueError("unsupported MCP credential storage version")
        servers = payload.get("servers")
        if not isinstance(servers, dict):
            raise ValueError("invalid MCP credential storage payload")
        return servers

    def _write(self, bindings: dict[str, dict[str, dict[str, str]]]) -> None:
        payload = {"version": _STORAGE_VERSION, "servers": bindings}
        envelope = encrypt_local_secret(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        )
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(envelope), encoding="utf-8")
            with contextlib.suppress(OSError):
                os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
        finally:
            with contextlib.suppress(OSError):
                temporary.unlink()
