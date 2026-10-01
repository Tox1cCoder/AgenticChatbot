"""Isolated one-time migration from legacy MCP documents to schema v2."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import TypeAdapter

from client_backend.core.config import client_settings
from client_backend.core.paths import get_profile_subdir, normalize_path
from client_backend.core.security import encrypt_local_secret
from client_backend.schemas.mcp_config import (
    BundledOverride,
    CustomServerDefinition,
    MCPProfileDocument,
    MCPProfileScope,
    MCPRegistryDocument,
)
from client_backend.services.mcp_config_store import MCPConfigStore

_CUSTOM_ADAPTER = TypeAdapter(CustomServerDefinition)


class MCPConfigMigrationConflictError(ValueError):
    """Raised when a modified server collides with a reserved bundled name."""


@dataclass(frozen=True)
class MigrationResult:
    status: Literal["migrated", "already_v2", "created_empty", "not_found"]
    migrated_servers: tuple[str, ...] = ()
    backup_path: Path | None = None
    receipt_path: Path | None = None


def prepare_mcp_config_store(
    scope: MCPProfileScope,
    *,
    store: MCPConfigStore | None = None,
    legacy_path: Path | None = None,
) -> tuple[MCPConfigStore, MigrationResult]:
    """Open a canonical store only after its one-time migration check."""

    resolved_store = store or MCPConfigStore(scope)
    if resolved_store.scope != scope:
        raise ValueError("configuration store scope does not match requested scope")
    if legacy_path is None:
        legacy_path = (
            normalize_path(client_settings.mcp_config_path)
            if client_settings.mcp_config_path
            else get_profile_subdir(scope.user_id, "mcp") / "mcp_config.json"
        )
    result = migrate_legacy_mcp_profile(
        scope,
        legacy_path=legacy_path,
        store=resolved_store,
    )
    return resolved_store, result


def _legacy_servers(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    snake = payload.get("mcp_servers")
    camel = payload.get("mcpServers")
    if isinstance(snake, dict):
        merged.update(
            (str(name), dict(value)) for name, value in snake.items() if isinstance(value, dict)
        )
    if isinstance(camel, dict):
        merged.update(
            (str(name), dict(value)) for name, value in camel.items() if isinstance(value, dict)
        )
    return merged


def _bundle_signature(value: dict[str, Any]) -> dict[str, Any]:
    result = {
        "transport": (
            "streamable_http"
            if str(value.get("transport") or "stdio").lower() == "http"
            else str(value.get("transport") or "stdio").lower()
        ),
        "command": value.get("command"),
        "args": list(value.get("args") or []),
        "cwd": value.get("cwd"),
        "url": value.get("url"),
        "description": str(value.get("description") or ""),
    }
    return {key: item for key, item in result.items() if item not in (None, [], "")}


_ServerCredentials = tuple[dict[str, str], dict[str, str]]


@dataclass(frozen=True)
class _MigrationPlan:
    """The v2 documents a legacy profile maps to, built before anything is written."""

    overrides: dict[str, BundledOverride]
    custom: dict[str, Any]
    credentials: dict[str, _ServerCredentials]


def _custom_server_definition(raw: dict[str, Any]) -> tuple[dict[str, Any], _ServerCredentials]:
    """A legacy custom server as a v2 definition plus its ``(env, headers)`` secrets."""
    transport = str(raw.get("transport") or "stdio").strip().lower()
    if transport == "http":
        transport = "streamable_http"
    env = {str(key): str(value) for key, value in (raw.get("env") or {}).items()}
    headers = {str(key): str(value) for key, value in (raw.get("headers") or {}).items()}
    definition: dict[str, Any] = {
        "transport": transport,
        "enabled": bool(raw.get("enabled", True)),
        "description": str(raw.get("description") or ""),
    }
    if transport == "stdio":
        definition.update(
            {
                "command": raw.get("command"),
                "args": list(raw.get("args") or []),
                "cwd": raw.get("cwd"),
                "envKeys": sorted(env),
            }
        )
    else:
        definition.update(
            {
                "url": raw.get("url"),
                "headerKeys": sorted(headers),
            }
        )
    return definition, (env, headers)


def _plan_migration(
    legacy_servers: dict[str, dict[str, Any]], registry: MCPRegistryDocument
) -> _MigrationPlan:
    """Map every legacy server; raise when a modified one uses a bundled name."""
    overrides: dict[str, BundledOverride] = {}
    custom: dict[str, Any] = {}
    credentials: dict[str, _ServerCredentials] = {}
    conflicts: list[str] = []

    for name, raw in legacy_servers.items():
        bundled = registry.servers.get(name)
        if bundled is not None:
            expected = bundled.model_dump(by_alias=True)
            if _bundle_signature(raw) != _bundle_signature(expected):
                conflicts.append(name)
                continue
            overrides[name] = BundledOverride(
                enabled=bool(raw.get("enabled", bundled.enabled_by_default))
            )
            continue

        definition, server_credentials = _custom_server_definition(raw)
        custom[name] = _CUSTOM_ADAPTER.validate_python(definition)
        credentials[name] = server_credentials

    if conflicts:
        joined = ", ".join(sorted(conflicts))
        raise MCPConfigMigrationConflictError(
            f"customized definitions use reserved bundled names: {joined}"
        )
    return _MigrationPlan(overrides=overrides, custom=custom, credentials=credentials)


def _write_encrypted_backup(backup_path: Path, source_bytes: bytes) -> None:
    backup_path.write_text(
        json.dumps(encrypt_local_secret(source_bytes)),
        encoding="utf-8",
    )
    with contextlib.suppress(OSError):
        os.chmod(backup_path, 0o600)


def _write_profile_with_credentials(
    store: MCPConfigStore,
    profile: MCPProfileDocument,
    credentials: dict[str, _ServerCredentials],
) -> None:
    """Store each server's secrets, then the profile; restore the secret file on failure."""
    secret_snapshot = (
        store.secret_store.path.read_bytes() if store.secret_store.path.is_file() else None
    )
    try:
        for name, (env, headers) in credentials.items():
            store.secret_store.set_for_server(name, env=env, headers=headers)
        store._write_profile(profile)
    except Exception:
        if secret_snapshot is None:
            store.secret_store.path.unlink(missing_ok=True)
        else:
            store.secret_store.path.write_bytes(secret_snapshot)
        raise


def _write_receipt(
    receipt_path: Path,
    *,
    source_bytes: bytes,
    backup_path: Path,
    servers: list[str],
) -> None:
    receipt = {
        "schemaVersion": 2,
        "migratedAt": datetime.now(timezone.utc).isoformat(),
        "sourceSha256": hashlib.sha256(source_bytes).hexdigest(),
        "backupPath": str(backup_path),
        "servers": servers,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")


def migrate_legacy_mcp_profile(
    scope: MCPProfileScope,
    *,
    legacy_path: Path,
    store: MCPConfigStore,
) -> MigrationResult:
    """Migrate one legacy user profile without mutating or deleting its source."""

    if scope != store.scope:
        raise ValueError("migration scope does not match configuration store scope")
    if store.profile_path.is_file():
        store.load_profile()
        return MigrationResult(status="already_v2")
    legacy_path = Path(legacy_path)
    if not legacy_path.is_file():
        store.load_profile()
        return MigrationResult(status="created_empty")

    source_bytes = legacy_path.read_bytes()
    payload = json.loads(source_bytes.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("legacy MCP configuration must be an object")
    legacy_servers = _legacy_servers(payload)
    plan = _plan_migration(legacy_servers, store.load_registry())

    migration_dir = store.profile_path.parent / "migration"
    migration_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup_path = migration_dir / f"legacy-{timestamp}.encrypted.json"
    _write_encrypted_backup(backup_path, source_bytes)

    profile = MCPProfileDocument(
        schemaVersion=2,
        bundledOverrides=plan.overrides,
        customServers=plan.custom,
    )
    _write_profile_with_credentials(store, profile, plan.credentials)

    receipt_path = migration_dir / f"receipt-{timestamp}.json"
    _write_receipt(
        receipt_path,
        source_bytes=source_bytes,
        backup_path=backup_path,
        servers=sorted(legacy_servers),
    )
    return MigrationResult(
        status="migrated",
        migrated_servers=tuple(sorted(legacy_servers)),
        backup_path=backup_path,
        receipt_path=receipt_path,
    )
