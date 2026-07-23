"""Pure resolver mapping saved logical Custom Agent selections to live capabilities.

This module is intentionally free of database, network, and runtime-singleton
dependencies. It converts account-wide saved MCP/skill selections into the
requesting device's current exact tool refs, reports missing capabilities, and
computes a structured availability status. It is shared by Custom Agent
management reads and runtime binding so both surfaces agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

CapabilityStatus = Literal["ready", "degraded", "device_unavailable"]
_LIVE_IDENTITY_FIELDS = (
    "device_id",
    "session_id",
    "catalog_version",
    "tool_instance_id",
    "server_name",
    "qualified_tool_id",
    "tool_name",
)


@dataclass(frozen=True)
class CapabilityResolution:
    status: CapabilityStatus
    resolved_client_tool_refs: list[dict[str, Any]]
    missing_tools: list[dict[str, Any]]
    missing_skills: list[dict[str, Any]]
    warnings: list[str]


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    metadata = getattr(value, "metadata", None)
    if isinstance(metadata, dict):
        merged = dict(metadata)
        merged.setdefault("tool_name", metadata.get("source_tool_name"))
        return merged
    return {
        "source": getattr(value, "source", None),
        "lookup_name": getattr(value, "lookup_name", None),
        "name": getattr(value, "name", None),
        "device_id": getattr(value, "bound_device_id", None),
        "session_id": getattr(value, "bound_session_id", None),
    }


def client_tool_logical_key(value: Any) -> tuple[str, str] | None:
    item = _mapping(value)
    explicit_type = str(item.get("type") or "").strip()
    if explicit_type and explicit_type != "client":
        return None
    if not explicit_type and not (
        bool(item.get("is_client_tool"))
        or "client" in str(item.get("tool_origin") or item.get("origin") or "")
    ):
        return None
    server_name = str(item.get("server_name") or "").strip()
    qualified_id = str(item.get("qualified_tool_id") or "").strip()
    if not server_name or not qualified_id:
        return None
    return server_name, qualified_id


def client_tool_binding_key(value: Any) -> tuple[str, ...] | None:
    """Full identity required when a caller submits a new current option."""
    item = _mapping(value)
    logical = client_tool_logical_key(item)
    fields = (
        item.get("device_id"),
        item.get("session_id"),
        item.get("catalog_version"),
        item.get("tool_instance_id"),
        item.get("tool_name") or item.get("source_tool_name"),
    )
    if logical is None or any(not str(field or "").strip() for field in fields):
        return None
    return (*logical, *(str(field) for field in fields))


def _skill_aliases(value: Any) -> tuple[str, set[str]]:
    item = _mapping(value)
    source = str(item.get("source") or "client").strip().lower()
    if source == "server":
        source = "client"
    aliases = {
        str(candidate).strip()
        for candidate in (item.get("lookup_name"), item.get("name"))
        if str(candidate or "").strip()
    }
    return source, aliases


def skill_logical_key(value: Any) -> tuple[str, str] | None:
    item = _mapping(value)
    source, aliases = _skill_aliases(item)
    lookup_name = str(item.get("lookup_name") or item.get("name") or "").strip()
    if not source or not lookup_name or not aliases:
        return None
    return source, lookup_name


def skill_refs_match(left: Any, right: Any) -> bool:
    left_source, left_aliases = _skill_aliases(left)
    right_source, right_aliases = _skill_aliases(right)
    return bool(
        left_source
        and left_source == right_source
        and left_aliases
        and left_aliases & right_aliases
    )


def resolve_custom_agent_capabilities(
    *,
    selected_tool_refs: list[dict[str, Any]],
    selected_skill_refs: list[dict[str, Any]],
    live_tool_refs: list[Any],
    live_skill_refs: list[Any],
    request_device_id: str | None,
    device_available: bool,
) -> CapabilityResolution:
    selected_clients = [ref for ref in selected_tool_refs if str(ref.get("type") or "") == "client"]
    live_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for tool in live_tool_refs if device_available and request_device_id else []:
        item = _mapping(tool)
        if request_device_id and str(item.get("device_id") or "") != str(request_device_id):
            continue
        key = client_tool_logical_key(item)
        if key is not None:
            live_by_key[key] = item
    resolved: list[dict[str, Any]] = []
    missing_tools: list[dict[str, Any]] = []
    warnings: list[str] = []
    missing_tool_keys: set[tuple[Any, ...]] = set()

    for ref in selected_clients:
        live = live_by_key.get(client_tool_logical_key(ref))
        if live is None:
            missing_key = client_tool_logical_key(ref) or ("invalid", id(ref))
            if missing_key in missing_tool_keys:
                continue
            missing_tool_keys.add(missing_key)
            missing = {
                "server_name": str(ref.get("server_name") or ""),
                "qualified_tool_id": str(ref.get("qualified_tool_id") or ""),
                "tool_name": ref.get("tool_name"),
            }
            missing_tools.append(missing)
            warnings.append(
                f"Selected MCP tool '{missing['qualified_tool_id']}' "
                "is not available on this device."
            )
            continue
        rebound = dict(ref)
        for field in _LIVE_IDENTITY_FIELDS:
            if live.get(field) is not None:
                rebound[field] = live[field]
        resolved.append(rebound)

    live_skills = list(live_skill_refs) if device_available and request_device_id else []
    missing_skills: list[dict[str, Any]] = []
    missing_skill_keys: set[tuple[Any, ...]] = set()
    for ref in selected_skill_refs:
        if any(skill_refs_match(ref, live) for live in live_skills):
            continue
        missing_key = skill_logical_key(ref) or ("invalid", id(ref))
        if missing_key in missing_skill_keys:
            continue
        missing_skill_keys.add(missing_key)
        lookup_name = str(ref.get("lookup_name") or ref.get("name") or "")
        name = str(ref.get("name") or lookup_name)
        missing_skills.append({"lookup_name": lookup_name, "name": name})
        warnings.append(f"Selected skill '{lookup_name}' is not available on this device.")

    has_local_dependencies = bool(selected_clients or selected_skill_refs)
    if has_local_dependencies and not device_available:
        status: CapabilityStatus = "device_unavailable"
    elif missing_tools or missing_skills:
        status = "degraded"
    else:
        status = "ready"

    return CapabilityResolution(
        status=status,
        resolved_client_tool_refs=resolved,
        missing_tools=missing_tools,
        missing_skills=missing_skills,
        warnings=list(dict.fromkeys(warnings)),
    )
