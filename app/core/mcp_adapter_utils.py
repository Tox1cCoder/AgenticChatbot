"""Shared MCP adapter normalization helpers for server and client managers."""

from __future__ import annotations

from copy import copy
from typing import Any

from pydantic import BaseModel as PydanticBaseModel


def default_mcp_input_schema() -> dict[str, Any]:
    """Return the normalized empty input schema used for tools without arguments."""

    return {"type": "object", "properties": {}}


def normalize_mcp_transport(value: Any) -> str:
    """Normalize transport aliases into the names expected by MCP adapters."""

    transport = str(value or "stdio").strip().lower()
    if transport == "http":
        return "streamable_http"
    return transport


def build_mcp_server_entry(
    *,
    transport: Any,
    command: str | None = None,
    args: list[str] | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    url: str | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """Build one MultiServerMCPClient server entry from normalized config fields."""

    normalized_transport = normalize_mcp_transport(transport)

    if normalized_transport == "stdio":
        if not command:
            return None

        entry: dict[str, Any] = {
            "transport": "stdio",
            "command": command,
            "args": list(args or []),
        }
        if cwd:
            entry["cwd"] = cwd
        if env:
            entry["env"] = env
        return entry

    if normalized_transport in {"streamable_http", "sse"}:
        if not url:
            return None

        entry = {
            "transport": normalized_transport,
            "url": url,
        }
        if headers:
            entry["headers"] = headers
        return entry

    return None


def clean_mcp_tool_name(tool_name: str) -> str:
    """Strip the optional server prefix that some MCP adapters include in tool names."""

    if ":" in tool_name:
        return tool_name.split(":", 1)[-1]
    return tool_name


def _filter_mcp_schema_recursively(schema: Any) -> Any:
    unsupported_keys = {"$schema", "additionalProperties"}

    if isinstance(schema, dict):
        filtered: dict[str, Any] = {}
        for key, value in schema.items():
            if key in unsupported_keys or value is None:
                continue

            if key in {"properties", "items", "anyOf", "allOf", "oneOf", "definitions"}:
                filtered_value = _filter_mcp_schema_recursively(value)
                if filtered_value:
                    filtered[key] = filtered_value
                continue

            if isinstance(value, dict):
                filtered_value = _filter_mcp_schema_recursively(value)
                if filtered_value:
                    filtered[key] = filtered_value
                continue

            if isinstance(value, list):
                filtered[key] = [
                    _filter_mcp_schema_recursively(item) for item in value if item is not None
                ]
                continue

            filtered[key] = value

        if filtered.get("type") == "array" and "items" not in filtered:
            filtered["items"] = {"type": "string"}

        return filtered

    if isinstance(schema, list):
        return [_filter_mcp_schema_recursively(item) for item in schema if item is not None]

    return schema


def _remove_non_string_enums(schema: Any) -> Any:
    if isinstance(schema, dict):
        cleaned: dict[str, Any] = {}
        for key, value in schema.items():
            if (
                key == "enum"
                and isinstance(value, list)
                and any(not isinstance(item, str) for item in value)
            ):
                continue
            cleaned[key] = _remove_non_string_enums(value)
        return cleaned

    if isinstance(schema, list):
        return [_remove_non_string_enums(item) for item in schema]

    return schema


def _normalize_schema_dict(schema: dict[str, Any]) -> dict[str, Any]:
    filtered = _filter_mcp_schema_recursively(schema)
    filtered = _remove_non_string_enums(filtered)
    if not filtered.get("properties"):
        filtered["properties"] = {}
    if not filtered.get("type"):
        filtered["type"] = "object"
    return filtered


def sanitize_mcp_schema(schema: Any) -> dict[str, Any]:
    """Export one schema value into a sanitized JSON-schema dict without mutation."""

    if not schema:
        return default_mcp_input_schema()

    if isinstance(schema, dict):
        return _normalize_schema_dict(schema)

    exported_schema: Any = None

    if (
        PydanticBaseModel is not None
        and (
            isinstance(schema, type)
            and issubclass(schema, PydanticBaseModel)
            or isinstance(schema, PydanticBaseModel)
        )
        and hasattr(schema, "model_json_schema")
    ):
        exported_schema = schema.model_json_schema()

    if not exported_schema:
        for attr_name in ("model_json_schema", "json_schema", "schema"):
            exporter = getattr(schema, attr_name, None)
            if not callable(exporter):
                continue
            try:
                exported_schema = exporter()
                break
            except TypeError:
                try:
                    exported_schema = exporter(by_alias=True)
                    break
                except Exception:
                    continue
            except Exception:
                continue

    if isinstance(exported_schema, dict):
        return _normalize_schema_dict(exported_schema)

    return default_mcp_input_schema()


def clone_mcp_tool(tool: Any) -> Any:
    """Clone one MCP tool with a normalized name and sanitized args schema."""

    cloned_tool = tool.model_copy(deep=False) if hasattr(tool, "model_copy") else copy(tool)
    cloned_tool.name = clean_mcp_tool_name(str(getattr(tool, "name", "") or "unknown"))
    cloned_tool.args_schema = sanitize_mcp_schema(getattr(tool, "args_schema", None))
    return cloned_tool
