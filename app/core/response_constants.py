"""Centralized response messages and helper functions."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..schemas.workflow import WorkflowResponse

_logger = logging.getLogger(__name__)
_WIDGET_TOOLS = {"widget_create", "widget_update"}


# === Response Fallback Messages ===
NO_RESPONSE_GENERATED = "No response generated"

# === Error Messages ===
ERROR_NO_RESPONSE = "Error: No response generated"
ERROR_NO_RESPONSE_RESUME = "Error: No response generated after resume"
ERROR_RESPONSE_AFTER_RESUME = "Error: No response after resuming"

# === General Errors ===
UNKNOWN_ERROR = "Unknown error"


def _extract_metadata_message(metadata: dict[str, Any] | None) -> str:
    """Derive displayable content from message metadata when body text is empty."""
    if not isinstance(metadata, dict):
        return ""

    for key in ("error", "message"):
        value = metadata.get(key)
        if isinstance(value, str):
            trimmed = value.strip()
            if trimmed:
                return trimmed

    interrupt_payload = metadata.get("interrupt")
    if isinstance(interrupt_payload, dict):
        interrupt_metadata = interrupt_payload.get("metadata")
        if isinstance(interrupt_metadata, dict):
            for key in ("message", "reason"):
                value = interrupt_metadata.get(key)
                if isinstance(value, str):
                    trimmed = value.strip()
                    if trimmed:
                        return trimmed

    return ""


def extract_response_content(
    response: Optional["WorkflowResponse"],
    fallback: str = NO_RESPONSE_GENERATED,
) -> str:
    """Extract content from a service-owned workflow response with fallback."""
    if response and response.message:
        metadata: dict[str, Any] = {}
        if isinstance(getattr(response, "metadata", None), dict):
            metadata.update(response.metadata)
        if isinstance(getattr(response.message, "metadata", None), dict):
            metadata.update(response.message.metadata)

        content = normalize_message_content(response.message.content, metadata)
        if content:
            return content

    if response and response.error:
        error_text = str(response.error).strip()
        if error_text:
            return error_text

    return fallback


def normalize_message_content(
    content: str | None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Normalize message content without injecting placeholder text."""
    if isinstance(content, str):
        trimmed = content.strip()
        if trimmed:
            return trimmed
    elif content is not None:
        rendered = str(content).strip()
        if rendered:
            return rendered

    return _extract_metadata_message(metadata)


def extract_live_widgets_from_artifacts(
    tool_artifacts: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Derive ``live_widgets`` metadata from widget tool artifacts.

    Scans tool artifacts for successful ``widget_create`` / ``widget_update``
    results and extracts the minimal metadata the frontend needs to mount
    each widget.
    """
    if not tool_artifacts:
        return []

    widgets_by_id: dict[str, dict[str, Any]] = {}

    for artifact in tool_artifacts:
        tool_name = artifact.get("tool") or artifact.get("tool_name")
        if tool_name not in _WIDGET_TOOLS:
            continue
        status = str(artifact.get("status") or "").strip().lower()
        if status in {"error", "failed", "rejected"}:
            continue
        if artifact.get("error"):
            continue
        raw_output = artifact.get("output")
        if raw_output in (None, ""):
            raw_output = artifact.get("tool_output")
        if raw_output in (None, ""):
            raw_output = artifact.get("result")
        if raw_output in (None, ""):
            continue

        try:
            parsed = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
        except (json.JSONDecodeError, TypeError):
            continue

        if not isinstance(parsed, dict):
            continue

        widget_id = parsed.get("widget_id")
        if not widget_id:
            continue

        widgets_by_id[str(widget_id)] = {
            "widget_id": widget_id,
            "session_id": parsed.get("session_id", ""),
            "widget_type": parsed.get("widget_type", ""),
            "title": parsed.get("title"),
            "status": parsed.get("status", "active"),
            "version": parsed.get("version", 1),
            "connection_endpoint": f"/widgets/{widget_id}/connection",
        }

    return list(widgets_by_id.values())


def build_bot_metadata(
    response: Optional["WorkflowResponse"],
    persona: str | None = None,
) -> dict[str, Any]:
    """Build standard bot response metadata from a workflow response."""
    metadata: dict[str, Any] = {}

    if response and response.metadata:
        metadata = dict(response.metadata)

    if persona:
        metadata.setdefault("persona_used", persona)

    if response and response.tool_artifacts:
        existing_artifacts = metadata.get("tool_artifacts")
        if isinstance(existing_artifacts, list):
            merged_artifacts = list(existing_artifacts)
            for artifact in response.tool_artifacts:
                if artifact not in merged_artifacts:
                    merged_artifacts.append(artifact)
            metadata["tool_artifacts"] = merged_artifacts
        else:
            metadata["tool_artifacts"] = list(response.tool_artifacts)

    if response and response.metadata and "images" in response.metadata:
        metadata["images"] = response.metadata["images"]

    # Derive live_widgets from widget tool artifacts
    live_widgets = extract_live_widgets_from_artifacts(
        metadata.get("tool_artifacts")
    )
    if live_widgets:
        metadata["live_widgets"] = live_widgets

    return metadata
