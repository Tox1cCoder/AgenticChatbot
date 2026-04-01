"""Centralized response messages and helper functions."""

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..schemas.workflow import WorkflowResponse


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
        metadata.setdefault("tool_artifacts", response.tool_artifacts)

    if response and response.metadata and "images" in response.metadata:
        metadata["images"] = response.metadata["images"]

    return metadata
