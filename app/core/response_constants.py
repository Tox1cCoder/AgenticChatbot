"""Centralized response messages and helper functions."""

from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..ai.schemas import AgentResponse


# === Response Fallback Messages ===
NO_RESPONSE_GENERATED = "No response generated"
EMPTY_MESSAGE_PLACEHOLDER = "[Empty message]"

# === Error Messages ===
ERROR_NO_RESPONSE = "Error: No response generated"
ERROR_NO_RESPONSE_RESUME = "Error: No response generated after resume"
ERROR_NO_RESPONSE_DECISIONS = "Error: No response generated after resuming with decisions"
ERROR_RESPONSE_AFTER_RESUME = "Error: No response after resuming"

# === Tool/Approval Messages ===
TOOL_APPROVAL_REQUIRED = "Tool execution requires approval"
WORKFLOW_PAUSED_MESSAGE = "Workflow paused - awaiting approval for tool execution"

# === General Errors ===
UNKNOWN_ERROR = "Unknown error"
DEFAULT_ERROR_MESSAGE = "An error occurred while generating a response"


def extract_response_content(
    response: Optional["AgentResponse"],
    fallback: str = NO_RESPONSE_GENERATED,
) -> str:
    """Extract content from AgentResponse with fallback."""
    if response and response.message and response.message.content:
        content = response.message.content.strip()
        return content if content else fallback
    return fallback


def normalize_message_content(content: Optional[str]) -> str:
    """Ensure message content is not empty."""
    if content is None:
        return EMPTY_MESSAGE_PLACEHOLDER
    trimmed = content.strip()
    return trimmed if trimmed else EMPTY_MESSAGE_PLACEHOLDER


def build_bot_metadata(
    response: Optional["AgentResponse"],
    persona: Optional[str] = None,
) -> Dict[str, Any]:
    """Build standard bot response metadata from AgentResponse."""
    metadata: Dict[str, Any] = {}
    
    if response and response.metadata:
        metadata = dict(response.metadata)
    
    if persona:
        metadata.setdefault("persona_used", persona)
    
    if response and response.tool_artifacts:
        metadata.setdefault("tool_artifacts", response.tool_artifacts)
    
    if response and response.metadata and "images" in response.metadata:
        metadata["images"] = response.metadata["images"]
    
    return metadata
