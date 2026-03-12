"""Human-in-the-loop configuration utilities for LangGraph agents."""

import uuid
from typing import Dict, Any, List, Optional

from app.core.config import settings
from app.ai.schemas import InterruptResponse, ToolInterruptRequest

from langgraph.types import Interrupt as LangGraphInterrupt


def is_hitl_enabled() -> bool:
    """Check if human-in-the-loop is enabled in settings."""
    return getattr(settings, "enable_human_in_the_loop", True)


def get_tools_requiring_approval() -> List[str]:
    """Get the list of tool names that require human approval."""
    return getattr(settings, "hitl_tools_require_approval", [])


def requires_human_approval(tool_names: List[str]) -> bool:
    """
    Check if any of the given tool names require human approval.

    Args:
        tool_names: List of tool names to check

    Returns:
        True if HITL is enabled and any tool requires approval
    """
    if not is_hitl_enabled():
        return False

    approval_list = get_tools_requiring_approval()
    if not approval_list:
        # Empty list means no tools require approval
        return False

    return any(name in approval_list for name in tool_names if name)


def _parse_review_configs(data: Optional[List[Dict[str, Any]]]) -> Dict[str, List[str]]:
    """Parse review configurations into a mapping of action names to allowed decisions."""
    review_configs: Dict[str, List[str]] = {}
    for cfg in data or []:
        if isinstance(cfg, dict):
            action_name = cfg.get("action_name")
            allowed = cfg.get("allowed_decisions")
            if action_name and isinstance(allowed, list):
                review_configs[action_name] = allowed
    return review_configs


def _build_tool_interrupt_request(
    task: Dict[str, Any],
    idx: int,
    default_prefix: str,
    allowed_map: Dict[str, List[str]],
) -> ToolInterruptRequest:
    """Build a single ToolInterruptRequest from a task dictionary."""
    # ID mapping priority: tool_call_id (primary) → id → task_id → generated ID
    task_id = (
        task.get("tool_call_id")
        or task.get("id")
        or task.get("task_id")
        or f"{default_prefix}:{idx}"
    )
    tool_name = task.get("action") or task.get("tool") or task.get("name") or "unknown"
    tool_args = (
        task.get("args") or task.get("tool_input") or task.get("arguments") or {}
    )
    allowed = allowed_map.get(tool_name)

    return ToolInterruptRequest(
        action=tool_name,
        args=tool_args,
        description=task.get("description"),
        task_id=task_id,
        tool_call_id=task.get("tool_call_id"),
        allowed_decisions=allowed,
    )


def _extract_action_requests(
    tasks: List[Dict[str, Any]],
    allowed_map: Dict[str, List[str]],
    default_prefix: str,
) -> List[ToolInterruptRequest]:
    """Extract ToolInterruptRequest objects from a list of tasks."""
    action_requests: List[ToolInterruptRequest] = []
    for idx, task in enumerate(tasks):
        if not isinstance(task, dict):
            continue
        request = _build_tool_interrupt_request(task, idx, default_prefix, allowed_map)
        action_requests.append(request)
    return action_requests


def build_interrupt_response(
    interrupt_data: Any, thread_id: str, conversation_id: str
) -> Dict[str, Any]:
    """Build a structured InterruptResponse from raw interrupt data."""
    action_requests: List[ToolInterruptRequest] = []
    interrupt_id: Optional[str] = None
    response_metadata: Dict[str, Any] = {}

    # Normalize interrupt_data to a list of payloads
    payloads: List[Any] = []
    if isinstance(interrupt_data, LangGraphInterrupt):
        interrupt_id = interrupt_data.id or interrupt_id
        payloads.append(interrupt_data.value)
    elif isinstance(interrupt_data, (list, tuple)):
        for item in interrupt_data:
            if isinstance(item, LangGraphInterrupt):
                interrupt_id = interrupt_id or item.id
                payloads.append(item.value)
            else:
                payloads.append(item)
    else:
        payloads.append(interrupt_data)

    for payload in payloads:
        if not isinstance(payload, dict):
            continue

        if not interrupt_id:
            interrupt_id = payload.get("interrupt_id")

        payload_metadata = payload.get("metadata")
        if isinstance(payload_metadata, dict):
            response_metadata.update(payload_metadata)

        for key in ("message", "reason"):
            value = payload.get(key)
            if (
                key not in response_metadata
                and isinstance(value, str)
                and value.strip()
            ):
                response_metadata[key] = value.strip()

        default_prefix = interrupt_id or "task"

        if "action_requests" in payload:
            allowed_map = _parse_review_configs(payload.get("review_configs", []))
            requests = _extract_action_requests(
                payload.get("action_requests", []), allowed_map, default_prefix
            )
            action_requests.extend(requests)
            continue

        # Fallback support for older interrupt shapes
        tasks = (
            payload.get("tasks")
            or payload.get("__interrupt__")
            or payload.get("tool_calls")
            or []
        )
        allowed_map = _parse_review_configs(payload.get("review_configs", []))
        requests = _extract_action_requests(tasks, allowed_map, default_prefix)
        action_requests.extend(requests)

    interrupt_id = interrupt_id or str(uuid.uuid4())

    response = InterruptResponse(
        interrupt_id=interrupt_id,
        action_requests=action_requests,
        thread_id=thread_id,
        conversation_id=conversation_id,
        metadata=response_metadata,
    )

    return response.model_dump()
