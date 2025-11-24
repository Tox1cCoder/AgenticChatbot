"""Human-in-the-loop configuration utilities for LangGraph agents."""

import uuid
from typing import Dict, Any, List, Optional

from app.core.config import settings
from app.ai.schemas import InterruptResponse, ToolInterruptRequest

try:
    from langgraph.types import Interrupt as LangGraphInterrupt
except Exception:
    LangGraphInterrupt = None


def should_enable_hitl() -> bool:
    """Check if human-in-the-loop is enabled in settings."""
    return getattr(settings, "enable_human_in_the_loop", True)


def build_interrupt_response(
    interrupt_data: Any, thread_id: str, conversation_id: str
) -> Dict[str, Any]:
    """Build a structured InterruptResponse from raw interrupt data."""
    action_requests: List[ToolInterruptRequest] = []
    interrupt_id: Optional[str] = None

    def _parse_review_configs(data: Dict[str, Any]) -> Dict[str, List[str]]:
        review_configs = {}
        for cfg in data or []:
            if isinstance(cfg, dict):
                action_name = cfg.get("action_name")
                allowed = cfg.get("allowed_decisions")
                if action_name and isinstance(allowed, list):
                    review_configs[action_name] = allowed
        return review_configs

    def _add_requests(tasks: List[Dict[str, Any]], allowed_map: Dict[str, List[str]]):
        default_prefix = interrupt_id or "task"
        for idx, task in enumerate(tasks):
            if not isinstance(task, dict):
                continue
            task_id = (
                task.get("id")
                or task.get("task_id")
                or task.get("tool_call_id")
                or f"{default_prefix}:{idx}"
            )
            tool_name = (
                task.get("action") or task.get("tool") or task.get("name") or "unknown"
            )
            tool_args = (
                task.get("args")
                or task.get("tool_input")
                or task.get("arguments")
                or {}
            )
            allowed = allowed_map.get(tool_name)

            action_requests.append(
                ToolInterruptRequest(
                    action=tool_name,
                    args=tool_args,
                    description=task.get("description"),
                    task_id=task_id,
                    tool_call_id=task.get("tool_call_id"),
                    allowed_decisions=allowed,
                )
            )

    # Normalize interrupt_data to a list of payloads
    payloads: List[Any] = []
    if LangGraphInterrupt and isinstance(interrupt_data, LangGraphInterrupt):
        interrupt_id = interrupt_data.id or interrupt_id
        payloads.append(interrupt_data.value)
    elif isinstance(interrupt_data, (list, tuple)):
        for item in interrupt_data:
            if LangGraphInterrupt and isinstance(item, LangGraphInterrupt):
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

        # New LangGraph HITL payload shape
        if "action_requests" in payload:
            allowed_map = _parse_review_configs(payload.get("review_configs", []))
            _add_requests(payload.get("action_requests", []), allowed_map)
            continue

        # Fallback support for older interrupt shapes
        tasks = (
            payload.get("tasks")
            or payload.get("__interrupt__")
            or payload.get("tool_calls")
            or []
        )
        allowed_map = _parse_review_configs(payload.get("review_configs", []))
        _add_requests(tasks, allowed_map)

    interrupt_id = interrupt_id or str(uuid.uuid4())

    response = InterruptResponse(
        interrupt_id=interrupt_id,
        action_requests=action_requests,
        thread_id=thread_id,
        conversation_id=conversation_id,
    )

    return response.model_dump()
