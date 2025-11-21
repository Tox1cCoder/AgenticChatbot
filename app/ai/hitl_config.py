"""
Human-in-the-loop configuration utilities for LangGraph agents.

This module provides centralized configuration for human approval workflows,
including middleware settings and interrupt response handling.
"""

import uuid
from typing import Dict, Any, List, Optional

from app.core.config import settings
from app.ai.schemas import InterruptResponse, ToolInterruptRequest

try:
    from langgraph.types import Interrupt as LangGraphInterrupt
except Exception:  # pragma: no cover - defensive import for older runtimes
    LangGraphInterrupt = None


def get_hitl_middleware_config(tool_names: List[str]) -> Dict[str, Any]:
    if not should_enable_hitl():
        return {}

    interrupt_config = {}
    tools_requiring_approval = getattr(settings, "hitl_tools_require_approval", [])
    allow_edit = getattr(settings, "hitl_default_allow_edit", True)
    allow_respond = getattr(settings, "hitl_default_allow_respond", True)

    for tool_name in tool_names:
        if tool_name in tools_requiring_approval:
            # Build allowed decisions list
            allowed = ["approve"]  # Always allow approval
            if allow_edit:
                allowed.append("edit")
            if allow_respond:
                allowed.append("reject")  # reject is the decision type for responding

            # Use True for all decisions, otherwise use explicit config
            if len(allowed) == 3:
                interrupt_config[tool_name] = True
            else:
                interrupt_config[tool_name] = {"allowed_decisions": allowed}

    return interrupt_config


def should_enable_hitl() -> bool:
    enabled = getattr(settings, "enable_human_in_the_loop", True)

    return enabled


def build_interrupt_response(
    interrupt_data: Any, thread_id: str, conversation_id: str
) -> Dict[str, Any]:
    """
    Build a structured InterruptResponse from raw interrupt data.

    Extracts task IDs and tool call IDs from the interrupt mechanism
    to enable proper resume functionality.
    """
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
                task.get("action")
                or task.get("tool")
                or task.get("name")
                or "unknown"
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

    # Normalize interrupt_data to a list of payloads we can inspect
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

        # New LangChain/ LangGraph HITL payload shape
        if "action_requests" in payload:
            allowed_map = _parse_review_configs(payload.get("review_configs", []))
            _add_requests(payload.get("action_requests", []), allowed_map)
            # No need to inspect fallback keys once action_requests are handled
            continue

        # Fallback support for older/alternative interrupt shapes
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


def is_agent_response_interrupted(agent_response: Dict[str, Any]) -> bool:
    if not isinstance(agent_response, dict):
        return False

    # Check for common interrupt indicators
    messages = agent_response.get("messages", [])
    if messages:
        last_message = messages[-1] if isinstance(messages, list) else messages
        if hasattr(last_message, "type") and last_message.type == "interrupt":
            return True
        if isinstance(last_message, dict):
            if last_message.get("type") == "interrupt":
                return True
            # Check for tool_calls with pending status
            tool_calls = last_message.get("tool_calls", [])
            if tool_calls and any(
                tc.get("status") == "pending"
                for tc in tool_calls
                if isinstance(tc, dict)
            ):
                return True

    # Check for interrupt marker in response metadata
    if agent_response.get("__interrupt__"):
        return True

    return False


def extract_interrupt_data_from_agent_response(agent_response: Dict[str, Any]) -> Any:
    """
    Extract interrupt data from an agent response for building InterruptResponse.
    """
    # Try __interrupt__ key first
    if "__interrupt__" in agent_response:
        return agent_response["__interrupt__"]

    # Extract from messages with tool_calls
    messages = agent_response.get("messages", [])
    if messages:
        last_message = messages[-1] if isinstance(messages, list) else messages
        if isinstance(last_message, dict):
            tool_calls = last_message.get("tool_calls", [])
            if tool_calls:
                # Filter pending tool calls
                pending_calls = [
                    tc
                    for tc in tool_calls
                    if isinstance(tc, dict) and tc.get("status") == "pending"
                ]
                if pending_calls:
                    return {"tool_calls": pending_calls}
        elif hasattr(last_message, "tool_calls"):
            return {"tool_calls": last_message.tool_calls}

    return None
