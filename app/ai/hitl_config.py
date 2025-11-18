"""
Human-in-the-loop configuration utilities for LangGraph agents.

This module provides centralized configuration for human approval workflows,
including middleware settings and interrupt response handling.
"""

import uuid
from typing import Dict, Any, List
from app.core.config import settings
from app.ai.schemas import InterruptResponse, ToolInterruptRequest


def get_hitl_middleware_config(tool_names: List[str]) -> Dict[str, Any]:
    if not should_enable_hitl():
        return {}

    interrupt_config = {}
    tools_requiring_approval = getattr(settings, "hitl_tools_require_approval", [])
    allow_edit = getattr(settings, "hitl_default_allow_edit", True)
    allow_respond = getattr(settings, "hitl_default_allow_respond", True)

    for tool_name in tool_names:
        if tool_name in tools_requiring_approval:
            interrupt_config[tool_name] = {
                "allow_accept": True,
                "allow_edit": allow_edit,
                "allow_respond": allow_respond,
            }

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
    action_requests = []

    # Handle different interrupt data structures from HumanInTheLoopMiddleware or LangGraph
    if isinstance(interrupt_data, dict):
        # Try multiple possible keys for task lists
        tasks = (
            interrupt_data.get("tasks")
            or interrupt_data.get("__interrupt__")
            or interrupt_data.get("tool_calls")
            or []
        )

        for task in tasks:
            if isinstance(task, dict):
                # Extract task/tool call identifiers
                task_id = (
                    task.get("id") or task.get("task_id") or task.get("tool_call_id")
                )
                tool_name = (
                    task.get("action")
                    or task.get("tool")
                    or task.get("name", "unknown")
                )
                tool_args = (
                    task.get("args")
                    or task.get("tool_input")
                    or task.get("arguments", {})
                )

                action_requests.append(
                    ToolInterruptRequest(
                        action=tool_name,
                        args=tool_args,
                        description=task.get("description"),
                        task_id=task_id,
                        tool_call_id=task.get("tool_call_id"),
                    )
                )

    elif isinstance(interrupt_data, list):
        # Direct list of tasks
        for task in interrupt_data:
            if isinstance(task, dict):
                task_id = (
                    task.get("id") or task.get("task_id") or task.get("tool_call_id")
                )
                tool_name = (
                    task.get("action")
                    or task.get("tool")
                    or task.get("name", "unknown")
                )
                tool_args = (
                    task.get("args")
                    or task.get("tool_input")
                    or task.get("arguments", {})
                )

                action_requests.append(
                    ToolInterruptRequest(
                        action=tool_name,
                        args=tool_args,
                        description=task.get("description"),
                        task_id=task_id,
                        tool_call_id=task.get("tool_call_id"),
                    )
                )

    interrupt_id = str(uuid.uuid4())

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
