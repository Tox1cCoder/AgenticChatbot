"""Human-in-the-loop configuration utilities for LangGraph agents."""

import uuid
from dataclasses import dataclass
from typing import Any

from langgraph.types import Interrupt as LangGraphInterrupt

from app.ai.schemas import InterruptResponse, ToolInterruptRequest
from app.core.config import settings

CLIENT_TOOL_PREFIX = "client__"


def _tool_call_name(tool_call) -> str:
    """Best-effort tool name from a normalized dict or a tool-call object."""
    if isinstance(tool_call, dict):
        return tool_call.get("name") or ""
    return getattr(tool_call, "name", "") or ""


def is_hitl_enabled() -> bool:
    """Check if human-in-the-loop is enabled in settings."""
    return getattr(settings, "enable_human_in_the_loop", True)


def get_tools_requiring_approval() -> list[str]:
    """Get the list of tool names that require human approval."""
    return getattr(settings, "hitl_tools_require_approval", [])


def requires_human_approval(tool_names: list[str]) -> bool:
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


@dataclass(frozen=True)
class CallIdentity:
    """Resolved provenance for a single pending tool call."""

    name: str
    server_name: str | None
    qualified_tool_id: str | None
    origin: str  # "client_mcp" | "server_mcp" | "internal"


def build_global_policy() -> dict:
    """Back-compat policy derived only from process settings (no per-user rules)."""
    return {
        "master_enabled": is_hitl_enabled(),
        "servers": {},
        "tools": {},
        "global_tools": list(get_tools_requiring_approval()),
    }


def policy_from_context(context: dict | None) -> dict:
    """Return the per-turn policy stashed in graph context, or the global fallback."""
    if isinstance(context, dict):
        policy = context.get("hitl_policy")
        if isinstance(policy, dict):
            return policy
    return build_global_policy()


def resolve_call_identity(
    tool_call, *, tool_map: dict | None = None, mcp_manager=None
) -> CallIdentity:
    """Derive (name, server_name, qualified_tool_id, origin) for one tool call.

    Client tools resolve from their ``client__<server>__<tool>`` name (and metadata
    when bound); server tools resolve their server via the MCP manager.
    """
    name = _tool_call_name(tool_call)

    tool = tool_map.get(name) if (tool_map and name) else None
    meta = getattr(tool, "metadata", None)
    meta = meta if isinstance(meta, dict) else {}
    server_name = meta.get("server_name")
    qualified = meta.get("qualified_tool_id")
    origin = meta.get("tool_origin")

    if name.startswith(CLIENT_TOOL_PREFIX):
        remainder = name[len(CLIENT_TOOL_PREFIX) :]
        parsed_server, _, base = remainder.partition("__")
        server_name = server_name or (parsed_server or None)
        if not qualified and server_name and base:
            qualified = f"{server_name}::{base}"
        origin = origin or "client_mcp"
    else:
        # Server MCP tools usually carry these metadata fields after clone_mcp_tool(),
        # but fake/raw tools and older objects may not. Ambiguous deferred server
        # tools can be copied under a public alias such as "brave__search"; when
        # that alias metadata is present, recover the server from the alias prefix
        # and the raw tool name from "aliased_from_tool_name".
        aliased_from = meta.get("aliased_from_tool_name")
        if server_name is None and aliased_from and "__" in name:
            parsed_server, _, _ = name.partition("__")
            server_name = parsed_server or None
        if server_name is None and tool is not None and mcp_manager is not None:
            server_name = mcp_manager.get_server_for_tool(tool)
        base_tool_name = aliased_from or name
        if not qualified and server_name and base_tool_name:
            qualified = f"{server_name}::{base_tool_name}"
        origin = origin or ("server_mcp" if server_name else "internal")

    return CallIdentity(
        name=name, server_name=server_name, qualified_tool_id=qualified, origin=origin
    )


def identity_requires_approval(identity: CallIdentity, policy: dict) -> bool:
    """Apply the precedence ladder (tool override > server default > legacy floor)."""
    if not policy.get("master_enabled", True):
        return False

    tools = policy.get("tools") or {}
    if identity.qualified_tool_id and identity.qualified_tool_id in tools:
        return bool(tools[identity.qualified_tool_id])
    if identity.name in tools:
        return bool(tools[identity.name])

    servers = policy.get("servers") or {}
    if identity.server_name and identity.server_name in servers:
        return bool(servers[identity.server_name])

    if identity.name and identity.name in set(policy.get("global_tools") or []):
        return True

    return False


def any_call_requires_approval(
    tool_calls, *, policy: dict, tool_map: dict | None = None, mcp_manager=None
) -> bool:
    """True if any pending call requires approval under the given policy."""
    if not policy.get("master_enabled", True):
        return False
    for tool_call in tool_calls or []:
        identity = resolve_call_identity(
            tool_call, tool_map=tool_map, mcp_manager=mcp_manager
        )
        if identity_requires_approval(identity, policy):
            return True
    return False


def _parse_review_configs(data: list[dict[str, Any]] | None) -> dict[str, list[str]]:
    """Parse review configurations into a mapping of action names to allowed decisions."""
    review_configs: dict[str, list[str]] = {}
    for cfg in data or []:
        if isinstance(cfg, dict):
            action_name = cfg.get("action_name")
            allowed = cfg.get("allowed_decisions")
            if action_name and isinstance(allowed, list):
                review_configs[action_name] = allowed
    return review_configs


def _build_tool_interrupt_request(
    task: dict[str, Any],
    idx: int,
    default_prefix: str,
    allowed_map: dict[str, list[str]],
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
    tool_args = task.get("args") or task.get("tool_input") or task.get("arguments") or {}
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
    tasks: list[dict[str, Any]],
    allowed_map: dict[str, list[str]],
    default_prefix: str,
) -> list[ToolInterruptRequest]:
    """Extract ToolInterruptRequest objects from a list of tasks."""
    action_requests: list[ToolInterruptRequest] = []
    for idx, task in enumerate(tasks):
        if not isinstance(task, dict):
            continue
        request = _build_tool_interrupt_request(task, idx, default_prefix, allowed_map)
        action_requests.append(request)
    return action_requests


def build_interrupt_response(
    interrupt_data: Any, thread_id: str, conversation_id: str
) -> dict[str, Any]:
    """Build a structured InterruptResponse from raw interrupt data."""
    action_requests: list[ToolInterruptRequest] = []
    interrupt_id: str | None = None
    response_metadata: dict[str, Any] = {}

    # Normalize interrupt_data to a list of payloads
    payloads: list[Any] = []
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
            if key not in response_metadata and isinstance(value, str) and value.strip():
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
            payload.get("tasks") or payload.get("__interrupt__") or payload.get("tool_calls") or []
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
