from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from anyio import BrokenResourceError, ClosedResourceError

from ..core.config import settings
from .client_runtime_tools import (
    CLIENT_TOOL_PREFIX,
    get_client_tool_device_id,
    is_client_tool,
)
from .tool_search_tool import create_tool_search_tool
from .utils import extract_content_from_result, normalize_tool_call

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Tool names that may load additional tools dynamically
TOOL_LOADING_TOOLS = {"tool_search"}


def extract_images_from_tool_result(result_text: str) -> list[dict[str, str]]:
    if not result_text:
        return []

    try:
        parsed = json.loads(result_text)
    except (json.JSONDecodeError, TypeError):
        return []

    if not isinstance(parsed, dict):
        return []

    images = parsed.get("images")
    if not isinstance(images, list):
        return []

    extracted: list[dict[str, str]] = []
    for image in images:
        if not isinstance(image, dict):
            continue
        url = image.get("url")
        if not url:
            continue
        extracted.append(
            {
                "url": str(url),
                "description": str(image.get("description") or ""),
            }
        )
    return extracted


def build_tool_artifact(
    *,
    tool_call_id: str | None,
    tool_name: str,
    tool_args: Any,
    output_text: str | None,
    error: str | None,
    status: str | None = None,
    max_output_chars: int = 1000,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "tool_call_id": tool_call_id,
        "tool": tool_name,
        "args": tool_args,
        "output": None,
        "error": error,
        "status": status or ("error" if error else "success"),
    }

    if output_text is not None:
        artifact["output"] = (
            output_text[:max_output_chars] if len(output_text) > max_output_chars else output_text
        )

    return artifact


def build_rejected_tool_artifacts(
    *,
    tool_calls: list[Any],
    rejected_feedback: dict[str, str],
    max_output_chars: int = 1000,
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []

    for raw_tool_call in tool_calls:
        tool_call = normalize_tool_call(raw_tool_call)
        tool_call_id = tool_call.get("id")
        if not tool_call_id or tool_call_id not in rejected_feedback:
            continue

        artifacts.append(
            build_tool_artifact(
                tool_call_id=tool_call_id,
                tool_name=tool_call.get("name") or "unknown",
                tool_args=tool_call.get("args", {}),
                output_text=rejected_feedback[tool_call_id],
                error=None,
                status="rejected",
                max_output_chars=max_output_chars,
            )
        )

    return artifacts


def _validate_client_tool_device_binding(
    tool: Any,
    context_device_id: str | None,
    tool_name: str,
) -> str | None:
    """
    Validate that a client tool is being executed on its bound device.

    This provides defense-in-depth for client tool isolation. The primary
    validation happens inside the tool's dispatch function, but this catches
    mismatches earlier in the execution path.

    Args:
        tool: The tool being executed
        context_device_id: The device_id from the current execution context
        tool_name: The tool name (for error messages)

    Returns:
        Error message if validation fails, None if validation passes
    """
    if not is_client_tool(tool):
        return None

    bound_device_id = get_client_tool_device_id(tool)
    if not bound_device_id:
        return None

    if not context_device_id:
        return (
            f"Client tool '{tool_name}' requires a device context but none was provided. "
            "Ensure the request includes device_id."
        )

    if str(context_device_id) != str(bound_device_id):
        logger.warning(
            "Client tool device mismatch: tool '%s' bound to device %s but context has device %s",
            tool_name,
            bound_device_id,
            context_device_id,
        )
        return (
            f"Client tool '{tool_name}' is bound to a different device. "
            "This tool cannot be executed from the current device session."
        )

    return None


async def ensure_agent_tool_map(
    agent: Any,
    conversation_id: str | None = None,
    user_id: str | None = None,
    device_id: str | None = None,
) -> dict[str, Any]:
    """
    Build a tool map for executing tool calls.

    In deferred mode, the execution map is built from the same reduced tool set
    used for model binding (tool_search + pinned + loaded + required internal tools).
    In non-deferred mode, it falls back to all initialized agent tools.

    Client tools are added separately and are always scoped to a specific device.
    Tool discovery may surface both server MCP tools and device-scoped client
    tools, but execution still keeps client tools bound to the active device
    session.

    Args:
        agent: The agent instance
        conversation_id: Optional conversation ID for deferred tool lookup
        user_id: Optional user ID for client tool lookup
        device_id: Optional device ID for client tool scoping

    Returns:
        Dict mapping tool names to tool objects
    """
    if not agent:
        return {}

    initialized_tools = getattr(agent, "tools", None) or []
    if not initialized_tools:
        if hasattr(agent, "_init_mcp"):
            await agent._init_mcp()
        elif hasattr(agent, "_init_tools"):
            await agent._init_tools()
        initialized_tools = getattr(agent, "tools", None) or []

    tools = initialized_tools

    # Keep execution permissions aligned with the tools that were actually
    # exposed to the model for this turn.
    if hasattr(agent, "_get_tools_for_binding"):
        try:
            tools = agent._get_tools_for_binding(
                conversation_id=conversation_id,
                user_id=user_id,
                device_id=device_id,
            )
        except TypeError:
            # Backward-compat fallback for non-keyword signatures.
            tools = agent._get_tools_for_binding(conversation_id)
        manages_client_tools = True
    elif settings.mcp_tool_search_enabled:
        # Safety fallback for custom agents that don't implement binding helpers.
        agent_key = getattr(agent, "agent_config_key", None)
        allowlist = None
        if agent_key:
            allowlist_key = f"{agent_key}_agent_allowed_tools"
            allowlist = getattr(settings, allowlist_key, None) or []
        tool_search = create_tool_search_tool(allowlist=allowlist)
        tools = list(initialized_tools)
        if not any(getattr(t, "name", None) == tool_search.name for t in tools):
            tools.append(tool_search)
        manages_client_tools = False
    else:
        manages_client_tools = False

    # Add client runtime tools (device-scoped, separate from server MCP tools)
    if not manages_client_tools and hasattr(agent, "_get_client_runtime_tools"):
        try:
            remote_tools = agent._get_client_runtime_tools(user_id=user_id, device_id=device_id)
        except TypeError:
            remote_tools = agent._get_client_runtime_tools(user_id=user_id, device_id=device_id)
        existing_names = {getattr(t, "name", None) for t in tools}
        for tool in remote_tools:
            tool_name = getattr(tool, "name", None)
            if tool_name and tool_name not in existing_names:
                tools.append(tool)
                existing_names.add(tool_name)

    tool_map = {t.name: t for t in tools if getattr(t, "name", None)}

    return tool_map


def _mark_tool_used_if_deferred(tool_name: str) -> None:
    """
    Mark a tool as used in the deferred tool state if applicable.

    This updates the LRU timestamp so frequently-used tools are less
    likely to be evicted.

    Args:
        tool_name: The name of the tool that was executed
    """
    if not settings.mcp_tool_search_enabled:
        return

    from .deferred_tool_state import get_deferred_tool_state
    from .tool_context import get_tool_context

    ctx = get_tool_context()
    if not ctx.conversation_id:
        return

    state = get_deferred_tool_state()
    state.mark_tool_used(ctx.conversation_id, ctx.agent_key, tool_name)


async def _refresh_tool_map_after_search(
    tool_map: dict[str, Any],
    agent: Any | None,
    conversation_id: str | None,
    user_id: str | None,
    device_id: str | None,
) -> None:
    """
    Refresh the tool map after tool_search executes to include newly loaded tools.

    This is critical for multi-client scenarios where tool_search autoloads tools
    that the model then tries to call in the same turn. Without this refresh,
    the tool map wouldn't include the newly loaded tools.

    Args:
        tool_map: The existing tool map to update in-place
        agent: The agent instance (may be None)
        conversation_id: Current conversation ID
        user_id: Current user ID
        device_id: Current device ID
    """
    if not settings.mcp_tool_search_enabled:
        return

    from .client_runtime_tools import get_client_runtime_tools
    from .deferred_tool_binding import get_deferred_tools_for_binding
    from .deferred_tool_state import get_deferred_tool_state
    from .mcp_registry import get_global_mcp_manager

    if not conversation_id:
        return

    try:
        mcp_manager = await get_global_mcp_manager()

        # Get the agent key for looking up loaded tools
        agent_key = "default"
        if agent:
            agent_key = getattr(agent, "agent_config_key", None) or "default"

        # Get all MCP tools from the manager
        all_mcp_tools = await mcp_manager.get_tools() if mcp_manager else []

        # Get the newly loaded deferred server tools
        deferred_tools = get_deferred_tools_for_binding(
            conversation_id=conversation_id,
            agent_key=agent_key,
            mcp_manager=mcp_manager,
            all_tools=all_mcp_tools,
        )

        # Add any new deferred tools to the map
        for tool in deferred_tools:
            tool_name = getattr(tool, "name", None)
            if tool_name and tool_name not in tool_map:
                tool_map[tool_name] = tool
                logger.debug(
                    "Added newly loaded deferred tool '%s' to tool map",
                    tool_name,
                )

        # Also refresh client tools (they may have been loaded via tool_search)
        state = get_deferred_tool_state()
        loaded_client_tools = state.get_loaded_client_tools(conversation_id, agent_key)

        if loaded_client_tools and device_id:
            # Get fresh client tools and add any that match loaded references
            client_tools = get_client_runtime_tools(user_id=user_id, device_id=device_id)
            for client_tool in client_tools:
                tool_name = getattr(client_tool, "name", None)
                if tool_name and tool_name not in tool_map:
                    # Check if this tool was loaded by tool_search
                    for loaded in loaded_client_tools:
                        if loaded.tool_name == tool_name:
                            tool_map[tool_name] = client_tool
                            logger.debug(
                                "Added newly loaded client tool '%s' to tool map",
                                tool_name,
                            )
                            break
    except Exception as e:
        # Don't fail the entire tool execution if refresh fails
        logger.warning("Failed to refresh tool map after tool_search: %s", e)


async def invoke_tool(tool: Any, tool_args: Any) -> Any:
    """Execute a tool, preferring async paths to avoid blocking the event loop.

    Priority: coroutine attr → ainvoke → invoke (via asyncio.to_thread) → callable.
    Sync fallbacks are wrapped in ``asyncio.to_thread`` so MCP or other I/O-bound
    tools never block the running event loop.
    """
    if getattr(tool, "coroutine", None):
        return await tool.ainvoke(tool_args)

    ainvoke = getattr(tool, "ainvoke", None)
    if callable(ainvoke):
        return await ainvoke(tool_args)

    invoke = getattr(tool, "invoke", None)
    if callable(invoke):
        return await asyncio.to_thread(invoke, tool_args)

    if callable(tool):
        return await asyncio.to_thread(tool, tool_args)

    raise TypeError("Tool has no invoke/ainvoke and is not callable")


async def execute_tool_calls(
    *,
    tool_calls: list[Any],
    tool_map: dict[str, Any],
    capture_images: bool = True,
    artifact_max_output_chars: int = 1000,
    device_id: str | None = None,
    agent: Any | None = None,
    conversation_id: str | None = None,
    user_id: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    """
    Execute a list of tool calls and return outputs, artifacts, and images.

    For client tools (those with names starting with CLIENT_TOOL_PREFIX), the
    device_id parameter is used to validate that the tool is being executed
    on its bound device.

    When tool_search is among the tool calls, this function will automatically
    refresh the tool_map after tool_search executes to include newly loaded
    deferred tools. This ensures that tools discovered via tool_search can be
    called in the same turn without a second round-trip.

    Args:
        tool_calls: List of tool call objects to execute
        tool_map: Dict mapping tool names to tool objects (modified in-place if refresh needed)
        capture_images: Whether to extract images from tool results
        artifact_max_output_chars: Max chars for artifact output truncation
        device_id: Current device_id for client tool validation
        agent: Optional agent instance for tool map refresh
        conversation_id: Optional conversation ID for tool map refresh
        user_id: Optional user ID for tool map refresh

    Returns:
        Tuple of (outputs, artifacts, images)
    """
    outputs: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    images: list[dict[str, str]] = []

    for raw_tool_call in tool_calls:
        tool_call = normalize_tool_call(raw_tool_call)
        tool_name = tool_call.get("name")
        tool_id = tool_call.get("id")
        tool_args = tool_call.get("args", {})

        if not tool_name:
            error_msg = "Error: Tool name missing"
            outputs.append(
                {
                    "tool_call_id": tool_id,
                    "name": tool_name or "unknown",
                    "content": error_msg,
                }
            )
            artifacts.append(
                build_tool_artifact(
                    tool_call_id=tool_id,
                    tool_name=tool_name or "unknown",
                    tool_args=tool_args,
                    output_text=None,
                    error=error_msg,
                    max_output_chars=artifact_max_output_chars,
                )
            )
            continue

        tool = tool_map.get(tool_name)
        if not tool:
            if settings.mcp_tool_search_enabled:
                # Check if this looks like a client tool name that isn't available
                if tool_name.startswith(CLIENT_TOOL_PREFIX):
                    error_msg = (
                        f"Error: Client tool {tool_name} not found. "
                        "This tool may not be available from the current device, "
                        "or the device may have disconnected."
                    )
                else:
                    error_msg = (
                        f"Error: Tool {tool_name} not found. "
                        "Use tool_search first to discover and load the correct tool, "
                        "then call the discovered tool by name."
                    )
            else:
                error_msg = f"Error: Tool {tool_name} not found"
            outputs.append({"tool_call_id": tool_id, "name": tool_name, "content": error_msg})
            artifacts.append(
                build_tool_artifact(
                    tool_call_id=tool_id,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    output_text=None,
                    error=error_msg,
                    max_output_chars=artifact_max_output_chars,
                )
            )
            continue

        # Validate client tool device binding before execution
        device_error = _validate_client_tool_device_binding(tool, device_id, tool_name)
        if device_error:
            outputs.append(
                {"tool_call_id": tool_id, "name": tool_name, "content": f"Error: {device_error}"}
            )
            artifacts.append(
                build_tool_artifact(
                    tool_call_id=tool_id,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    output_text=None,
                    error=device_error,
                    max_output_chars=artifact_max_output_chars,
                )
            )
            continue

        try:
            result = await invoke_tool(tool, tool_args)
            result = extract_content_from_result(result)
            result_text = str(result)

            outputs.append({"tool_call_id": tool_id, "name": tool_name, "content": result_text})
            artifacts.append(
                build_tool_artifact(
                    tool_call_id=tool_id,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    output_text=result_text,
                    error=None,
                    max_output_chars=artifact_max_output_chars,
                )
            )
            if capture_images:
                images.extend(extract_images_from_tool_result(result_text))

            # Update LRU timestamp for deferred tools on successful execution
            _mark_tool_used_if_deferred(tool_name)

            # If this was a tool-loading tool (like tool_search), refresh the
            # tool map to include newly loaded deferred tools. This allows
            # subsequent tool calls in the same batch to use the discovered tools.
            if tool_name in TOOL_LOADING_TOOLS and settings.mcp_tool_search_enabled:
                await _refresh_tool_map_after_search(
                    tool_map=tool_map,
                    agent=agent,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    device_id=device_id,
                )
        except (ClosedResourceError, BrokenResourceError) as session_exc:
            # MCP session died – attempt reconnect once, then retry
            logger.warning(
                "Session error executing tool '%s': %s. Attempting reconnect…",
                tool_name,
                session_exc,
            )
            reconnected = False
            try:
                from .mcp_registry import get_global_mcp_manager

                manager = await get_global_mcp_manager()
                fresh_tool = await manager.reconnect_and_get_tool(tool_name)
                if fresh_tool:
                    # Update tool_map so later calls in the same batch also use it
                    tool_map[tool_name] = fresh_tool
                    result = await invoke_tool(fresh_tool, tool_args)
                    result = extract_content_from_result(result)
                    result_text = str(result)

                    outputs.append(
                        {
                            "tool_call_id": tool_id,
                            "name": tool_name,
                            "content": result_text,
                        }
                    )
                    artifacts.append(
                        build_tool_artifact(
                            tool_call_id=tool_id,
                            tool_name=tool_name,
                            tool_args=tool_args,
                            output_text=result_text,
                            error=None,
                            max_output_chars=artifact_max_output_chars,
                        )
                    )
                    if capture_images:
                        images.extend(extract_images_from_tool_result(result_text))
                    _mark_tool_used_if_deferred(tool_name)
                    reconnected = True
            except Exception as retry_exc:
                logger.error(
                    "Retry after reconnect failed for tool '%s': %s",
                    tool_name,
                    retry_exc,
                )

            if not reconnected:
                error_msg = (
                    f"Error: MCP session lost for tool {tool_name}. "
                    f"Reconnection failed. Please try again."
                )
                outputs.append({"tool_call_id": tool_id, "name": tool_name, "content": error_msg})
                artifacts.append(
                    build_tool_artifact(
                        tool_call_id=tool_id,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        output_text=None,
                        error=str(session_exc),
                        max_output_chars=artifact_max_output_chars,
                    )
                )
        except Exception as exc:
            error_msg = f"Error: {exc}"
            outputs.append({"tool_call_id": tool_id, "name": tool_name, "content": error_msg})
            artifacts.append(
                build_tool_artifact(
                    tool_call_id=tool_id,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    output_text=None,
                    error=str(exc),
                    max_output_chars=artifact_max_output_chars,
                )
            )

    return outputs, artifacts, images
