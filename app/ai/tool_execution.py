from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from anyio import ClosedResourceError, BrokenResourceError

from ..core.config import settings
from .tool_search_tool import create_tool_search_tool

from .utils import extract_content_from_result, normalize_tool_call

logger = logging.getLogger(__name__)


def extract_images_from_tool_result(result_text: str) -> List[Dict[str, str]]:
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

    extracted: List[Dict[str, str]] = []
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
    tool_call_id: Optional[str],
    tool_name: str,
    tool_args: Any,
    output_text: Optional[str],
    error: Optional[str],
    max_output_chars: int = 1000,
) -> Dict[str, Any]:
    artifact: Dict[str, Any] = {
        "tool_call_id": tool_call_id,
        "tool": tool_name,
        "args": tool_args,
        "output": None,
        "error": error,
    }

    if output_text is not None:
        artifact["output"] = (
            output_text[:max_output_chars]
            if len(output_text) > max_output_chars
            else output_text
        )

    return artifact


async def ensure_agent_tool_map(
    agent: Any,
    conversation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a tool map for executing tool calls.

    In deferred mode, the execution map is built from the same reduced tool set
    used for model binding (tool_search + pinned + loaded + required internal tools).
    In non-deferred mode, it falls back to all initialized agent tools.

    Args:
        agent: The agent instance
        conversation_id: Optional conversation ID for deferred tool lookup

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

    # In deferred mode, keep execution permissions aligned with the tools that
    # were actually exposed to the model for this turn.
    if settings.mcp_tool_search_enabled and hasattr(agent, "_get_tools_for_binding"):
        try:
            tools = agent._get_tools_for_binding(conversation_id=conversation_id)
        except TypeError:
            # Backward-compat fallback for non-keyword signatures.
            tools = agent._get_tools_for_binding(conversation_id)
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

    from .tool_context import get_tool_context
    from .deferred_tool_state import get_deferred_tool_state

    ctx = get_tool_context()
    if not ctx.conversation_id:
        return

    state = get_deferred_tool_state()
    state.mark_tool_used(ctx.conversation_id, ctx.agent_key, tool_name)


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
    tool_calls: List[Any],
    tool_map: Dict[str, Any],
    capture_images: bool = True,
    artifact_max_output_chars: int = 1000,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, str]]]:
    outputs: List[Dict[str, Any]] = []
    artifacts: List[Dict[str, Any]] = []
    images: List[Dict[str, str]] = []

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
                error_msg = (
                    f"Error: Tool {tool_name} not found. "
                    "Use tool_search first to discover and load the correct tool, "
                    "then call the discovered tool by name."
                )
            else:
                error_msg = f"Error: Tool {tool_name} not found"
            outputs.append(
                {"tool_call_id": tool_id, "name": tool_name, "content": error_msg}
            )
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

        try:
            result = await invoke_tool(tool, tool_args)
            result = extract_content_from_result(result)
            result_text = str(result)

            outputs.append(
                {"tool_call_id": tool_id, "name": tool_name, "content": result_text}
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

            # Update LRU timestamp for deferred tools on successful execution
            _mark_tool_used_if_deferred(tool_name)
        except (ClosedResourceError, BrokenResourceError) as session_exc:
            # MCP session died – attempt reconnect once, then retry
            logger.warning(
                "Session error executing tool '%s': %s. Attempting reconnect…",
                tool_name,
                session_exc,
            )
            reconnected = False
            try:
                from .mcp_integration import get_global_mcp_manager

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
                outputs.append(
                    {"tool_call_id": tool_id, "name": tool_name, "content": error_msg}
                )
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
            outputs.append(
                {"tool_call_id": tool_id, "name": tool_name, "content": error_msg}
            )
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
