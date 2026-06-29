from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from anyio import BrokenResourceError, ClosedResourceError

from ..core.config import settings
from ..core.rich_response import (
    GENERIC_IMAGE_ALT_TEXT,
    RichDisplayPolicy,
    RichItemType,
)
from .client_runtime_tools import (
    CLIENT_TOOL_PREFIX,
    get_active_client_runtime_session,
    get_client_runtime_tools,
    get_client_tool_device_id,
    is_client_tool,
)
from .tool_error_policy import (
    build_tool_error_payloads,
    classify_tool_error,
    should_auto_retry_tool,
)
from .tool_result_rendering import normalize_tool_result_for_rendering
from .tool_scope import is_client_only_scope
from .tool_search_tool import create_tool_search_tool
from .utils import normalize_tool_call

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Tool names that may load additional tools dynamically
TOOL_LOADING_TOOLS = {"tool_search"}
_WIDGET_ARTIFACT_TOOLS = {"widget_create", "widget_update"}
_WIDGET_SESSION_BOUND_TOOLS = {"widget_create", "session_list_widgets"}
_FULL_MODEL_HANDOFF_TOOLS = {"dispatch_subagents"}

# Render-type values that should never produce a public tool_render candidate.
# Live-widget renders have a dedicated `widget:<id>` candidate; error/text/json
# noise is not meaningful as an inline placement and would only clutter the
# inventory.
_NON_INLINE_RENDER_TYPES = {
    "live_widget",
    "error",
    "text",
    "json",
    "subagent_dispatch",
}


def _guess_mime_from_url(url: str) -> str:
    lowered = url.lower().split("?", 1)[0]
    if lowered.endswith(".jpg") or lowered.endswith(".jpeg"):
        return "image/jpeg"
    if lowered.endswith(".webp"):
        return "image/webp"
    if lowered.endswith(".gif"):
        return "image/gif"
    return "image/png"


def build_image_candidates_from_tool_result(
    result_text: str,
    *,
    tool_call_id: str | None,
    tool_name: str,
) -> list[dict[str, Any]]:
    """Build typed rich-item image candidates from a tool result payload.

    Each candidate is a public-shape dict (matching the discriminated
    ``RichItem`` schema) with deterministic id, source, provenance, and
    payload. The candidate dicts are safe to forward to
    ``build_bot_metadata()`` as transient `_rich_item_candidates`.
    """
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

    candidates: list[dict[str, Any]] = []
    for index, image in enumerate(images):
        if not isinstance(image, dict):
            continue
        url = image.get("url")
        data = image.get("data") or image.get("b64_data")
        if not url and not data:
            continue
        payload: dict[str, Any] = {}
        mime_type = image.get("mime_type") or image.get("mimeType")
        if url:
            payload["url"] = str(url)
            if not mime_type:
                mime_type = _guess_mime_from_url(str(url))
        elif data:
            payload["data"] = str(data)
            if not mime_type:
                mime_type = "image/png"
        payload["mime_type"] = str(mime_type)
        source_url = image.get("source_url")
        if source_url:
            payload["source_url"] = str(source_url)
        description = image.get("description")
        if description:
            payload["description"] = str(description)
        candidate_id_base = tool_call_id or tool_name or "tool"
        candidates.append(
            {
                "id": f"image:tool:{candidate_id_base}:{index}",
                "type": RichItemType.image.value,
                "source": "web_search" if tool_name == "tavily_search" else "tool_image",
                "display_policy": RichDisplayPolicy.inline_only.value,
                "alt_text": str(description or image.get("alt") or GENERIC_IMAGE_ALT_TEXT),
                "title": image.get("title"),
                "payload": payload,
                "provenance": {
                    "tool_call_id": tool_call_id,
                    "tool": tool_name,
                    "index": index,
                },
            }
        )
    return candidates


def build_tool_render_candidate(
    render: dict[str, Any] | None,
    *,
    tool_call_id: str | None,
    tool_name: str,
) -> dict[str, Any] | None:
    """Build a ``tool_render`` rich-item candidate from a normalized render.

    Returns ``None`` for render types that have a dedicated rich-item channel
    (live widgets) or are not useful as an inline placement (error/text/etc.).
    """
    if not isinstance(render, dict) or not tool_call_id:
        return None
    render_type = render.get("type")
    if not isinstance(render_type, str):
        return None
    if render_type in _NON_INLINE_RENDER_TYPES:
        return None
    return {
        "id": f"tool:{tool_call_id}",
        "type": RichItemType.tool_render.value,
        "source": "tool",
        "display_policy": RichDisplayPolicy.inline_or_append.value,
        "title": render.get("title"),
        "payload": {"render": render},
        "provenance": {
            "tool_call_id": tool_call_id,
            "tool": tool_name,
        },
    }


def build_live_widget_candidate_from_tool_result(
    result: str | dict[str, Any] | None,
    *,
    tool_name: str,
) -> dict[str, Any] | None:
    """Build a public widget mount record without exposing widget state."""
    if tool_name not in _WIDGET_ARTIFACT_TOOLS or result in (None, ""):
        return None
    try:
        parsed = json.loads(result) if isinstance(result, str) else result
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict) or not parsed.get("widget_id"):
        return None

    widget_id = str(parsed["widget_id"])
    return {
        "id": f"widget:{widget_id}",
        "type": RichItemType.live_widget.value,
        "source": "widget_tool",
        "display_policy": RichDisplayPolicy.inline_or_append.value,
        "title": parsed.get("title"),
        "payload": {
            "widget_id": widget_id,
            "session_id": str(parsed.get("session_id") or ""),
            "widget_type": str(parsed.get("widget_type") or ""),
            "status": str(parsed.get("status") or "active"),
            "version": int(parsed.get("version") or 1),
            "connection_endpoint": f"/widgets/{widget_id}/connection",
        },
    }


def _attach_rich_candidates_to_artifact(
    artifact: dict[str, Any],
    *,
    result_text: str,
    render: dict[str, Any] | None,
    tool_call_id: str | None,
    tool_name: str,
) -> None:
    """Compute rich-item candidates for a tool result and attach them as
    ``artifact["_rich_item_candidates"]`` for the graph layer to lift into
    ``context["rich_item_candidates"]``.
    """
    candidates: list[dict[str, Any]] = []
    candidates.extend(
        build_image_candidates_from_tool_result(
            result_text, tool_call_id=tool_call_id, tool_name=tool_name
        )
    )
    live_widget = build_live_widget_candidate_from_tool_result(
        result_text,
        tool_name=tool_name,
    )
    if live_widget is not None:
        candidates.append(live_widget)
    tool_render = build_tool_render_candidate(
        render, tool_call_id=tool_call_id, tool_name=tool_name
    )
    if tool_render is not None:
        candidates.append(tool_render)
    if candidates:
        artifact["_rich_item_candidates"] = candidates


def extract_images_from_tool_result(
    result_text: str,
    *,
    tool_call_id: str | None = None,
    tool_name: str | None = None,
) -> list[dict[str, str]]:
    """Legacy URL/description extraction used by callers that only need the
    ``tool_images`` shape. New callers should prefer
    ``build_image_candidates_from_tool_result()`` to receive typed records.

    The kwargs are optional so the existing call sites keep working; when
    provided they are ignored here (candidate construction is the candidate
    builder's responsibility).
    """
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


def _resolve_offload_service():
    """Resolve the tool result blob service from the DI container, if configured."""

    if not getattr(settings, "tool_result_offload_enabled", False):
        return None
    try:
        from ..core.container import Container

        container = Container()
        return container.tool_result_blob_service()
    except Exception as exc:
        logger.debug("Tool result offload service unavailable: %s", exc)
        return None


def apply_tool_output_offload(
    *,
    output_text: str | None,
    tool_call_id: str | None,
    tool_name: str | None,
    conversation_id: str | None,
    user_id: str | None,
    offload_service: Any | None = None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Offload large tool outputs to durable storage.

    Returns ``(public_text, blob_info)`` where ``public_text`` is the text the
    model and artifact should now show (preview + offload notice when
    offloaded) and ``blob_info`` carries the artifact metadata to merge.
    """

    if output_text is None:
        return output_text, None
    service = offload_service if offload_service is not None else _resolve_offload_service()
    if service is None or not conversation_id or not user_id:
        return output_text, None

    threshold = getattr(service, "threshold_chars", None) or getattr(
        settings, "tool_result_offload_threshold_chars", 0
    )
    if not threshold or len(output_text) <= int(threshold):
        return output_text, None

    try:
        from uuid import UUID

        offload = service.offload_if_large(
            conversation_id=UUID(str(conversation_id)),
            user_id=UUID(str(user_id)),
            tool_call_id=tool_call_id,
            tool_name=tool_name or "unknown",
            output_text=output_text,
        )
    except Exception as exc:
        logger.warning(
            "Failed to offload large tool output for %s: %s",
            tool_name,
            exc,
        )
        return output_text, None

    blob_id = offload.get("blob_id")
    if not blob_id:
        return output_text, None

    return offload.get("output", output_text), {
        "blob_id": blob_id,
        "blob_size_bytes": offload.get("size_bytes"),
        "output_truncated": True,
    }


def _apply_offload_to_outputs_and_artifacts(
    *,
    outputs: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    conversation_id: str | None,
    user_id: str | None,
    offload_service: Any | None = None,
) -> None:
    """Walk paired outputs/artifacts and offload any large outputs in-place."""

    if not outputs or not artifacts:
        return
    service = offload_service if offload_service is not None else _resolve_offload_service()
    if service is None or not conversation_id or not user_id:
        return

    artifact_index = {
        artifact.get("tool_call_id"): artifact
        for artifact in artifacts
        if artifact.get("tool_call_id")
    }

    for output in outputs:
        # The Planning supervisor must receive complete worker answers from
        # dispatch_subagents; replacing that ToolMessage with a blob preview
        # would hide the result it needs to reconcile todos.
        if output.get("name") in _FULL_MODEL_HANDOFF_TOOLS:
            continue
        tool_call_id = output.get("tool_call_id")
        public_text, blob_info = apply_tool_output_offload(
            output_text=output.get("content"),
            tool_call_id=tool_call_id,
            tool_name=output.get("name"),
            conversation_id=conversation_id,
            user_id=user_id,
            offload_service=service,
        )
        if blob_info is None:
            continue
        output["content"] = public_text
        artifact = artifact_index.get(tool_call_id)
        if artifact is not None:
            artifact["output"] = public_text
            artifact.update(blob_info)


def build_tool_artifact(
    *,
    tool_call_id: str | None,
    tool_name: str,
    tool_args: Any,
    output_text: str | None,
    error: str | None,
    status: str | None = None,
    max_output_chars: int = 1000,
    render: dict[str, Any] | None = None,
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
        if tool_name in _WIDGET_ARTIFACT_TOOLS:
            output_text = _compact_widget_artifact_output(output_text)
        artifact["output"] = (
            output_text[:max_output_chars] if len(output_text) > max_output_chars else output_text
        )

    if render is not None:
        artifact["render"] = render

    return artifact


def _compact_widget_artifact_output(output_text: str) -> str:
    """Store a compact, parseable widget descriptor in artifacts.

    Widget tool results can include a large ``state`` payload. The backend only
    needs the stable mount metadata to derive ``live_widgets``, so keep a small
    JSON object here to avoid truncating the artifact into invalid JSON.
    """
    try:
        parsed = json.loads(output_text)
    except (json.JSONDecodeError, TypeError):
        return output_text

    if not isinstance(parsed, dict) or not parsed.get("widget_id"):
        return output_text

    compact = {
        "widget_id": parsed.get("widget_id"),
        "session_id": parsed.get("session_id", ""),
        "widget_type": parsed.get("widget_type", ""),
        "title": parsed.get("title"),
        "status": parsed.get("status", "active"),
        "version": parsed.get("version", 1),
    }
    return json.dumps(compact, separators=(",", ":"), ensure_ascii=False)


def _bind_widget_session_args(
    tool_name: str,
    tool_args: Any,
    conversation_id: str | None,
) -> Any:
    """Bind widget session-scoped tools to the active conversation.

    Widget tools operate inside the current conversation. Models may still emit
    placeholders like ``current_session`` or stale IDs, so normalize those
    arguments here before the MCP tool is invoked.
    """
    if tool_name not in _WIDGET_SESSION_BOUND_TOOLS:
        return tool_args
    if not conversation_id or not isinstance(tool_args, dict):
        return tool_args

    bound_conversation_id = str(conversation_id)
    current_session_id = tool_args.get("session_id")
    if current_session_id == bound_conversation_id:
        return tool_args

    bound_args = dict(tool_args)
    bound_args["session_id"] = bound_conversation_id

    if current_session_id not in (None, "", bound_conversation_id):
        logger.debug(
            "Binding widget tool '%s' session_id from %r to active conversation %s",
            tool_name,
            current_session_id,
            bound_conversation_id,
        )

    return bound_args


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
    tool_scope: str | None = None,
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
    client_only_scope = is_client_only_scope(device_id=device_id, tool_scope=tool_scope)
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
                tool_scope=tool_scope,
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
        tools = [] if client_only_scope else list(initialized_tools)
        if not any(getattr(t, "name", None) == tool_search.name for t in tools):
            tools.append(tool_search)
        manages_client_tools = False
    else:
        tools = [] if client_only_scope else list(tools)
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
    tool_scope: str | None = None,
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
        client_only_scope = is_client_only_scope(device_id=device_id, tool_scope=tool_scope)

        # Get the agent key for looking up loaded tools. Custom agents bind
        # loaded tools under tool_state_key, so deferred-state reads must use it
        # (not the shared agent_config_key) to find the right tools.
        agent_key = "default"
        if agent:
            agent_key = (
                getattr(agent, "tool_state_key", None)
                or getattr(agent, "agent_config_key", None)
                or "default"
            )

        # Custom agents must refresh from their own restricted binding helper so
        # stale or previously loaded tools outside the current tool_refs policy
        # cannot appear in the same-turn execution map.
        if getattr(agent, "spec", None) is not None and hasattr(agent, "_get_tools_for_binding"):
            try:
                refreshed_tools = agent._get_tools_for_binding(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    device_id=device_id,
                    tool_scope=tool_scope,
                )
            except TypeError:
                refreshed_tools = agent._get_tools_for_binding(conversation_id)

            for tool in refreshed_tools:
                tool_name = getattr(tool, "name", None)
                if tool_name and tool_name not in tool_map:
                    tool_map[tool_name] = tool
            return

        # Get all MCP tools from the manager
        all_mcp_tools = await mcp_manager.get_tools() if mcp_manager else []

        # Get the newly loaded deferred server tools
        if not client_only_scope:
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
        active_session = get_active_client_runtime_session(
            user_id=user_id,
            device_id=device_id,
        )
        loaded_client_tools = state.get_loaded_client_tools(
            conversation_id,
            agent_key,
            device_id=device_id,
            session_id=active_session.session_id if active_session is not None else None,
        )

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


async def _recover_missing_tool(
    *,
    tool_name: str,
    tool_map: dict[str, Any],
    agent: Any | None,
    conversation_id: str | None,
    user_id: str | None,
    device_id: str | None,
    tool_scope: str | None = None,
) -> Any | None:
    """
    Best-effort recovery for exact-name tool calls that are absent from the
    current execution map.

    This primarily covers client tools discovered through `tool_search` where
    the model later calls the exact `client__...` name but the execution map
    was built from stale scope state.
    """
    if not tool_name:
        return None

    if settings.mcp_tool_search_enabled and conversation_id:
        await _refresh_tool_map_after_search(
            tool_map=tool_map,
            agent=agent,
            conversation_id=conversation_id,
            user_id=user_id,
            device_id=device_id,
            tool_scope=tool_scope,
        )
        recovered = tool_map.get(tool_name)
        if recovered is not None:
            return recovered

    # Server tool recovery: look up by exact name in the live MCP manager.
    # Covers the case where a server MCP tool was autoloaded via tool_search,
    # the graph paused for HITL, and the in-memory deferred state was lost
    # (server restart, process migration, or LRU eviction past the approval
    # wait window). The tool is still live on the manager, so we can rebind
    # it directly by name.
    if not tool_name.startswith(CLIENT_TOOL_PREFIX):
        if is_client_only_scope(device_id=device_id, tool_scope=tool_scope):
            return None

        # Alias-aware recovery first: the model may call the public alias
        # (e.g. "brave__search") while the live MCP manager exposes the raw name
        # ("search"). Resolve the raw name + server from the loaded deferred
        # state, then rebind under the public alias.
        try:
            from .deferred_tool_binding import _tool_with_call_name
            from .deferred_tool_state import get_deferred_tool_state

            agent_key = (
                getattr(agent, "tool_state_key", None)
                or getattr(agent, "agent_config_key", None)
                or "default"
            )
            state = get_deferred_tool_state()
            server_name = state.get_server_for_loaded_tool(conversation_id, agent_key, tool_name)
            raw_tool_name = state.get_raw_tool_name_for_loaded_tool(
                conversation_id, agent_key, tool_name
            )
            if raw_tool_name and server_name:
                from .mcp_registry import get_global_mcp_manager

                manager = await get_global_mcp_manager()
                if manager is not None:
                    for server_tool in await manager.get_tools():
                        if (
                            getattr(server_tool, "name", None) == raw_tool_name
                            and manager.get_server_for_tool(server_tool) == server_name
                        ):
                            tool_map[tool_name] = _tool_with_call_name(server_tool, tool_name)
                            logger.debug(
                                "Recovered aliased server tool '%s' (raw '%s', server '%s')",
                                tool_name,
                                raw_tool_name,
                                server_name,
                            )
                            return tool_map[tool_name]
        except Exception as exc:
            logger.warning("Failed alias-aware recovery for '%s': %s", tool_name, exc)

        if settings.mcp_tool_search_enabled and conversation_id:
            logger.debug(
                "Skipping raw server-tool recovery for unloaded deferred tool '%s'",
                tool_name,
            )
            return None

        try:
            from .mcp_registry import get_global_mcp_manager

            manager = await get_global_mcp_manager()
            if manager is not None:
                for server_tool in await manager.get_tools():
                    if getattr(server_tool, "name", None) == tool_name:
                        tool_map[tool_name] = server_tool
                        logger.debug(
                            "Recovered missing server tool '%s' from MCP manager",
                            tool_name,
                        )
                        return server_tool
        except Exception as exc:
            logger.warning("Failed recovering missing server tool '%s': %s", tool_name, exc)
        return None

    if not device_id:
        return None

    try:
        for client_tool in get_client_runtime_tools(user_id=user_id, device_id=device_id):
            candidate_name = getattr(client_tool, "name", None)
            if candidate_name != tool_name:
                continue
            tool_map[tool_name] = client_tool
            logger.debug(
                "Recovered missing client tool '%s' from active runtime catalog", tool_name
            )
            return client_tool
    except Exception as exc:
        logger.warning("Failed recovering missing client tool '%s': %s", tool_name, exc)

    return None


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


def _tool_execution_timeout_seconds() -> float:
    value = getattr(settings, "tool_execution_timeout", 30) or 30
    try:
        parsed = float(value)
    except Exception:
        return 30.0
    return max(0.001, parsed)


def _tool_execution_max_retries() -> int:
    value = getattr(settings, "tool_execution_max_retries", 0) or 0
    try:
        parsed = int(value)
    except Exception:
        return 0
    return max(0, parsed)


async def invoke_tool_with_policy(
    tool: Any,
    tool_args: Any,
    *,
    tool_name: str,
) -> tuple[Any | None, dict[str, Any] | None, str | None]:
    """Invoke a tool with a bounded timeout and conservative automatic retries.

    Returns ``(result, None, None)`` on success. On a handled failure it returns
    ``(None, artifact_detail, model_content)`` where ``model_content`` is compact
    JSON for the ToolMessage and ``artifact_detail`` carries full diagnostics.

    Session errors (``ClosedResourceError`` / ``BrokenResourceError``) are
    re-raised so the caller's MCP reconnect path can run.
    """
    timeout_seconds = _tool_execution_timeout_seconds()
    max_retries = _tool_execution_max_retries()
    attempts = 0
    last_exc: BaseException | None = None

    while attempts <= max_retries:
        attempts += 1
        try:
            result = await asyncio.wait_for(
                invoke_tool(tool, tool_args),
                timeout=timeout_seconds,
            )
            return result, None, None
        except asyncio.TimeoutError:
            last_exc = TimeoutError(f"Tool timed out after {timeout_seconds}s")
        except (ClosedResourceError, BrokenResourceError):
            raise
        except Exception as exc:
            last_exc = exc

        summary = classify_tool_error(
            last_exc,
            tool_name=tool_name,
            timeout_seconds=timeout_seconds,
            attempts=attempts,
        )
        if attempts > max_retries or not should_auto_retry_tool(
            tool,
            summary,
            tool_name=tool_name,
            retry_safe_tool_names=TOOL_LOADING_TOOLS,
        ):
            model_content, artifact_detail = build_tool_error_payloads(
                summary,
                tool_name=tool_name,
                exception=last_exc,
            )
            return None, artifact_detail, model_content

    fallback_exc = last_exc or RuntimeError("Tool failed")
    summary = classify_tool_error(
        fallback_exc,
        tool_name=tool_name,
        timeout_seconds=timeout_seconds,
        attempts=attempts,
    )
    model_content, artifact_detail = build_tool_error_payloads(
        summary,
        tool_name=tool_name,
        exception=fallback_exc,
    )
    return None, artifact_detail, model_content


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
    tool_scope: str | None = None,
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
        Tuple of (outputs, artifacts, images).

    The list of public rich-item candidates produced during execution is
    attached to each artifact as ``artifact["_rich_item_candidates"]`` so the
    graph layer can lift them into ``context["rich_item_candidates"]`` without
    a separate return-shape change. The key is intentionally prefixed with an
    underscore so existing consumers (renderers, persistence) ignore it.
    """
    outputs: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    images: list[dict[str, str]] = []

    def _append_tool_error_output(
        *,
        tool_call_id: str | None,
        tool_name: str,
        tool_args: Any,
        model_content: str,
        artifact_detail: dict[str, Any],
    ) -> None:
        """Append a compact-error output/artifact pair with an error render.

        Shared by missing-name, missing-tool, device-binding, policy-timeout,
        and exception failures so every error path renders identically.
        """
        normalized = normalize_tool_result_for_rendering(model_content, tool_name=tool_name)
        render = dict(normalized.render)
        render["type"] = "error"
        render["error"] = str(
            artifact_detail.get("diagnostic")
            or artifact_detail.get("error_type")
            or "Tool execution failed"
        )
        outputs.append(
            {
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": model_content,
                "render": render,
            }
        )
        artifact = build_tool_artifact(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_args=tool_args,
            output_text=model_content,
            error=str(artifact_detail.get("diagnostic") or artifact_detail.get("error_type")),
            max_output_chars=artifact_max_output_chars,
            render=render,
        )
        artifact.update(artifact_detail)
        artifacts.append(artifact)

    for raw_tool_call in tool_calls:
        tool_call = normalize_tool_call(raw_tool_call)
        tool_name = tool_call.get("name")
        tool_id = tool_call.get("id")
        tool_args = tool_call.get("args", {})
        tool_args = _bind_widget_session_args(tool_name or "", tool_args, conversation_id)

        if not tool_name:
            error_msg = "Error: Tool name missing"
            normalized_result = normalize_tool_result_for_rendering(
                error_msg,
                tool_name=tool_name or "unknown",
                error=error_msg,
            )
            outputs.append(
                {
                    "tool_call_id": tool_id,
                    "name": tool_name or "unknown",
                    "content": normalized_result.model_content,
                    "render": normalized_result.render,
                }
            )
            artifacts.append(
                build_tool_artifact(
                    tool_call_id=tool_id,
                    tool_name=tool_name or "unknown",
                    tool_args=tool_args,
                    output_text=normalized_result.model_content,
                    error=error_msg,
                    max_output_chars=artifact_max_output_chars,
                    render=normalized_result.render,
                )
            )
            continue

        tool = tool_map.get(tool_name)
        if not tool:
            tool = await _recover_missing_tool(
                tool_name=tool_name,
                tool_map=tool_map,
                agent=agent,
                conversation_id=conversation_id,
                user_id=user_id,
                device_id=device_id,
                tool_scope=tool_scope,
            )
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
            normalized_result = normalize_tool_result_for_rendering(
                error_msg,
                tool_name=tool_name,
                error=error_msg,
            )
            outputs.append(
                {
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": normalized_result.model_content,
                    "render": normalized_result.render,
                }
            )
            artifacts.append(
                build_tool_artifact(
                    tool_call_id=tool_id,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    output_text=normalized_result.model_content,
                    error=error_msg,
                    max_output_chars=artifact_max_output_chars,
                    render=normalized_result.render,
                )
            )
            continue

        # Validate client tool device binding before execution
        device_error = _validate_client_tool_device_binding(tool, device_id, tool_name)
        if device_error:
            normalized_result = normalize_tool_result_for_rendering(
                f"Error: {device_error}",
                tool_name=tool_name,
                error=device_error,
            )
            outputs.append(
                {
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": normalized_result.model_content,
                    "render": normalized_result.render,
                }
            )
            artifacts.append(
                build_tool_artifact(
                    tool_call_id=tool_id,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    output_text=normalized_result.model_content,
                    error=device_error,
                    max_output_chars=artifact_max_output_chars,
                    render=normalized_result.render,
                )
            )
            continue

        try:
            result, error_detail, error_content = await invoke_tool_with_policy(
                tool,
                tool_args,
                tool_name=tool_name,
            )
            if error_detail is not None:
                _append_tool_error_output(
                    tool_call_id=tool_id,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    model_content=error_content,
                    artifact_detail=error_detail,
                )
                continue

            normalized_result = normalize_tool_result_for_rendering(
                result,
                tool_name=tool_name,
            )
            result_text = normalized_result.model_content

            outputs.append(
                {
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": result_text,
                    "render": normalized_result.render,
                }
            )
            artifact = build_tool_artifact(
                tool_call_id=tool_id,
                tool_name=tool_name,
                tool_args=tool_args,
                output_text=result_text,
                error=None,
                max_output_chars=artifact_max_output_chars,
                render=normalized_result.render,
            )
            _attach_rich_candidates_to_artifact(
                artifact,
                result_text=result_text,
                render=normalized_result.render,
                tool_call_id=tool_id,
                tool_name=tool_name,
            )
            artifacts.append(artifact)
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
                    tool_scope=tool_scope,
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
                    result = await asyncio.wait_for(
                        invoke_tool(fresh_tool, tool_args),
                        timeout=_tool_execution_timeout_seconds(),
                    )
                    normalized_result = normalize_tool_result_for_rendering(
                        result,
                        tool_name=tool_name,
                    )
                    result_text = normalized_result.model_content

                    outputs.append(
                        {
                            "tool_call_id": tool_id,
                            "name": tool_name,
                            "content": result_text,
                            "render": normalized_result.render,
                        }
                    )
                    artifact = build_tool_artifact(
                        tool_call_id=tool_id,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        output_text=result_text,
                        error=None,
                        max_output_chars=artifact_max_output_chars,
                        render=normalized_result.render,
                    )
                    _attach_rich_candidates_to_artifact(
                        artifact,
                        result_text=result_text,
                        render=normalized_result.render,
                        tool_call_id=tool_id,
                        tool_name=tool_name,
                    )
                    artifacts.append(artifact)
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
                summary = classify_tool_error(
                    session_exc,
                    tool_name=tool_name,
                    timeout_seconds=_tool_execution_timeout_seconds(),
                    attempts=1,
                )
                model_content, artifact_detail = build_tool_error_payloads(
                    summary,
                    tool_name=tool_name,
                    exception=session_exc,
                )
                _append_tool_error_output(
                    tool_call_id=tool_id,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    model_content=model_content,
                    artifact_detail=artifact_detail,
                )
        except Exception as exc:
            error_msg = f"Error: {exc}"
            normalized_result = normalize_tool_result_for_rendering(
                error_msg,
                tool_name=tool_name,
                error=str(exc),
            )
            outputs.append(
                {
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": normalized_result.model_content,
                    "render": normalized_result.render,
                }
            )
            artifacts.append(
                build_tool_artifact(
                    tool_call_id=tool_id,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    output_text=normalized_result.model_content,
                    error=str(exc),
                    max_output_chars=artifact_max_output_chars,
                    render=normalized_result.render,
                )
            )

    _apply_offload_to_outputs_and_artifacts(
        outputs=outputs,
        artifacts=artifacts,
        conversation_id=conversation_id,
        user_id=user_id,
    )

    return outputs, artifacts, images
