import asyncio
import contextlib
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any
from uuid import UUID

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool

from ...core.config import settings
from ...core.runtime_modeling import (
    ResolvedRuntimeModelConfig,
    RuntimeFallbackConfig,
)
from ...interfaces.runtime_model_resolver_interface import IRuntimeModelResolver
from ...observability.conversation_compaction import conversation_compaction_metrics
from ...observability.model_usage import usage_user_hash
from ...usage import (
    NormalizedUsage,
    UsageOperation,
    begin_usage_operation,
    bind_usage_context,
    current_usage_context,
)
from ..agent_config import AGENT_CONFIG, create_gemini_client, create_langchain_model
from ..client_runtime_tools import (
    get_active_client_runtime_session,
    get_client_runtime_tools,
)
from ..context_overflow import is_context_overflow_error, prepare_aggressive_context_retry
from ..deferred_tool_binding import (
    build_deferred_tool_list,
    should_use_deferred_loading,
)
from ..image_context import build_multimodal_content, has_image_parts
from ..mcp_registry import get_global_mcp_manager, get_mcp_tools_generation
from ..model_context import build_context_window_usage, resolve_model_context_window
from ..prompts import TOOL_CONTEXT_SUFFIX, TOOL_EXPLORATION_SUFFIX
from ..request_budget import (
    BudgetConfig,
    BudgetResult,
    ContextBudgetExceededError,
    RequestBudgetService,
    RequestEnvelope,
)
from ..schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from ..skills_tool import create_activate_skill_tool, get_available_skill_summaries
from ..time_context import build_runtime_time_context_block
from ..token_counter import TokenCounter
from ..token_instrumentation import (
    compute_token_breakdown,
    estimate_output_tokens,
    extract_actual_usage,
)
from ..tool_execution import _WIDGET_SESSION_BOUND_TOOLS, _bind_widget_session_args
from ..tool_scope import is_client_only_scope
from ..user_memory_tools import create_user_memory_tools
from ..utils import (
    coerce_response_text,
    extract_inline_images_from_content,
    extract_openai_reasoning_summary,
    extract_openai_reasoning_tokens,
    extract_public_thinking_summary,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ...usage.recorder import ModelUsageRecorder

# Canvas and image_generator are accepted by the runtime resolver so the
# Planning Agent's subagent dispatch can target them with a per-task model
# override. They are NOT exposed as persistable agent_model_configs entries.
_MODEL_REQUEST_SUPPORTED_AGENT_KEYS = {
    "chat",
    "rag",
    "search",
    "planning",
    "canvas",
    "image_generator",
    # Custom agents resolve their model through the generic "custom" key with a
    # per-agent request override; not a persistable agent_model_configs entry.
    "custom",
}

_OPENAI_REASONING_SUMMARY_DISABLED_USERS: set[str] = set()

# Agents that must NOT receive widget tools.
# Widgets are for in-chat visual aids on chat/rag/search agents only.
_WIDGET_TARGET_AGENT_KEYS = {"chat", "rag", "search"}
_WIDGET_EXCLUDED_AGENT_KEYS = {"image_generator", "planning"}
_WIDGET_TOOL_NAMES = {
    "widget_create",
    "widget_update",
    "widget_get_state",
    "widget_close",
    "session_list_widgets",
}


def _get_effective_tool_allowlist(agent_key: str) -> list[str]:
    """Return the configured allowlist with widget access preserved for target agents."""
    allowlist_key = f"{agent_key}_agent_allowed_tools"
    allowlist = list(getattr(settings, allowlist_key, []) or [])
    if agent_key in _WIDGET_TARGET_AGENT_KEYS and allowlist and "widgets" not in allowlist:
        allowlist.append("widgets")
    return allowlist


def _wrap_widget_session_tool(tool: BaseTool, conversation_id: str) -> BaseTool:
    """Return a tool wrapper that forces widget session-scoped args to the active conversation."""

    async def _dispatch_widget_tool(**kwargs: Any) -> Any:
        bound_args = _bind_widget_session_args(tool.name, kwargs, conversation_id)
        return await tool.ainvoke(bound_args)

    structured_tool_kwargs: dict[str, Any] = {
        "coroutine": _dispatch_widget_tool,
        "name": tool.name,
        "description": tool.description or f"Conversation-bound wrapper for {tool.name}",
        "return_direct": bool(getattr(tool, "return_direct", False)),
        "metadata": {
            **(getattr(tool, "metadata", {}) or {}),
            "conversation_bound_widget_tool": True,
            "bound_conversation_id": str(conversation_id),
            "source_tool_name": tool.name,
        },
    }

    args_schema = getattr(tool, "args_schema", None)
    if args_schema is not None:
        structured_tool_kwargs["args_schema"] = args_schema
        structured_tool_kwargs["infer_schema"] = False

    return StructuredTool.from_function(**structured_tool_kwargs)


def _bind_widget_session_tools(
    tools: list[BaseTool],
    conversation_id: str | None,
) -> list[BaseTool]:
    """Bind widget tools to the active conversation for every execution path."""
    if not conversation_id:
        return tools

    bound_tools: list[BaseTool] = []
    for tool in tools:
        tool_name = getattr(tool, "name", "")
        if tool_name in _WIDGET_SESSION_BOUND_TOOLS:
            bound_tools.append(_wrap_widget_session_tool(tool, str(conversation_id)))
        else:
            bound_tools.append(tool)
    return bound_tools


def _breakdown_int(value: Any) -> int | None:
    """Non-boolean, non-negative int from a token-breakdown field, else None."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def _normalized_usage_from_token_breakdown(
    token_breakdown: dict[str, Any] | None,
    *,
    output_estimate: int | None = None,
    source_override: str | None = None,
) -> NormalizedUsage:
    """Convert a legacy ``TokenBudgetBreakdown`` dict into a ``NormalizedUsage``.

    Prefers the provider-reported ``actual`` block; falls back to the estimated
    total. Keeps older persisted metadata readable by the limit-aware gauge
    without forcing ``total == input + output``.

    ``output_estimate``/``source_override`` fill an absent output count from a
    local estimate (Task 8 Step 4) and relabel the source accordingly; a
    provider-reported output is never overwritten.
    """
    if isinstance(token_breakdown, dict):
        actual = token_breakdown.get("actual") or {}
        if isinstance(actual, dict):
            input_tokens = _breakdown_int(actual.get("input_tokens"))
            output_tokens = _breakdown_int(actual.get("output_tokens"))
            total_tokens = _breakdown_int(actual.get("total_tokens"))
            reasoning_tokens = _breakdown_int(actual.get("reasoning_tokens"))
            if any(
                value is not None
                for value in (input_tokens, output_tokens, total_tokens, reasoning_tokens)
            ):
                source = "provider_reported"
                if output_tokens is None and output_estimate is not None:
                    output_tokens = _breakdown_int(output_estimate)
                    source = source_override or "mixed_reported_estimated"
                return NormalizedUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    reasoning_tokens=reasoning_tokens,
                    source=source,
                )

        estimated = token_breakdown.get("estimated") or {}
        if isinstance(estimated, dict):
            estimated_total = _breakdown_int(estimated.get("total_tokens"))
            if estimated_total is not None and estimated_total > 0:
                return NormalizedUsage(total_tokens=estimated_total, source="locally_estimated")

    return NormalizedUsage(source="unavailable")


class BaseAgent(ABC):
    """Abstract base class for all agents.

    Child classes must implement: agent_type, agent_id, _get_base_system_prompt().
    """

    def __init__(
        self,
        model_name: str | None = None,
        agent_config_key: str = "chat",
        runtime_model_resolver: IRuntimeModelResolver | None = None,
        recorder: "ModelUsageRecorder | None" = None,
    ):
        self.agent_config_key = agent_config_key
        # Key used for deferred/loaded tool state. Defaults to the model config
        # key; custom agents override it to their runtime id (custom_agent:<uuid>)
        # so multiple custom agents in one conversation never share tool state.
        self.tool_state_key = agent_config_key
        self.model_name = model_name or AGENT_CONFIG[agent_config_key]["model"]
        self.runtime_model_resolver = runtime_model_resolver
        self.recorder = recorder
        self.gemini_client = None
        self.langchain_model = None
        self.mcp_manager = None
        self.tools: list[BaseTool] = []
        self._request_compaction_coordinator = None

        # Track tools generation to detect when refresh is needed
        self._tools_generation_seen: int = 0

        self._init_gemini()

    def _init_gemini(self) -> None:
        try:
            self.gemini_client = create_gemini_client()
            self.langchain_model = create_langchain_model(
                agent_type=self.agent_config_key,
                model_override=self.model_name,
            )
            logger.debug(
                f"Initialized Gemini client and LangChain model with model: {self.model_name}"
            )
        except Exception as e:
            logger.warning(
                "Default Gemini initialization unavailable for %s: %s",
                self.agent_id,
                e,
            )
            self.gemini_client = None
            self.langchain_model = None

    async def _init_tools(self) -> None:
        """
        Initialize or refresh tools from MCP manager.

        This method checks the tools_generation version from the registry
        to detect when tools need to be refreshed (e.g., after a server is
        disabled).

        Tools are also filtered based on per-agent allowlists configured in settings.
        """
        current_generation = get_mcp_tools_generation()

        # Check if we need to refresh tools
        needs_refresh = (
            self.mcp_manager is None or self._tools_generation_seen != current_generation
        )

        if not needs_refresh:
            return

        try:
            self.mcp_manager = await get_global_mcp_manager()

            all_tools = await self.mcp_manager.get_tools()

            # Deduplicate first
            unique_tools = self._deduplicate_tools(all_tools)

            # Apply per-agent tool allowlist filtering
            self.tools = self._filter_tools_by_allowlist(unique_tools)

            # Update our tracked generation
            self._tools_generation_seen = current_generation

        except Exception as e:
            logger.error(f"Error initializing MCP tools: {e}")
            self.tools = []

    def _deduplicate_tools(self, tools: list[BaseTool]) -> list[BaseTool]:
        """Deduplicate tools by provenance-aware key.

        For MCP tools, use (name, server_name) so same-name tools from
        different servers are kept distinct. For non-MCP tools, use name alone.
        The first occurrence wins when two tools share a provenance key.
        """
        seen: dict[tuple[str, str], BaseTool] = {}
        for tool in tools:
            name = getattr(tool, "name", "")
            # Try to get server identity for MCP tools; non-MCP tools use ""
            server_name = ""
            if self.mcp_manager:
                with contextlib.suppress(Exception):
                    server_name = self.mcp_manager.get_server_for_tool(tool) or ""
            key = (name, server_name)
            if key not in seen:
                seen[key] = tool
        return list(seen.values())

    def _filter_tools_by_allowlist(self, tools: list[BaseTool]) -> list[BaseTool]:
        """
        Filter tools based on per-agent allowlist configuration.

        Allowlist can contain:
        - Tool names (e.g., "tavily_search")
        - Server names (e.g., "tavily") - matches all tools from that server

        If allowlist is empty, all tools are allowed.

        Widget tools are always excluded for agents listed in
        _WIDGET_EXCLUDED_AGENT_KEYS regardless of allowlist.
        """
        exclude_widgets = self.agent_config_key in _WIDGET_EXCLUDED_AGENT_KEYS

        # Get agent-specific allowlist from settings
        allowlist = _get_effective_tool_allowlist(self.agent_config_key)

        # Empty allowlist means all tools allowed (modulo widget exclusion)
        if not allowlist:
            if exclude_widgets:
                return [t for t in tools if getattr(t, "name", "") not in _WIDGET_TOOL_NAMES]
            return tools

        # Build set of allowed names
        allowed_set = set(allowlist)

        filtered_tools = []
        for tool in tools:
            tool_name = getattr(tool, "name", "")

            # Hard widget exclusion for non-target agents
            if exclude_widgets and tool_name in _WIDGET_TOOL_NAMES:
                continue

            # Check if tool name is directly in allowlist
            if tool_name in allowed_set:
                filtered_tools.append(tool)
                continue

            # Check if tool's server is in allowlist
            if self.mcp_manager:
                server_name = self.mcp_manager.get_server_for_tool(tool)
                if server_name and server_name in allowed_set:
                    filtered_tools.append(tool)
                    continue

        if len(filtered_tools) < len(tools):
            logger.debug(
                "%s: Filtered tools from %d to %d based on allowlist %s",
                self.agent_id,
                len(tools),
                len(filtered_tools),
                allowlist,
            )

        return filtered_tools

    def _get_allowlist(self) -> list[str] | None:
        """Get the per-agent tool allowlist from settings."""
        return _get_effective_tool_allowlist(self.agent_config_key)

    def _get_skills_internal_tools(
        self,
        *,
        user_id: str | None,
        device_id: str | None,
    ) -> list[BaseTool]:
        """Return the activate_skill tool when server or client skills are available."""
        if get_available_skill_summaries(user_id=user_id, device_id=device_id):
            return [create_activate_skill_tool(user_id=user_id, device_id=device_id)]
        return []

    def _get_client_runtime_tools(
        self,
        *,
        user_id: str | None,
        device_id: str | None,
    ) -> list[BaseTool]:
        return get_client_runtime_tools(user_id=user_id, device_id=device_id)

    def _get_tools_for_binding(
        self,
        conversation_id: str | None = None,
        internal_tools: list[BaseTool] | None = None,
        user_id: str | None = None,
        device_id: str | None = None,
        tool_scope: str | None = None,
        include_hand_off: bool | None = None,
        excluded_tool_names: set[str] | frozenset[str] | None = None,
    ) -> list[BaseTool]:
        """
        Get the tools to bind to the model for this invocation.

        When mcp_tool_search_enabled is True, returns a reduced set:
        - Internal tools (if provided) + activate_skill
        - tool_search tool
        - Pinned server MCP tools
        - Loaded deferred server tools for this conversation
        - Device-scoped client runtime tools available for the active device

        When mcp_tool_search_enabled is False, returns all tools (current behavior).

        Args:
            conversation_id: Current conversation ID for deferred tool lookup
            internal_tools: Non-MCP internal tools to always include

        Returns:
            List of tools to bind to the model
        """
        # Keep always-on internal tools available even in deferred mode.
        skills_tools = self._get_skills_internal_tools(user_id=user_id, device_id=device_id)
        merged_internal: list[BaseTool] = []
        seen_internal: set[str] = set()

        def _add_internal(tool: BaseTool) -> None:
            if tool.name not in seen_internal:
                merged_internal.append(tool)
                seen_internal.add(tool.name)

        for tool in skills_tools:
            _add_internal(tool)

        # Caller-provided internal tools include the graph-scoped ``hand_off``
        # tool. The graph owns its roster, so BaseAgent never supplies a static
        # fallback that could become stale or permit self-delegation.
        for tool in internal_tools or []:
            _add_internal(tool)

        if getattr(settings, "enable_user_memory_tools", False) and user_id:
            try:
                from ...core.container import Container

                memory_repository = Container().user_memory_repository()
            except Exception as exc:
                logger.debug("User memory repository unavailable: %s", exc)
                memory_repository = None
            if memory_repository is not None:
                for tool in create_user_memory_tools(
                    repository=memory_repository, user_id=str(user_id)
                ):
                    _add_internal(tool)

        internal_tools = merged_internal or None

        use_deferred = should_use_deferred_loading(self.agent_config_key)
        client_only_scope = is_client_only_scope(device_id=device_id, tool_scope=tool_scope)
        # When bridge is enabled and device_id is absent, bind zero client tools.
        # This prevents a missing device_id (by accident rather than by design)
        # from falling through to include all loaded client tools.
        if settings.enable_client_runtime_bridge and not device_id:
            remote_tools = []
        else:
            remote_tools = self._get_client_runtime_tools(user_id=user_id, device_id=device_id)

        if use_deferred:
            # Build deferred tool list
            tools = build_deferred_tool_list(
                conversation_id=conversation_id,
                agent_key=getattr(self, "tool_state_key", None) or self.agent_config_key,
                mcp_manager=None if client_only_scope else self.mcp_manager,
                all_mcp_tools=[] if client_only_scope else self.tools,
                internal_tools=internal_tools,
                allowlist=self._get_allowlist(),
                excluded_tool_names=excluded_tool_names,
            )
            if conversation_id and remote_tools:
                from ..deferred_tool_state import get_deferred_tool_state

                active_session = get_active_client_runtime_session(
                    user_id=user_id,
                    device_id=device_id,
                )
                loaded_client_names = {
                    loaded.tool_name
                    for loaded in get_deferred_tool_state().get_loaded_client_tools(
                        conversation_id,
                        getattr(self, "tool_state_key", None) or self.agent_config_key,
                        device_id=str(device_id) if device_id else None,
                        session_id=(
                            active_session.session_id if active_session is not None else None
                        ),
                    )
                }
                remote_tools = [tool for tool in remote_tools if tool.name in loaded_client_names]
            else:
                remote_tools = []
        else:
            # Traditional mode: return all tools (with internal tools prepended)
            if client_only_scope:
                tools = list(internal_tools or [])
            elif internal_tools:
                # Combine internal tools with MCP tools, avoiding duplicates
                seen = {t.name for t in internal_tools}
                tools = list(internal_tools)
                for tool in self.tools:
                    if tool.name not in seen:
                        tools.append(tool)
                        seen.add(tool.name)
            else:
                tools = list(self.tools)

        if include_hand_off is False:
            tools = [tool for tool in tools if getattr(tool, "name", None) != "hand_off"]

        seen_names = {tool.name for tool in tools}
        for tool in remote_tools:
            if tool.name not in seen_names:
                tools.append(tool)
                seen_names.add(tool.name)

        # Apply invocation-scoped denials after merging every tool source,
        # including already-loaded client runtime tools.
        if excluded_tool_names:
            tools = [
                tool for tool in tools if getattr(tool, "name", None) not in excluded_tool_names
            ]

        return _bind_widget_session_tools(tools, conversation_id)

    def _resolve_model_request(self, model_request: dict[str, Any] | None) -> dict[str, Any] | None:
        if self.agent_config_key not in _MODEL_REQUEST_SUPPORTED_AGENT_KEYS:
            return None

        if not model_request or not isinstance(model_request, dict):
            return None

        override = model_request.get(self.agent_config_key)
        if isinstance(override, dict):
            return override

        shared = model_request.get("all")
        return shared if isinstance(shared, dict) else None

    def _get_llm_with_tools(
        self,
        model: Any = None,
        conversation_id: str | None = None,
        internal_tools: list[BaseTool] | None = None,
        user_id: str | None = None,
        device_id: str | None = None,
        include_hand_off: bool | None = None,
        excluded_tool_names: set[str] | frozenset[str] | None = None,
    ) -> Any:
        """
        Bind tools to the model for invocation.

        Args:
            model: Optional model override
            conversation_id: Conversation ID for deferred tool lookup
            internal_tools: Non-MCP internal tools to include
            include_hand_off: When ``False``, remove any graph-injected
                ``hand_off`` from the binding. Graph nodes pass a live handoff
                tool through ``internal_tools`` when delegation is available.

        Returns:
            Model with tools bound
        """
        llm = model or self.langchain_model

        # Get tools for binding (respects deferred loading setting)
        tools = self._get_tools_for_binding(
            conversation_id=conversation_id,
            internal_tools=internal_tools,
            user_id=user_id,
            device_id=device_id,
            include_hand_off=include_hand_off,
            excluded_tool_names=excluded_tool_names,
        )

        if not tools or llm is None:
            return llm

        tool_choice = getattr(settings, "tool_choice_mode", "auto")

        from ..model_factory import ModelFactory

        return ModelFactory.bind_tools_to_model(
            llm,
            tools,
            tool_choice=tool_choice,
        )

    def _parse_user_uuid(self, user_id: str | None) -> UUID | None:
        if not user_id:
            return None

        try:
            return UUID(str(user_id))
        except Exception:
            return None

    def _resolve_runtime_model_config(
        self,
        user_id: str | None,
        model_request: dict[str, Any] | None = None,
    ) -> ResolvedRuntimeModelConfig:
        request_override = self._resolve_model_request(model_request)
        user_uuid = self._parse_user_uuid(user_id)
        resolver = self.runtime_model_resolver

        if resolver and user_uuid and self.agent_config_key in _MODEL_REQUEST_SUPPORTED_AGENT_KEYS:
            return resolver.resolve_runtime_config(
                user_uuid, self.agent_config_key, request_override
            )

        warnings: list[str] = []
        if request_override:
            warnings.append(
                "Runtime model override could not be fully resolved from backend state; "
                "using default Gemini configuration."
            )

        return ResolvedRuntimeModelConfig(
            agent_key=self.agent_config_key,
            provider="gemini",
            model=self.model_name,
            temperature=float(AGENT_CONFIG.get(self.agent_config_key, {}).get("temperature", 1.0)),
            api_key=None,
            key_source="none",
            source="default",
            warnings=warnings,
            capabilities={
                "supports_vision": True,
                "supports_tool_calling": True,
                "supports_streaming": True,
                "supports_reasoning": True,
            },
        )

    def _create_fallback_runtime_config(
        self,
        fallback: RuntimeFallbackConfig | None,
        *,
        reason: str,
        from_provider: str,
        inherited_warnings: list[str] | None = None,
    ) -> ResolvedRuntimeModelConfig | None:
        if fallback is None:
            return None

        warnings = list(inherited_warnings or [])
        warnings.append(
            f"Provider fallback applied: {from_provider} -> {fallback.provider} ({reason})."
        )

        # Fallback bypasses ModelConfigService, so we resolve the context
        # window from the static registry. Provider catalog metadata is not
        # available here; the registry is good enough for the rare fallback
        # path.
        context_window = resolve_model_context_window(fallback.provider, fallback.model).to_dict()

        return ResolvedRuntimeModelConfig(
            agent_key=self.agent_config_key,
            provider=fallback.provider,
            model=fallback.model,
            temperature=fallback.temperature,
            api_key=fallback.api_key,
            key_source=fallback.key_source,
            source="fallback",
            warnings=warnings,
            capabilities={
                "supports_vision": True,
                "supports_tool_calling": True,
                "supports_streaming": True,
                "supports_reasoning": True,
            },
            provider_fallback={
                "from": from_provider,
                "to": fallback.provider,
                "reason": reason,
            },
            context_window=context_window,
        )

    def _create_langchain_model_from_runtime(
        self,
        runtime_config: ResolvedRuntimeModelConfig,
        *,
        user_id: str | None = None,
        enable_reasoning_summary: bool = True,
    ) -> tuple[Any, bool]:
        native_effort = getattr(runtime_config, "reasoning_effort", None)

        if runtime_config.provider == "gemini":
            if not runtime_config.api_key:
                if (
                    self.langchain_model is not None
                    and runtime_config.model == self.model_name
                    and not native_effort
                ):
                    return self.langchain_model, False
                raise ValueError("Gemini runtime config is missing an API key")

            thinking_level_override = None
            if native_effort:
                thinking_level_override = native_effort

            llm = create_langchain_model(
                agent_type=self.agent_config_key,
                model_override=runtime_config.model,
                temperature_override=runtime_config.temperature,
                api_key_override=runtime_config.api_key,
                thinking_level_override=thinking_level_override,
            )
            return llm, False

        from ..model_factory import ModelFactory

        include_reasoning_summary = enable_reasoning_summary
        user_key = str(user_id).strip() if user_id else ""
        if user_key and user_key in _OPENAI_REASONING_SUMMARY_DISABLED_USERS:
            include_reasoning_summary = False

        openai_kwargs: dict[str, Any] = {
            "provider": "openai",
            "model": runtime_config.model,
            "api_key": runtime_config.api_key,
            "temperature": runtime_config.temperature,
            "timeout": settings.openai_request_timeout_seconds,
            "streaming": True,
        }

        model_lower = runtime_config.model.lower()
        if native_effort:
            openai_kwargs["reasoning"] = {"effort": native_effort}
        elif include_reasoning_summary:
            if "o1" in model_lower or "o3" in model_lower:
                openai_kwargs["reasoning"] = {"effort": "medium"}
            else:
                openai_kwargs["reasoning"] = {"summary": "auto"}

        llm = ModelFactory.create_model(**openai_kwargs)
        return llm, include_reasoning_summary

    def _create_gemini_client_from_runtime(
        self, runtime_config: ResolvedRuntimeModelConfig
    ) -> Any | None:
        if runtime_config.provider != "gemini":
            return None
        if not runtime_config.api_key:
            return self.gemini_client

        return create_gemini_client(api_key_override=runtime_config.api_key)

    def _apply_runtime_metadata(
        self,
        metadata: dict[str, Any],
        runtime_config: ResolvedRuntimeModelConfig,
    ) -> None:
        metadata["provider"] = runtime_config.provider
        metadata["model"] = runtime_config.model
        metadata["key_source"] = runtime_config.key_source
        metadata["config_source"] = runtime_config.source

        if runtime_config.warnings:
            metadata["config_warnings"] = list(runtime_config.warnings)
        if runtime_config.is_custom_model:
            metadata["custom_model_override"] = True
        if runtime_config.provider_fallback:
            metadata["provider_fallback"] = runtime_config.provider_fallback
        if getattr(runtime_config, "reasoning_effort", None):
            metadata["reasoning_effort"] = runtime_config.reasoning_effort

        if runtime_config.context_window:
            metadata["context_window"] = dict(runtime_config.context_window)

    def _merge_context_window_usage(
        self,
        metadata: dict[str, Any],
        token_breakdown: dict[str, Any] | None,
        *,
        output_estimate: int | None = None,
        usage_source: str | None = None,
    ) -> None:
        """Merge token-usage fields into ``metadata['context_window']``.

        Expects ``_apply_runtime_metadata`` to have already populated the
        static context-window fields. Computes the dynamic usage fields
        (``used_tokens``, ``used_token_source``, ``usage_ratio``,
        ``display_state``) from the supplied token breakdown and merges them
        into the existing payload.

        Only ``invoke_model_with_history`` calls this — other response paths
        (chat vision, agentic RAG, planning emission) emit static context
        fields via ``_apply_runtime_metadata`` but skip usage because they
        do not assemble a ``token_breakdown``.
        """
        context_window = metadata.get("context_window")
        if not context_window:
            return

        usage = _normalized_usage_from_token_breakdown(
            token_breakdown,
            output_estimate=output_estimate,
            source_override=usage_source,
        )
        merged = dict(context_window)
        merged.update(build_context_window_usage(context_window, usage))
        metadata["context_window"] = merged

    async def _preflight_model_request(
        self,
        runtime_config: ResolvedRuntimeModelConfig,
        *,
        system_messages: list[Any],
        history_messages: list[Any],
        current_messages: list[Any],
        tools: list[Any] | None = None,
        attachments: list[Any] | None = None,
        durable_request: Any | None = None,
        emergency_compact: Any | None = None,
        conversation_id: str | None = None,
        user_id: str | None = None,
    ) -> BudgetResult | None:
        """Enforce the resolved provider's complete input budget before I/O."""
        context_window = runtime_config.context_window
        if context_window is None:
            context_window = resolve_model_context_window(
                runtime_config.provider,
                runtime_config.model,
            ).to_dict()
        max_input = context_window.get("max_input_tokens")
        if max_input is None:
            return None

        try:
            config = BudgetConfig(
                max_input_tokens=int(max_input),
                reserved_output_tokens=(
                    settings.conversation_summary_default_reserved_output_tokens
                ),
                safety_margin_tokens=settings.conversation_summary_safety_margin_tokens,
                soft_ratio=settings.conversation_summary_soft_context_ratio,
                hard_ratio=settings.conversation_summary_hard_context_ratio,
                emergency_timeout_seconds=settings.conversation_summary_timeout_seconds,
            )
        except (TypeError, ValueError) as exc:
            raise ContextBudgetExceededError("context_budget_invalid") from exc

        if conversation_id and user_id and (durable_request is None or emergency_compact is None):
            default_durable, default_emergency = self._build_compaction_callbacks(
                conversation_id,
                user_id,
            )
            durable_request = durable_request or default_durable
            emergency_compact = emergency_compact or default_emergency

        result = await RequestBudgetService(TokenCounter()).preflight(
            RequestEnvelope(
                provider=runtime_config.provider,
                model=runtime_config.model,
                system_messages=tuple(system_messages),
                history_messages=tuple(history_messages),
                current_messages=tuple(current_messages),
                tools=tuple(tools or ()),
                attachments=tuple(attachments or ()),
            ),
            config,
            durable_request=durable_request,
            emergency_compact=emergency_compact,
        )
        if result.error_code:
            raise ContextBudgetExceededError(result.error_code)
        if result.removed_groups:
            conversation_compaction_metrics.record_deterministic_trim(
                removed_groups=result.removed_groups
            )
        if result.emergency_compacted:
            conversation_compaction_metrics.record_compaction(
                mode="emergency",
                outcome="success",
                provider=runtime_config.provider,
                model=runtime_config.model,
                content_class=(
                    "mixed"
                    if tools and attachments
                    else "tools"
                    if tools
                    else "multimodal"
                    if attachments
                    else "text"
                ),
                input_tokens=result.input_tokens,
                output_tokens=0,
                duration_seconds=0,
            )
        return result

    def _build_compaction_callbacks(self, conversation_id: str, user_id: str):
        coordinator = self._get_request_compaction_coordinator()
        if coordinator is None:
            return None, None

        def request_durable():
            return coordinator.request_durable(conversation_id)

        async def compact_history(history_messages):
            reference = await coordinator.compact_now(conversation_id, user_id)
            if reference is None:
                return history_messages
            retained = []
            for message in history_messages:
                metadata = getattr(message, "additional_kwargs", {}) or {}
                if metadata.get("conversation_memory"):
                    continue
                sequence = metadata.get("sequence")
                if sequence is None or int(sequence) > reference.cursor:
                    retained.append(message)
            return [
                HumanMessage(
                    content=reference.content,
                    additional_kwargs={
                        "conversation_memory": True,
                        "memory_sequence": reference.cursor,
                    },
                ),
                *retained,
            ]

        return request_durable, compact_history

    def _get_request_compaction_coordinator(self):
        coordinator = getattr(self, "_request_compaction_coordinator", None)
        if coordinator is not None:
            return coordinator
        try:
            from app.ai.request_compaction import RequestCompactionCoordinator
            from app.core.container import get_container
            from app.workers.conversation_compaction import (
                publish_conversation_compaction,
                run_conversation_compaction_now,
            )

            self._request_compaction_coordinator = RequestCompactionCoordinator(
                repository=get_container().conversation_compaction_repository(),
                publisher=publish_conversation_compaction,
                runner=run_conversation_compaction_now,
                enabled=settings.conversation_summary_enabled,
            )
        except Exception as exc:
            logger.warning("Request compaction coordinator unavailable: %s", exc)
            return None
        return self._request_compaction_coordinator

    @staticmethod
    def _request_budget_metadata(result: BudgetResult | None) -> dict[str, Any] | None:
        if result is None:
            return None
        return {
            "action": result.action,
            "input_tokens": result.input_tokens,
            "available_input_tokens": result.available_input_tokens,
            "usage_ratio": result.usage_ratio,
            "count_strategy": result.count_strategy,
            "durable_requested": result.durable_requested,
            "emergency_compacted": result.emergency_compacted,
            "removed_groups": result.removed_groups,
        }

    async def _ainvoke_with_retries(
        self,
        llm_with_tools: Any,
        messages: list[BaseMessage],
        run_config: RunnableConfig | None = None,
        *,
        operation: UsageOperation | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> Any:
        attempts = getattr(settings, "provider_retry_attempts", 3) or 3
        delay = getattr(settings, "provider_retry_delay_seconds", 1.0) or 1.0

        try:
            attempts = int(attempts)
        except Exception:
            attempts = 3
        try:
            delay = float(delay)
        except Exception:
            delay = 1.0

        if attempts < 1:
            attempts = 1
        if delay < 0:
            delay = 0.0

        def _ainvoke() -> Any:
            if run_config is not None:
                return llm_with_tools.ainvoke(messages, run_config)
            return llm_with_tools.ainvoke(messages)

        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                # Record exactly one attempt per generic-retry iteration when a
                # usage operation is in flight; otherwise call the provider
                # directly so recorder-less construction is byte-for-byte today.
                if self.recorder is not None and operation is not None:
                    return await self.recorder.record_one_async_attempt(
                        call=_ainvoke,
                        provider=provider or "unknown",
                        model=model or "unknown",
                        operation=operation,
                        usage_transform=self._transform_recorded_usage,
                    )
                return await _ainvoke()
            except Exception as exc:
                last_exc = exc

                # Context overflows have their own single structure-aware retry
                # at the assembled-request boundary. Generic backoff cannot make
                # the same oversized payload succeed.
                if is_context_overflow_error(exc):
                    raise

                if attempt >= attempts:
                    break
                sleep_for = delay * (2 ** (attempt - 1))
                if sleep_for:
                    await asyncio.sleep(sleep_for)

        raise last_exc or RuntimeError("Provider call failed")

    def _augment_run_config_with_usage(
        self,
        run_config: RunnableConfig | None,
        operation: UsageOperation | None,
        user_id: str | None,
    ) -> RunnableConfig | None:
        """Correlate LangSmith with the usage operation without leaking identity.

        Merges the usage operation id and a keyed user hash (never the raw user
        id, email, or username) into an existing runnable config's metadata and
        tags. Returns ``run_config`` unchanged when there is nothing to add, so
        the main path that carries no config keeps calling ``ainvoke`` without
        one (byte-for-byte today).
        """
        if run_config is None or operation is None:
            return run_config

        metadata = dict(run_config.get("metadata") or {})
        metadata["usage_operation_id"] = str(operation.operation_id)
        user_hash = usage_user_hash(user_id)
        if user_hash is not None:
            metadata["usage_user_hash"] = user_hash

        usage_tag = f"usage_op:{operation.operation_id}"
        tags = list(run_config.get("tags") or [])
        if usage_tag not in tags:
            tags = [*tags, usage_tag]

        merged: RunnableConfig = dict(run_config)  # type: ignore[assignment]
        merged["metadata"] = metadata
        merged["tags"] = tags
        return merged

    def _convert_history_to_langchain_messages(
        self, conversation_history: list[Any]
    ) -> list[BaseMessage]:
        """
        Convert AgentMessage history to LangChain BaseMessage format.

        Args:
            conversation_history: List of AgentMessage objects from memory

        Returns:
            List of HumanMessage/AIMessage objects for LangChain
        """
        langchain_history = []
        for msg in conversation_history:
            if hasattr(msg, "role") and hasattr(msg, "content"):
                role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
                content = msg.content or ""
                metadata = dict(getattr(msg, "metadata", {}) or {})

                if role == "user":
                    attachments = getattr(msg, "attachments", None)
                    multimodal_content = build_multimodal_content(content, attachments)
                    if has_image_parts(multimodal_content):
                        langchain_history.append(
                            HumanMessage(
                                content=multimodal_content,
                                additional_kwargs=metadata,
                            )
                        )
                    else:
                        langchain_history.append(
                            HumanMessage(content=content, additional_kwargs=metadata)
                        )
                elif role == "assistant":
                    langchain_history.append(AIMessage(content=content, additional_kwargs=metadata))
                elif role == "memory":
                    metadata["conversation_memory"] = True
                    langchain_history.append(
                        HumanMessage(content=content, additional_kwargs=metadata)
                    )
                # Skip system messages as we add our own system prompt
        return langchain_history

    async def invoke_model_with_history(
        self,
        messages: list[BaseMessage],
        conversation_history: list[Any],
        persona: str | None,
        conversation_id: str | None = None,
        user_id: str | None = None,
        device_id: str | None = None,
        model_request: dict[str, Any] | None = None,
        disable_tools: bool = False,
        tool_budget_notice: str | None = None,
        rich_response_inventory: str | None = None,
        internal_tools: list[BaseTool] | None = None,
        run_config: RunnableConfig | None = None,
        include_hand_off: bool | None = None,
        excluded_tool_names: set[str] | frozenset[str] | None = None,
        **system_prompt_kwargs: Any,
    ) -> AgentResponse:
        try:
            durable_compaction_request = system_prompt_kwargs.pop(
                "durable_compaction_request", None
            )
            emergency_compact = system_prompt_kwargs.pop("emergency_compact", None)
            await self._init_tools()
            runtime_config = self._resolve_runtime_model_config(user_id, model_request)
            llm, openai_reasoning_summary_requested = self._create_langchain_model_from_runtime(
                runtime_config,
                user_id=user_id,
            )
            if disable_tools:
                llm_with_tools = llm
                bound_tools = []
            else:
                llm_with_tools = self._get_llm_with_tools(
                    llm,
                    conversation_id=conversation_id,
                    internal_tools=internal_tools,
                    user_id=user_id,
                    device_id=device_id,
                    include_hand_off=include_hand_off,
                    excluded_tool_names=excluded_tool_names,
                )
                bound_tools = self._get_tools_for_binding(
                    conversation_id=conversation_id,
                    internal_tools=internal_tools,
                    user_id=user_id,
                    device_id=device_id,
                    include_hand_off=include_hand_off,
                    excluded_tool_names=excluded_tool_names,
                )
            has_tool_context = any(
                isinstance(msg, ToolMessage)
                or (hasattr(msg, "tool_calls") and msg.tool_calls)
                or (hasattr(msg, "additional_kwargs") and msg.additional_kwargs.get("tool_calls"))
                for msg in messages
            )

            if tool_budget_notice:
                system_prompt_kwargs["tool_budget_notice"] = tool_budget_notice
            if rich_response_inventory:
                system_prompt_kwargs["rich_response_inventory"] = rich_response_inventory
            system_prompt_kwargs["include_hand_off"] = bool(
                include_hand_off is not False
                and any(getattr(tool, "name", None) == "hand_off" for tool in bound_tools)
            )

            system_prompt = self._build_system_prompt(
                persona,
                has_tool_context,
                user_id=user_id,
                device_id=device_id,
                **system_prompt_kwargs,
            )

            # Build message list: System + History + Current Turn
            langchain_messages = [SystemMessage(content=system_prompt)]

            # Convert and prepend conversation history (from database)
            history_messages_lc = []
            if conversation_history:
                history_messages_lc = self._convert_history_to_langchain_messages(
                    conversation_history
                )
                langchain_messages.extend(history_messages_lc)

            # Add current turn messages
            langchain_messages.extend(messages)

            budget_result = await self._preflight_model_request(
                runtime_config,
                system_messages=[langchain_messages[0]],
                history_messages=history_messages_lc,
                current_messages=messages,
                tools=bound_tools,
                durable_request=durable_compaction_request,
                emergency_compact=emergency_compact,
                conversation_id=conversation_id,
                user_id=user_id,
            )
            if budget_result is not None:
                history_messages_lc = list(budget_result.envelope.history_messages)
                langchain_messages = list(budget_result.envelope.messages)

            # === Token Instrumentation ===
            # Compute and log token breakdown for observability
            # Use bound_tools (not self.tools) to reflect actual schema tokens sent
            token_breakdown = compute_token_breakdown(
                system_prompt=system_prompt,
                history_messages=history_messages_lc,
                current_turn_messages=messages,
                tools=bound_tools if bound_tools else None,
            )

            context_overflow_retried = False

            # One usage operation spans the generic retries, the context-overflow
            # retry, and the provider fallback below — a later tool-loop
            # invocation (a new call to this method) starts a fresh operation.
            # Attribution flows through the bound UsageContext; here we only add
            # the acting agent id so per-agent rollups are correct.
            usage_operation: UsageOperation | None = None
            usage_context_cm = None
            usage_operation_cm = None
            if self.recorder is not None:
                usage_context_cm = bind_usage_context(
                    current_usage_context().child(agent_id=self.agent_id)
                )
                usage_context_cm.__enter__()
                usage_operation_cm = begin_usage_operation()
                usage_operation = usage_operation_cm.__enter__()

            usage_run_config = self._augment_run_config_with_usage(
                run_config, usage_operation, user_id
            )

            async def _invoke_with_optional_config(
                model: Any,
                model_messages: list[BaseMessage],
            ) -> Any:
                # ``runtime_config`` is read at call time so each fallback branch
                # records under its own provider/model.
                if usage_run_config is not None:
                    return await self._ainvoke_with_retries(
                        model,
                        model_messages,
                        run_config=usage_run_config,
                        operation=usage_operation,
                        provider=runtime_config.provider,
                        model=runtime_config.model,
                    )
                return await self._ainvoke_with_retries(
                    model,
                    model_messages,
                    operation=usage_operation,
                    provider=runtime_config.provider,
                    model=runtime_config.model,
                )

            try:
                try:
                    response = await _invoke_with_optional_config(
                        llm_with_tools,
                        langchain_messages,
                    )
                except Exception as exc:
                    if not settings.context_overflow_retry_enabled or not is_context_overflow_error(
                        exc
                    ):
                        raise
                    compacted_messages = prepare_aggressive_context_retry(
                        langchain_messages,
                        tool_preview_chars=(settings.context_overflow_retry_tool_preview_chars),
                    )
                    try:
                        response = await _invoke_with_optional_config(
                            llm_with_tools,
                            compacted_messages,
                        )
                    except Exception as retry_exc:
                        if is_context_overflow_error(retry_exc):
                            conversation_compaction_metrics.record_provider_overflow_retry(
                                "failure"
                            )
                            raise ContextBudgetExceededError(
                                "provider_context_overflow"
                            ) from retry_exc
                        raise
                    conversation_compaction_metrics.record_provider_overflow_retry("success")
                    context_overflow_retried = True
            except ContextBudgetExceededError:
                raise
            except Exception:
                if (
                    runtime_config.provider == "openai"
                    and openai_reasoning_summary_requested
                    and runtime_config.api_key
                ):
                    user_key = str(user_id).strip() if user_id else ""
                    if user_key:
                        _OPENAI_REASONING_SUMMARY_DISABLED_USERS.add(user_key)

                    try:
                        llm, _ = self._create_langchain_model_from_runtime(
                            runtime_config,
                            user_id=user_id,
                            enable_reasoning_summary=False,
                        )
                        llm_with_tools = (
                            llm
                            if disable_tools
                            else self._get_llm_with_tools(
                                llm,
                                conversation_id=conversation_id,
                                internal_tools=internal_tools,
                                user_id=user_id,
                                device_id=device_id,
                                include_hand_off=include_hand_off,
                            )
                        )
                        response = await _invoke_with_optional_config(
                            llm_with_tools,
                            langchain_messages,
                        )
                    except Exception:
                        fallback_runtime = self._create_fallback_runtime_config(
                            runtime_config.fallback_config,
                            reason="provider_error",
                            from_provider=runtime_config.provider,
                            inherited_warnings=runtime_config.warnings,
                        )
                        if not fallback_runtime or not fallback_runtime.api_key:
                            raise

                        runtime_config = fallback_runtime
                        budget_result = await self._preflight_model_request(
                            runtime_config,
                            system_messages=[langchain_messages[0]],
                            history_messages=history_messages_lc,
                            current_messages=messages,
                            tools=bound_tools,
                            durable_request=durable_compaction_request,
                            emergency_compact=emergency_compact,
                            conversation_id=conversation_id,
                            user_id=user_id,
                        )
                        if budget_result is not None:
                            history_messages_lc = list(budget_result.envelope.history_messages)
                            langchain_messages = list(budget_result.envelope.messages)
                        llm, _ = self._create_langchain_model_from_runtime(
                            runtime_config,
                            user_id=user_id,
                            enable_reasoning_summary=False,
                        )
                        llm_with_tools = (
                            llm
                            if disable_tools
                            else self._get_llm_with_tools(
                                llm,
                                conversation_id=conversation_id,
                                internal_tools=internal_tools,
                                user_id=user_id,
                                device_id=device_id,
                                include_hand_off=include_hand_off,
                            )
                        )
                        response = await _invoke_with_optional_config(
                            llm_with_tools,
                            langchain_messages,
                        )
                else:
                    fallback_runtime = self._create_fallback_runtime_config(
                        runtime_config.fallback_config,
                        reason="provider_error",
                        from_provider=runtime_config.provider,
                        inherited_warnings=runtime_config.warnings,
                    )
                    if not fallback_runtime or not fallback_runtime.api_key:
                        raise

                    runtime_config = fallback_runtime
                    budget_result = await self._preflight_model_request(
                        runtime_config,
                        system_messages=[langchain_messages[0]],
                        history_messages=history_messages_lc,
                        current_messages=messages,
                        tools=bound_tools,
                        durable_request=durable_compaction_request,
                        emergency_compact=emergency_compact,
                        conversation_id=conversation_id,
                        user_id=user_id,
                    )
                    if budget_result is not None:
                        history_messages_lc = list(budget_result.envelope.history_messages)
                        langchain_messages = list(budget_result.envelope.messages)
                    llm, _ = self._create_langchain_model_from_runtime(
                        runtime_config,
                        user_id=user_id,
                        enable_reasoning_summary=False,
                    )
                    llm_with_tools = (
                        llm
                        if disable_tools
                        else self._get_llm_with_tools(
                            llm,
                            conversation_id=conversation_id,
                            internal_tools=internal_tools,
                            user_id=user_id,
                            device_id=device_id,
                        )
                    )
                    response = await _invoke_with_optional_config(
                        llm_with_tools,
                        langchain_messages,
                    )
            finally:
                if usage_operation_cm is not None:
                    usage_operation_cm.__exit__(None, None, None)
                if usage_context_cm is not None:
                    usage_context_cm.__exit__(None, None, None)

            actual_usage = extract_actual_usage(response)
            if any(value is not None for value in actual_usage.values()):
                token_breakdown.actual_input_tokens = actual_usage["input_tokens"]
                token_breakdown.actual_output_tokens = actual_usage.get("output_tokens")
                token_breakdown.actual_total_tokens = actual_usage.get("total_tokens")
                token_breakdown.actual_reasoning_tokens = actual_usage.get("reasoning_tokens")
                logger.debug(
                    (
                        "%s: Actual token usage - input=%s, output=%s, "
                        "total=%s, reasoning=%s (estimated=%d)"
                    ),
                    self.agent_id,
                    actual_usage["input_tokens"],
                    actual_usage.get("output_tokens"),
                    actual_usage.get("total_tokens"),
                    actual_usage.get("reasoning_tokens"),
                    token_breakdown.total_tokens,
                )
                if budget_result is not None and actual_usage["input_tokens"] is not None:
                    conversation_compaction_metrics.record_token_calibration(
                        provider=runtime_config.provider,
                        model=runtime_config.model,
                        content_class=(
                            "mixed"
                            if bound_tools and has_tool_context
                            else "tools"
                            if bound_tools
                            else "text"
                        ),
                        estimated_tokens=budget_result.input_tokens,
                        actual_tokens=actual_usage["input_tokens"],
                    )

            tool_calls = None
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_calls = response.tool_calls

            thinking = None
            if hasattr(response, "thinking") and response.thinking:
                thinking = response.thinking
            if not thinking:
                # Gemini thought blocks are normalized to standard ``reasoning``
                # blocks so they survive the streaming pipeline, so accept both
                # shapes here. OpenAI ``reasoning`` blocks are excluded: they are
                # harvested into ``reasoning_summary`` below and must not also
                # land in ``thinking_summary``.
                thinking_block_types = {"thinking"}
                if runtime_config.provider != "openai":
                    thinking_block_types.add("reasoning")
                thinking = extract_public_thinking_summary(
                    response.content, block_types=thinking_block_types
                )

            response_text = coerce_response_text(response.content)

            reasoning_summary = None
            reasoning_tokens = actual_usage.get("reasoning_tokens")
            if runtime_config.provider == "openai":
                reasoning_summary = extract_openai_reasoning_summary(response.content)
                extracted_reasoning_tokens = extract_openai_reasoning_tokens(response)
                if extracted_reasoning_tokens is not None:
                    reasoning_tokens = extracted_reasoning_tokens

            # When the provider reported usage but omitted the output count,
            # estimate it from the response text so the gauge can show an
            # output figure. Never overwrite a reported output; the context
            # source is relabelled ``mixed_reported_estimated`` downstream.
            output_estimate: int | None = None
            usage_source_override: str | None = None
            if (
                any(value is not None for value in actual_usage.values())
                and actual_usage.get("output_tokens") is None
            ):
                output_estimate = estimate_output_tokens(
                    response_text,
                    provider=runtime_config.provider,
                    model=runtime_config.model,
                )
                if output_estimate is not None:
                    usage_source_override = "mixed_reported_estimated"

            token_breakdown_dict = token_breakdown.to_dict()
            metadata = {
                "conversation_id": conversation_id,
                "token_breakdown": token_breakdown_dict,
            }
            if context_overflow_retried:
                metadata["context_overflow_retry"] = True
            request_budget_metadata = self._request_budget_metadata(budget_result)
            if request_budget_metadata is not None:
                metadata["request_budget"] = request_budget_metadata
            self._apply_runtime_metadata(metadata, runtime_config)
            self._merge_context_window_usage(
                metadata,
                token_breakdown_dict,
                output_estimate=output_estimate,
                usage_source=usage_source_override,
            )

            if thinking:
                metadata["thinking_summary"] = str(thinking).strip()

            # Image-capable models return generated images as content blocks that
            # ``coerce_response_text`` drops. Subclasses that can act on inline
            # images (the image generator) opt in to harvest them here so they are
            # not silently lost. Transient — consumers move them to ``images``.
            if self._should_harvest_inline_images():
                inline_images = extract_inline_images_from_content(response.content)
                if inline_images:
                    metadata["response_inline_images"] = inline_images

            if isinstance(reasoning_summary, str) and reasoning_summary.strip():
                metadata["reasoning_summary"] = reasoning_summary.strip()
            if isinstance(reasoning_tokens, int) and reasoning_tokens >= 0:
                metadata["reasoning_tokens"] = reasoning_tokens

            agent_message = AgentMessage(
                role=MessageRole.ASSISTANT, content=response_text, tool_calls=tool_calls
            )

            self._augment_response_metadata(metadata)

            return AgentResponse(
                agent_type=self.agent_type,
                agent_id=self.agent_id,
                message=agent_message,
                metadata=metadata,
            )

        except ContextBudgetExceededError as e:
            logger.warning("Model request rejected before provider call code=%s", e.code)
            return self._build_error_response(
                message=(
                    "This request is too large for the selected model's context window. "
                    "Please shorten the current request or remove large attachments."
                ),
                conversation_id=conversation_id,
                error=e.code,
            )
        except Exception as e:
            logger.error(f"Error invoking model with history: {e}")
            return self._build_error_response(
                message="I encountered an error processing your request.",
                conversation_id=conversation_id,
                error=str(e),
            )

    def _augment_response_metadata(self, metadata: dict[str, Any]) -> None:
        """Hook for subclasses to inject extra response metadata.

        Base agents add nothing. ``CustomAgent`` overrides this to attach
        custom-agent identity and unavailable-tool/skill warnings.
        """
        return None

    def _should_harvest_inline_images(self) -> bool:
        """Whether to surface inline images from the model response.

        Off by default — only the image generator acts on images returned
        directly by an image-capable model.
        """
        return False

    def _transform_recorded_usage(self, response: Any, usage: NormalizedUsage) -> NormalizedUsage:
        """Allow response-aware subclasses to enrich usage before persistence."""
        return usage

    def _build_delegation_suffix(self, target_descriptions: dict[str, str] | None = None) -> str:
        """Delegation prompt suffix.

        With ``target_descriptions`` (graph-injected: base specialists +
        attached custom agents), render a dynamic, capability-aware target list
        so the agent can delegate to the right specialist — including custom
        agents it would otherwise never see. Without live targets, there is no
        handoff instruction because no handoff tool is bound.
        """
        if not target_descriptions:
            return ""

        lines = [
            f"- {target}" + (f": {description}" if description else "")
            for target, description in target_descriptions.items()
        ]
        return (
            "\n\nINTER-AGENT DELEGATION:\n"
            "You have a `hand_off` tool that transfers the conversation to a more "
            "suitable agent. Hand off when the request — or a distinct part of it — "
            "is clearly better handled by a listed specialist, especially when it "
            "needs a capability or tool you do not have. Do not delegate if you can "
            "handle the request yourself.\n"
            "Available targets:\n" + "\n".join(lines)
        )

    def _build_system_prompt(
        self,
        persona: str | None,
        has_tool_context: bool,
        **_: Any,
    ) -> str:
        system_prompt = self._get_base_system_prompt()

        user_id = _.get("user_id")
        device_id = _.get("device_id")

        # Append active client-side skills
        skills_suffix = self._build_skills_suffix(user_id=user_id, device_id=device_id)
        if skills_suffix:
            system_prompt = f"{system_prompt}{skills_suffix}"

        system_prompt = f"{system_prompt}{build_runtime_time_context_block()}"

        # Append shared tool-usage guidance.
        system_prompt = f"{system_prompt}{TOOL_EXPLORATION_SUFFIX}"

        # Append delegation instructions only when the tool is actually bound.
        hand_off_prompt_enabled = bool(_.get("include_hand_off"))
        if hand_off_prompt_enabled:
            handoff_target_descriptions = _.get("handoff_target_descriptions")
            system_prompt = (
                f"{system_prompt}{self._build_delegation_suffix(handoff_target_descriptions)}"
            )

        # Inject a multi-agent awareness block (active-agent identity, the roster
        # of reachable agents, and which agents were involved this turn) so the
        # agent can reason about — and answer questions about — the wider system.
        multi_agent_activity = _.get("multi_agent_activity")
        if multi_agent_activity:
            system_prompt = f"{system_prompt}\n\n{multi_agent_activity}"

        tool_budget_notice = _.get("tool_budget_notice")
        if tool_budget_notice:
            system_prompt = (
                f"{system_prompt}\n\nTOOL BUDGET NOTICE:\n{str(tool_budget_notice).strip()}"
            )

        rich_response_inventory = _.get("rich_response_inventory")
        if rich_response_inventory:
            system_prompt = f"{system_prompt}\n\n{str(rich_response_inventory).strip()}"

        if has_tool_context:
            system_prompt = f"{system_prompt}\n\n{TOOL_CONTEXT_SUFFIX}"

        if persona:
            system_prompt = f"Custom Persona:\n{persona.strip()}\n\n---\n{system_prompt}"

        return system_prompt

    @abstractmethod
    def _get_base_system_prompt(self) -> str:
        pass

    def _build_skills_suffix(
        self,
        *,
        user_id: str | None = None,
        device_id: str | None = None,
        allowed_skill_refs: list[dict[str, Any]] | None = None,
    ) -> str:
        """Build a suffix listing the connected client's skill summaries.

        ``allowed_skill_refs`` restricts the listed skills to a custom agent's
        selected skills (None = base-agent behavior, all skills visible).
        """
        active_skills = get_available_skill_summaries(
            user_id=user_id, device_id=device_id, allowed_skill_refs=allowed_skill_refs
        )

        if not active_skills:
            return ""

        parts = [
            "\n\n── Available Skills ──",
            "You have access to the following skills, provided by the client "
            "device connected to this chat session. Each skill contains "
            "detailed instructions that you can load on demand using the "
            "`activate_skill` tool. When a user's request seems related to "
            "a skill below, call `activate_skill` with the skill name to "
            "load its full instructions before responding.\n",
        ]
        for skill in active_skills:
            lookup_name = str(skill.get("lookup_name") or skill.get("name") or "").strip()
            source = str(skill.get("source") or "client").strip().lower()
            description = str(skill.get("description") or "").strip()
            parts.append(f"• **{lookup_name}** [{source}] – {description}")
        parts.append("\n── End Available Skills ──")
        return "\n".join(parts)

    def _get_full_system_prompt(
        self,
        *,
        user_id: str | None = None,
        device_id: str | None = None,
    ) -> str:
        """Return base system prompt + active skills suffix.

        Used by code paths that call the Gemini SDK directly (e.g. ChatAgent
        vision, RAGAgent traditional _generate) where _build_system_prompt()
        is not invoked.
        """
        base = self._get_base_system_prompt()
        suffix = self._build_skills_suffix(user_id=user_id, device_id=device_id)
        if suffix:
            base = f"{base}{suffix}"
        return f"{base}{build_runtime_time_context_block()}"

    def _build_error_response(
        self, message: str, conversation_id: str | None, error: str | None = None
    ) -> AgentResponse:
        return AgentResponse(
            agent_type=self.agent_type,
            agent_id=self.agent_id,
            message=AgentMessage(role=MessageRole.ASSISTANT, content=f"I'm sorry, but {message}"),
            metadata={
                "model": self.model_name,
                "conversation_id": conversation_id,
                "error": error or message,
            },
            error=error or message,
        )

    async def cleanup(self) -> None:
        self.mcp_manager = None
        self.tools = []
        logger.debug(f"{self.__class__.__name__} cleanup completed (MCP manager is shared)")

    @property
    @abstractmethod
    def agent_type(self) -> AgentType:
        pass

    @property
    @abstractmethod
    def agent_id(self) -> str:
        pass
