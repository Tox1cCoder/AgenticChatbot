import asyncio
import contextlib
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
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
from ..conversation_search_tools import create_conversation_search_tools
from ..deferred_tool_binding import (
    build_deferred_tool_list,
    ordinary_excluded_tool_names,
    should_use_deferred_loading,
)
from ..image_context import build_multimodal_content, has_image_parts
from ..mcp_registry import get_global_mcp_manager, get_mcp_tools_generation
from ..model_context import build_context_window_usage, resolve_model_context_window
from ..prompts import (
    CONVERSATION_SEARCH_SUFFIX,
    MARKDOWN_CURRENCY_GUIDANCE,
    TOOL_CONTEXT_SUFFIX,
    TOOL_EXPLORATION_SUFFIX,
    USER_MEMORY_SUFFIX,
)
from ..request_budget import (
    BudgetConfig,
    BudgetResult,
    ContextBudgetExceededError,
    RequestBudgetService,
    RequestEnvelope,
)
from ..schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from ..skills_tool import (
    create_activate_skill_tool,
    create_read_skill_resource_tool,
    get_available_skill_summaries,
)
from ..time_context import build_runtime_time_context_block
from ..token_counter import EphemeralTokenCounterStore, TokenCounter
from ..token_instrumentation import (
    compute_token_breakdown,
    estimate_output_tokens,
    extract_actual_usage,
)
from ..tool_execution import _WIDGET_SESSION_BOUND_TOOLS, _bind_widget_session_args
from ..tool_result_read_tool import create_read_tool_result_tool
from ..tool_scope import ToolScope, resolve_tool_scope
from ..user_memory_tools import create_user_memory_tools
from ..utils import (
    coerce_response_text,
    extract_inline_images_from_content,
    extract_openai_reasoning_summary,
    extract_openai_reasoning_tokens,
    extract_public_thinking_summary,
)
from ..web_tools import (
    create_web_open_tool,
    create_web_search_tool,
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


_STABLE_ERROR_CODE = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,63}")


def agent_error_code(exc: BaseException) -> str:
    """What a persisted error field may say about an exception.

    A stable string ``code`` the exception declares (``ContextBudgetExceededError``
    and several provider SDK errors carry one), else the type name. Never the
    message: it is uncontrolled provider text. A ``code`` that is not a short
    identifier is ignored for the same reason.
    """
    code = getattr(exc, "code", None)
    if isinstance(code, str) and _STABLE_ERROR_CODE.fullmatch(code):
        return code
    return type(exc).__name__


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


def _unique_tools_by_name(tools: list[BaseTool]) -> list[BaseTool]:
    """The tools in order, dropping any whose name an earlier tool already has."""
    unique: list[BaseTool] = []
    seen: set[str] = set()
    for tool in tools:
        if tool.name not in seen:
            unique.append(tool)
            seen.add(tool.name)
    return unique


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


def has_tool_context(messages: list) -> bool:
    """Whether this turn already carries tool results.

    Prompts differ before and after tools have run, so the flag is computed
    from the messages rather than tracked as loop state.
    """
    for message in messages or []:
        if getattr(message, "type", None) == "tool":
            return True
        if getattr(message, "tool_calls", None):
            return True
        additional = getattr(message, "additional_kwargs", None)
        if isinstance(additional, dict) and additional.get("tool_calls"):
            return True
    return False


def _finish_reason(response: Any) -> str | None:
    """The provider's stop reason, wherever the adapter put it.

    Gemini reports it under ``finish_reason``; OpenAI under ``finish_reason``
    too but sometimes only on ``response_metadata``; a streamed message may
    carry it on ``additional_kwargs``. Read all three rather than assume one.
    """

    for holder in (
        getattr(response, "response_metadata", None),
        getattr(response, "additional_kwargs", None),
    ):
        if not isinstance(holder, dict):
            continue
        for key in ("finish_reason", "stop_reason", "finishReason"):
            value = holder.get(key)
            if value:
                return str(value)
    return None

#: Stop reasons that mean the provider failed to generate, rather than that it
#: decided something. Only these are worth sending again.
#:
#: ``MALFORMED_FUNCTION_CALL`` is Gemini reporting that the tool call it began
#: could not be parsed -- on an HTTP 200, with the whole output spent on
#: thinking and nothing usable returned. Replaying the same request succeeds,
#: which is what makes it a transient failure and not an answer.
#:
#: Deliberately not here: ``SAFETY``, ``RECITATION`` and ``PROHIBITED_CONTENT``
#: are decisions -- an identical retry is refused identically, and the user is
#: owed the real reason. ``MAX_TOKENS`` would truncate again the same way.
_RETRYABLE_GENERATION_FAILURES = frozenset({"MALFORMED_FUNCTION_CALL"})


def _failed_generation_reason(response: Any) -> str | None:
    """The provider's own "this generation failed", if nothing usable came back.

    A malformed call that still carried text or a routable tool call is a
    partial success: the turn can use it, and retrying would throw it away.
    """
    reason = _finish_reason(response)
    if not reason or str(reason).upper() not in _RETRYABLE_GENERATION_FAILURES:
        return None
    if getattr(response, "tool_calls", None):
        return None
    if coerce_response_text(getattr(response, "content", None)).strip():
        return None
    return str(reason).upper()


@dataclass(frozen=True)
class _ToolBindingRequest:
    """The caller's tool-binding inputs for one ``invoke_model_with_history`` call."""

    conversation_id: str | None
    internal_tools: list[BaseTool] | None
    user_id: str | None
    device_id: str | None
    include_hand_off: bool | None
    excluded_tool_names: set[str] | frozenset[str] | None
    disable_tools: bool


@dataclass
class _ModelTurn:
    """One model request; a provider fallback replaces its runtime and budget in place."""

    runtime_config: ResolvedRuntimeModelConfig
    binding: _ToolBindingRequest
    bound_tools: list[BaseTool]
    langchain_messages: list[BaseMessage]
    history_messages: list[BaseMessage]
    current_messages: list[BaseMessage]
    durable_request: Any
    emergency_compact: Any
    budget_result: BudgetResult | None = None
    context_overflow_retried: bool = False


_CallModel = Callable[[Any, list[BaseMessage]], Awaitable[Any]]


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
        """Return the skill activation and resource tools when skills exist.

        Both or neither: a model told at activation to read a companion file has
        no way to do so if the reader is absent.
        """
        if get_available_skill_summaries(user_id=user_id, device_id=device_id):
            return [
                create_activate_skill_tool(user_id=user_id, device_id=device_id),
                create_read_skill_resource_tool(user_id=user_id, device_id=device_id),
            ]
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
        allow_raw_web_tools: bool = False,
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
        effective_scope = resolve_tool_scope(device_id=device_id, tool_scope=tool_scope)
        client_only_scope = effective_scope is ToolScope.CLIENT_ONLY
        # The raw provider denylist joins the caller's exclusions once, here,
        # so it covers deferred discovery, pinned tools, and the traditional
        # bind-everything path alike. A tool hidden from one and not the others
        # is not hidden.
        excluded_tool_names = ordinary_excluded_tool_names(
            excluded_tool_names, allow_raw_web_tools=allow_raw_web_tools
        )

        internal_tools = (
            self._internal_tools_for_binding(
                internal_tools,
                effective_scope=effective_scope,
                conversation_id=conversation_id,
                user_id=user_id,
                device_id=device_id,
            )
            or None
        )

        use_deferred = should_use_deferred_loading(self.agent_config_key)
        remote_tools = self._client_tools_for_binding(user_id=user_id, device_id=device_id)

        if use_deferred:
            tools = build_deferred_tool_list(
                conversation_id=conversation_id,
                agent_key=getattr(self, "tool_state_key", None) or self.agent_config_key,
                mcp_manager=None if client_only_scope else self.mcp_manager,
                all_mcp_tools=[] if client_only_scope else self.tools,
                internal_tools=internal_tools,
                allowlist=self._get_allowlist(),
                excluded_tool_names=excluded_tool_names,
                allow_raw_web_tools=allow_raw_web_tools,
            )
            remote_tools = self._loaded_client_tools(
                remote_tools,
                conversation_id=conversation_id,
                user_id=user_id,
                device_id=device_id,
            )
        else:
            tools = self._all_tools_for_binding(internal_tools, client_only_scope=client_only_scope)

        return self._finish_tool_binding(
            tools,
            remote_tools,
            conversation_id=conversation_id,
            include_hand_off=include_hand_off,
            excluded_tool_names=excluded_tool_names,
        )

    # ------------------------------------------------ _get_tools_for_binding parts

    def _internal_tools_for_binding(
        self,
        internal_tools: list[BaseTool] | None,
        *,
        effective_scope: ToolScope,
        conversation_id: str | None,
        user_id: str | None,
        device_id: str | None,
    ) -> list[BaseTool]:
        """Always-on internal tools, kept even in deferred mode; the first name wins."""
        candidates: list[BaseTool] = list(
            self._get_skills_internal_tools(user_id=user_id, device_id=device_id)
        )

        # A preview-only tool result is unusable without a reader, and the model's
        # only other recovery is to repeat the search that produced it.
        if getattr(settings, "tool_result_offload_enabled", False):
            candidates.append(create_read_tool_result_tool())

        # Web work is server-owned and split three ways: discovery, focused
        # page extraction, and image discovery. Each spends its own budget, and
        # selected images reach the answer only through the rich-item inventory.
        if (
            self.agent_config_key in {"chat", "search"}
            and effective_scope is not ToolScope.CLIENT_ONLY
        ):
            for factory in (
                create_web_search_tool,
                create_web_open_tool,
            ):
                candidates.append(factory(tool_scope=effective_scope.value))

        # Caller-provided internal tools include the graph-scoped ``hand_off``
        # tool. The graph owns its roster, so BaseAgent never supplies a static
        # fallback that could become stale or permit self-delegation.
        candidates.extend(internal_tools or [])
        candidates.extend(self._user_memory_tools(user_id, conversation_id))
        candidates.extend(self._conversation_search_tools(user_id, conversation_id))
        return _unique_tools_by_name(candidates)

    @staticmethod
    def _user_memory_tools(user_id: str | None, conversation_id: str | None) -> list[BaseTool]:
        if not (getattr(settings, "enable_user_memory_tools", False) and user_id):
            return []
        try:
            from ...core.container import Container

            memory_repository = Container().user_memory_repository()
        except Exception as exc:
            logger.debug("User memory repository unavailable: %s", exc)
            return []
        if memory_repository is None:
            return []
        return list(
            create_user_memory_tools(
                repository=memory_repository,
                user_id=str(user_id),
                conversation_id=str(conversation_id) if conversation_id else None,
            )
        )

    @staticmethod
    def _conversation_search_tools(
        user_id: str | None, conversation_id: str | None
    ) -> list[BaseTool]:
        if not (getattr(settings, "enable_conversation_search_tools", False) and user_id):
            return []
        try:
            from ...core.container import Container

            search_repository = Container().conversation_search_repository()
        except Exception as exc:
            logger.debug("Conversation search repository unavailable: %s", exc)
            return []
        if search_repository is None:
            return []
        return list(
            create_conversation_search_tools(
                repository=search_repository,
                user_id=str(user_id),
                conversation_id=str(conversation_id) if conversation_id else None,
            )
        )

    def _client_tools_for_binding(
        self, *, user_id: str | None, device_id: str | None
    ) -> list[BaseTool]:
        # When bridge is enabled and device_id is absent, bind zero client tools.
        # This prevents a missing device_id (by accident rather than by design)
        # from falling through to include all loaded client tools.
        if settings.enable_client_runtime_bridge and not device_id:
            return []
        return self._get_client_runtime_tools(user_id=user_id, device_id=device_id)

    def _loaded_client_tools(
        self,
        remote_tools: list[BaseTool],
        *,
        conversation_id: str | None,
        user_id: str | None,
        device_id: str | None,
    ) -> list[BaseTool]:
        """In deferred mode, only the client tools ``tool_search`` loaded for this session."""
        if not (conversation_id and remote_tools):
            return []
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
                session_id=(active_session.session_id if active_session is not None else None),
                user_id=str(user_id) if user_id else None,
            )
        }
        return [tool for tool in remote_tools if tool.name in loaded_client_names]

    def _all_tools_for_binding(
        self,
        internal_tools: list[BaseTool] | None,
        *,
        client_only_scope: bool,
    ) -> list[BaseTool]:
        """Traditional mode: every tool, with the internal tools first."""
        if client_only_scope:
            return list(internal_tools or [])
        if internal_tools:
            # Internal tools are already unique by name; MCP duplicates drop out.
            return _unique_tools_by_name([*internal_tools, *self.tools])
        return list(self.tools)

    @staticmethod
    def _finish_tool_binding(
        tools: list[BaseTool],
        remote_tools: list[BaseTool],
        *,
        conversation_id: str | None,
        include_hand_off: bool | None,
        excluded_tool_names: set[str] | frozenset[str] | None,
    ) -> list[BaseTool]:
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

        context_window = fallback.context_window or resolve_model_context_window(
            fallback.provider, fallback.model
        ).to_dict()

        return ResolvedRuntimeModelConfig(
            agent_key=self.agent_config_key,
            provider=fallback.provider,
            model=fallback.model,
            temperature=fallback.temperature,
            api_key=fallback.api_key,
            key_source=fallback.key_source,
            source="fallback",
            warnings=warnings,
            capabilities=dict(fallback.capabilities),
            provider_fallback={
                "from": from_provider,
                "to": fallback.provider,
                "reason": reason,
            },
            reasoning_effort=fallback.reasoning_effort,
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

        model_kwargs: dict[str, Any] = {
            "timeout": settings.openai_request_timeout_seconds,
            "streaming": True,
        }

        model_lower = runtime_config.model.lower()
        if not native_effort and include_reasoning_summary:
            if "o1" in model_lower or "o3" in model_lower:
                model_kwargs["reasoning"] = {"effort": "medium"}
            else:
                model_kwargs["reasoning"] = {"summary": "auto"}

        llm = ModelFactory.create_model_from_runtime(runtime_config, **model_kwargs)
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
        authoritative_allowance: bool = False,
        token_counter: TokenCounter | None = None,
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

        result = await RequestBudgetService(token_counter or TokenCounter()).preflight(
            RequestEnvelope(
                provider=runtime_config.provider,
                model=runtime_config.model,
                system_messages=tuple(system_messages),
                history_messages=tuple(history_messages),
                current_messages=tuple(current_messages),
                tools=tuple(tools or ()),
                attachments=tuple(attachments or ()),
                authoritative_allowance=authoritative_allowance,
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

    @staticmethod
    def _token_counter_for_model(provider: str, llm: Any) -> TokenCounter:
        """Use Gemini's async structured request counter when this SDK supports it.

        ``ChatGoogleGenerativeAI.get_num_tokens`` is a synchronous provider RPC
        over one synthetic text part. It neither represents the request sent by
        the chat model nor belongs on the event-loop thread, so it is deliberately
        not used here. The guarded adapters below accept only the installed
        wrapper shape that can prepare the real provider contents/config and
        expose the SDK's async ``count_tokens`` method. Any shape drift or runtime
        failure falls back to the local conservative counter without claiming
        provider authority.

        Two counters are registered for different jobs: the request counter is
        the one authoritative full-request count per model call, and the text
        counter is the one exact reconciliation of an assembled evidence pack.
        Neither is reachable from the per-candidate fit loop.
        """
        provider_key = str(provider or "").strip().casefold()
        if provider_key != "gemini":
            return TokenCounter()

        prepare_request = getattr(llm, "_prepare_request", None)
        try:
            async_client = getattr(llm, "async_client", None)
        except Exception:
            async_client = None
        count_tokens = getattr(getattr(async_client, "models", None), "count_tokens", None)
        if not callable(prepare_request) or not callable(count_tokens):
            return TokenCounter()

        def native_timeout() -> float:
            return max(
                0.1,
                float(getattr(settings, "conversation_summary_timeout_seconds", 10.0)),
            )

        async def native_text(*, model: str, text: str) -> int:
            # Mirrors the SDK's own single-text count shape; the SDK converts a
            # bare string into one user Content part.
            response = await asyncio.wait_for(
                count_tokens(
                    model=str(getattr(llm, "model", None) or model),
                    contents=str(text),
                ),
                timeout=native_timeout(),
            )
            total_tokens = getattr(response, "total_tokens", None)
            if total_tokens is None:
                raise ValueError("gemini_native_count_missing_total")
            return int(total_tokens)

        async def native_request(*, model: str, messages, tools, attachments) -> int:
            del model
            if attachments:
                # This layer cannot prove how detached attachments are projected
                # into provider parts. Refuse an inexact native claim; the caller
                # will use the explicit conservative local estimate instead.
                raise ValueError("gemini_native_count_detached_attachments_unsupported")

            request = prepare_request(
                list(messages),
                tools=list(tools) if tools else None,
            )
            if not isinstance(request, dict):
                raise ValueError("gemini_native_count_request_shape_unsupported")
            request_model = request.get("model")
            contents = request.get("contents")
            request_config = request.get("config")
            if not request_model or contents is None:
                raise ValueError("gemini_native_count_request_shape_unsupported")

            count_config: dict[str, Any] = {}
            for field in ("system_instruction", "tools"):
                value = (
                    request_config.get(field)
                    if isinstance(request_config, dict)
                    else getattr(request_config, field, None)
                )
                if value is not None:
                    count_config[field] = value

            response = await asyncio.wait_for(
                count_tokens(
                    model=request_model,
                    contents=contents,
                    config=count_config or None,
                ),
                timeout=native_timeout(),
            )
            total_tokens = getattr(response, "total_tokens", None)
            if total_tokens is None:
                raise ValueError("gemini_native_count_missing_total")
            return int(total_tokens)

        return TokenCounter(
            native_counters={provider_key: native_request},
            native_text_counters={provider_key: native_text},
        )

    def _register_evidence_token_counter(
        self,
        counter: TokenCounter,
        *,
        provider: str,
        model: str,
    ) -> dict[str, str]:
        """Keep a live counter out of checkpointed response metadata."""
        store = getattr(self, "_ephemeral_evidence_counters", None)
        if not isinstance(store, EphemeralTokenCounterStore):
            store = EphemeralTokenCounterStore()
            self._ephemeral_evidence_counters = store
        return {
            "reference": store.put(counter),
            "provider": str(provider or ""),
            "model": str(model or ""),
            "fallback": "deterministic_local_conservative",
        }

    def _take_evidence_token_counter(
        self,
        descriptor: dict[str, Any] | None,
        *,
        provider: str,
        model: str,
    ) -> TokenCounter:
        """Consume a live counter once, or reconstruct the safe restart fallback."""
        reference = descriptor.get("reference") if isinstance(descriptor, dict) else None
        identity_matches = bool(
            isinstance(descriptor, dict)
            and str(descriptor.get("provider") or "") == str(provider or "")
            and str(descriptor.get("model") or "") == str(model or "")
        )
        store = getattr(self, "_ephemeral_evidence_counters", None)
        counter = store.take(reference) if isinstance(store, EphemeralTokenCounterStore) else None
        if identity_matches and counter is not None:
            return counter
        return TokenCounter()

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
            "evidence_token_allowance": result.evidence_token_allowance,
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
        last_failed_generation: Any = None
        for attempt in range(1, attempts + 1):
            try:
                # Record exactly one attempt per generic-retry iteration when a
                # usage operation is in flight; otherwise call the provider
                # directly so recorder-less construction is byte-for-byte today.
                if self.recorder is not None and operation is not None:
                    response = await self.recorder.record_one_async_attempt(
                        call=_ainvoke,
                        provider=provider or "unknown",
                        model=model or "unknown",
                        operation=operation,
                        usage_transform=self._transform_recorded_usage,
                    )
                else:
                    response = await _ainvoke()

                # A 200 that declares its own generation failed is not a
                # success with nothing in it. Only an exception used to reach
                # the backoff below, so this arrived downstream as an ordinary
                # empty response and failed the turn `empty_public_content`.
                if _failed_generation_reason(response) is None:
                    return response
                last_failed_generation = response
                if attempt >= attempts:
                    # Returned rather than raised: the turn's own diagnostics
                    # report the finish reason, and inventing an exception here
                    # would replace that with a less specific failure.
                    return response
                logger.warning(
                    "%s: provider reported %s with nothing usable; retrying (%s/%s)",
                    self.agent_id,
                    _failed_generation_reason(response),
                    attempt,
                    attempts,
                )
                sleep_for = delay * (2 ** (attempt - 1))
                if sleep_for:
                    await asyncio.sleep(sleep_for)
                continue
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

        if last_failed_generation is not None:
            return last_failed_generation
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
            binding = _ToolBindingRequest(
                conversation_id=conversation_id,
                internal_tools=internal_tools,
                user_id=user_id,
                device_id=device_id,
                include_hand_off=include_hand_off,
                excluded_tool_names=excluded_tool_names,
                disable_tools=disable_tools,
            )
            runtime_config = self._resolve_runtime_model_config(user_id, model_request)
            llm, openai_reasoning_summary_requested = self._create_langchain_model_from_runtime(
                runtime_config,
                user_id=user_id,
            )
            llm_with_tools, bound_tools = self._bind_turn_tools(llm, binding)
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

            langchain_messages, history_messages_lc = self._assemble_turn_messages(
                system_prompt, conversation_history, messages
            )
            turn = _ModelTurn(
                runtime_config=runtime_config,
                binding=binding,
                bound_tools=bound_tools,
                langchain_messages=langchain_messages,
                history_messages=history_messages_lc,
                current_messages=messages,
                durable_request=durable_compaction_request,
                emergency_compact=emergency_compact,
            )
            await self._preflight_turn(turn, llm)

            # === Token Instrumentation ===
            # Compute and log token breakdown for observability
            # Use bound_tools (not self.tools) to reflect actual schema tokens sent
            token_breakdown = compute_token_breakdown(
                system_prompt=system_prompt,
                history_messages=turn.history_messages,
                current_turn_messages=messages,
                tools=bound_tools if bound_tools else None,
            )

            response = await self._invoke_turn_under_usage_operation(
                turn,
                llm_with_tools,
                run_config=run_config,
                reasoning_summary_requested=openai_reasoning_summary_requested,
            )
            return self._build_turn_response(
                turn, response, token_breakdown, has_tool_context=has_tool_context
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
            logger.error("Error invoking model with history: %s", e)
            return self._build_error_response(
                message="I encountered an error processing your request.",
                conversation_id=conversation_id,
                error=e,
            )

    # ------------------------------------------- invoke_model_with_history parts

    def _bind_turn_tools(
        self, llm: Any, binding: _ToolBindingRequest
    ) -> tuple[Any, list[BaseTool]]:
        """The model with its tools bound, and the tool list that binding used."""
        if binding.disable_tools:
            return llm, []
        llm_with_tools = self._get_llm_with_tools(
            llm,
            conversation_id=binding.conversation_id,
            internal_tools=binding.internal_tools,
            user_id=binding.user_id,
            device_id=binding.device_id,
            include_hand_off=binding.include_hand_off,
            excluded_tool_names=binding.excluded_tool_names,
        )
        bound_tools = self._get_tools_for_binding(
            conversation_id=binding.conversation_id,
            internal_tools=binding.internal_tools,
            user_id=binding.user_id,
            device_id=binding.device_id,
            include_hand_off=binding.include_hand_off,
            excluded_tool_names=binding.excluded_tool_names,
        )
        return llm_with_tools, bound_tools

    def _rebind_turn_tools(
        self,
        llm: Any,
        binding: _ToolBindingRequest,
        *,
        include_hand_off: bool | None,
    ) -> Any:
        """Bind tools to a replacement model after a provider error."""
        if binding.disable_tools:
            return llm
        return self._get_llm_with_tools(
            llm,
            conversation_id=binding.conversation_id,
            internal_tools=binding.internal_tools,
            user_id=binding.user_id,
            device_id=binding.device_id,
            include_hand_off=include_hand_off,
        )

    def _assemble_turn_messages(
        self,
        system_prompt: str,
        conversation_history: list[Any],
        messages: list[BaseMessage],
    ) -> tuple[list[BaseMessage], list[BaseMessage]]:
        """System + history + current turn, and the converted history on its own."""
        langchain_messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]

        # Convert and prepend conversation history (from database)
        history_messages_lc: list[BaseMessage] = []
        if conversation_history:
            history_messages_lc = self._convert_history_to_langchain_messages(
                conversation_history
            )
            langchain_messages.extend(history_messages_lc)

        langchain_messages.extend(messages)
        return langchain_messages, history_messages_lc

    async def _preflight_turn(self, turn: _ModelTurn, llm: Any) -> None:
        """Budget the turn for ``turn.runtime_config`` and adopt any reduced envelope."""
        turn.budget_result = await self._preflight_model_request(
            turn.runtime_config,
            system_messages=[turn.langchain_messages[0]],
            history_messages=turn.history_messages,
            current_messages=turn.current_messages,
            tools=turn.bound_tools,
            durable_request=turn.durable_request,
            emergency_compact=turn.emergency_compact,
            conversation_id=turn.binding.conversation_id,
            user_id=turn.binding.user_id,
            token_counter=self._token_counter_for_model(turn.runtime_config.provider, llm),
        )
        if turn.budget_result is not None:
            turn.history_messages = list(turn.budget_result.envelope.history_messages)
            turn.langchain_messages = list(turn.budget_result.envelope.messages)

    async def _invoke_turn_under_usage_operation(
        self,
        turn: _ModelTurn,
        llm_with_tools: Any,
        *,
        run_config: RunnableConfig | None,
        reasoning_summary_requested: bool,
    ) -> Any:
        """Call the provider for the turn, every retry and fallback inside one usage operation.

        A later tool-loop invocation (a new call to ``invoke_model_with_history``)
        starts a fresh operation. Attribution flows through the bound
        UsageContext; here we only add the acting agent id so per-agent rollups
        are correct.
        """
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
            run_config, usage_operation, turn.binding.user_id
        )

        async def call_model(model: Any, model_messages: list[BaseMessage]) -> Any:
            # ``turn.runtime_config`` is read at call time so each fallback
            # branch records under its own provider/model.
            if usage_run_config is not None:
                return await self._ainvoke_with_retries(
                    model,
                    model_messages,
                    run_config=usage_run_config,
                    operation=usage_operation,
                    provider=turn.runtime_config.provider,
                    model=turn.runtime_config.model,
                )
            return await self._ainvoke_with_retries(
                model,
                model_messages,
                operation=usage_operation,
                provider=turn.runtime_config.provider,
                model=turn.runtime_config.model,
            )

        try:
            return await self._invoke_turn_with_fallbacks(
                turn,
                llm_with_tools,
                call_model,
                reasoning_summary_requested=reasoning_summary_requested,
            )
        finally:
            if usage_operation_cm is not None:
                usage_operation_cm.__exit__(None, None, None)
            if usage_context_cm is not None:
                usage_context_cm.__exit__(None, None, None)

    async def _invoke_turn_with_fallbacks(
        self,
        turn: _ModelTurn,
        llm_with_tools: Any,
        call_model: _CallModel,
        *,
        reasoning_summary_requested: bool,
    ) -> Any:
        """The first attempt, then the OpenAI no-summary retry and the provider fallback."""
        try:
            return await self._invoke_turn_with_overflow_retry(turn, llm_with_tools, call_model)
        except ContextBudgetExceededError:
            raise
        except Exception:
            runtime_config = turn.runtime_config
            if (
                runtime_config.provider == "openai"
                and reasoning_summary_requested
                and runtime_config.api_key
            ):
                self._disable_openai_reasoning_summary(turn.binding.user_id)
                try:
                    return await self._invoke_turn_without_reasoning_summary(turn, call_model)
                except Exception:
                    fallback_runtime = self._provider_error_fallback(turn.runtime_config)
                    if fallback_runtime is None:
                        raise
                    return await self._invoke_turn_on_fallback_provider(
                        turn,
                        fallback_runtime,
                        call_model,
                        include_hand_off=turn.binding.include_hand_off,
                    )
            fallback_runtime = self._provider_error_fallback(turn.runtime_config)
            if fallback_runtime is None:
                raise
            return await self._invoke_turn_on_fallback_provider(
                turn, fallback_runtime, call_model, include_hand_off=None
            )

    async def _invoke_turn_with_overflow_retry(
        self,
        turn: _ModelTurn,
        llm_with_tools: Any,
        call_model: _CallModel,
    ) -> Any:
        """One attempt, plus the single structure-aware retry on a context overflow."""
        try:
            return await call_model(llm_with_tools, turn.langchain_messages)
        except Exception as exc:
            if not settings.context_overflow_retry_enabled or not is_context_overflow_error(exc):
                raise
            compacted_messages = prepare_aggressive_context_retry(
                turn.langchain_messages,
                tool_preview_chars=(settings.context_overflow_retry_tool_preview_chars),
            )
            try:
                response = await call_model(llm_with_tools, compacted_messages)
            except Exception as retry_exc:
                if is_context_overflow_error(retry_exc):
                    conversation_compaction_metrics.record_provider_overflow_retry("failure")
                    raise ContextBudgetExceededError("provider_context_overflow") from retry_exc
                raise
            conversation_compaction_metrics.record_provider_overflow_retry("success")
            turn.context_overflow_retried = True
            return response

    @staticmethod
    def _disable_openai_reasoning_summary(user_id: str | None) -> None:
        user_key = str(user_id).strip() if user_id else ""
        if user_key:
            _OPENAI_REASONING_SUMMARY_DISABLED_USERS.add(user_key)

    async def _invoke_turn_without_reasoning_summary(
        self, turn: _ModelTurn, call_model: _CallModel
    ) -> Any:
        llm, _ = self._create_langchain_model_from_runtime(
            turn.runtime_config,
            user_id=turn.binding.user_id,
            enable_reasoning_summary=False,
        )
        llm_with_tools = self._rebind_turn_tools(
            llm, turn.binding, include_hand_off=turn.binding.include_hand_off
        )
        return await call_model(llm_with_tools, turn.langchain_messages)

    def _provider_error_fallback(
        self, runtime_config: ResolvedRuntimeModelConfig
    ) -> ResolvedRuntimeModelConfig | None:
        """The configured fallback for a provider error, if it can be called."""
        fallback_runtime = self._create_fallback_runtime_config(
            runtime_config.fallback_config,
            reason="provider_error",
            from_provider=runtime_config.provider,
            inherited_warnings=runtime_config.warnings,
        )
        if not fallback_runtime or not fallback_runtime.api_key:
            return None
        return fallback_runtime

    async def _invoke_turn_on_fallback_provider(
        self,
        turn: _ModelTurn,
        fallback_runtime: ResolvedRuntimeModelConfig,
        call_model: _CallModel,
        *,
        include_hand_off: bool | None,
    ) -> Any:
        turn.runtime_config = fallback_runtime
        # Build the fallback model before the preflight so the budget is
        # counted with the provider about to be called, not with the
        # local upper bound the failed provider left behind.
        llm, _ = self._create_langchain_model_from_runtime(
            turn.runtime_config,
            user_id=turn.binding.user_id,
            enable_reasoning_summary=False,
        )
        await self._preflight_turn(turn, llm)
        llm_with_tools = self._rebind_turn_tools(
            llm, turn.binding, include_hand_off=include_hand_off
        )
        return await call_model(llm_with_tools, turn.langchain_messages)

    def _build_turn_response(
        self,
        turn: _ModelTurn,
        response: Any,
        token_breakdown: Any,
        *,
        has_tool_context: bool,
    ) -> AgentResponse:
        """The AgentResponse for the provider's reply, with its response metadata."""
        runtime_config = turn.runtime_config
        actual_usage = extract_actual_usage(response)
        if any(value is not None for value in actual_usage.values()):
            self._record_actual_usage(
                turn, actual_usage, token_breakdown, has_tool_context=has_tool_context
            )

        tool_calls = None
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_calls = response.tool_calls

        thinking = self._response_thinking(response, runtime_config.provider)
        response_text = coerce_response_text(response.content)
        reasoning_summary, reasoning_tokens = self._response_reasoning(
            response, runtime_config.provider, actual_usage
        )
        output_estimate, usage_source_override = self._estimate_missing_output_tokens(
            actual_usage, response_text, runtime_config
        )

        metadata = self._turn_usage_metadata(
            turn,
            token_breakdown,
            output_estimate=output_estimate,
            usage_source=usage_source_override,
        )
        self._add_response_content_metadata(
            metadata,
            response,
            thinking=thinking,
            reasoning_summary=reasoning_summary,
            reasoning_tokens=reasoning_tokens,
        )
        self._add_finish_reason(metadata, response, response_text, tool_calls)

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

    def _record_actual_usage(
        self,
        turn: _ModelTurn,
        actual_usage: dict[str, Any],
        token_breakdown: Any,
        *,
        has_tool_context: bool,
    ) -> None:
        """Copy the provider-reported usage onto the breakdown and calibrate the budget."""
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
        budget_result = turn.budget_result
        if budget_result is not None and actual_usage["input_tokens"] is not None:
            conversation_compaction_metrics.record_token_calibration(
                provider=turn.runtime_config.provider,
                model=turn.runtime_config.model,
                content_class=(
                    "mixed"
                    if turn.bound_tools and has_tool_context
                    else "tools"
                    if turn.bound_tools
                    else "text"
                ),
                estimated_tokens=budget_result.input_tokens,
                actual_tokens=actual_usage["input_tokens"],
            )

    @staticmethod
    def _response_thinking(response: Any, provider: str) -> Any:
        thinking = None
        if hasattr(response, "thinking") and response.thinking:
            thinking = response.thinking
        if not thinking:
            # Gemini thought blocks are normalized to standard ``reasoning``
            # blocks so they survive the streaming pipeline, so accept both
            # shapes here. OpenAI ``reasoning`` blocks are excluded: they are
            # harvested into ``reasoning_summary`` and must not also
            # land in ``thinking_summary``.
            thinking_block_types = {"thinking"}
            if provider != "openai":
                thinking_block_types.add("reasoning")
            thinking = extract_public_thinking_summary(
                response.content, block_types=thinking_block_types
            )
        return thinking

    @staticmethod
    def _response_reasoning(
        response: Any, provider: str, actual_usage: dict[str, Any]
    ) -> tuple[Any, Any]:
        """The OpenAI reasoning summary and the reasoning-token count to report."""
        reasoning_summary = None
        reasoning_tokens = actual_usage.get("reasoning_tokens")
        if provider == "openai":
            reasoning_summary = extract_openai_reasoning_summary(response.content)
            extracted_reasoning_tokens = extract_openai_reasoning_tokens(response)
            if extracted_reasoning_tokens is not None:
                reasoning_tokens = extracted_reasoning_tokens
        return reasoning_summary, reasoning_tokens

    @staticmethod
    def _estimate_missing_output_tokens(
        actual_usage: dict[str, Any],
        response_text: str,
        runtime_config: ResolvedRuntimeModelConfig,
    ) -> tuple[int | None, str | None]:
        """An output estimate when the provider reported usage but no output count.

        Lets the gauge show an output figure. Never overwrites a reported
        output; the context source is relabelled ``mixed_reported_estimated``
        downstream.
        """
        if not (
            any(value is not None for value in actual_usage.values())
            and actual_usage.get("output_tokens") is None
        ):
            return None, None
        output_estimate = estimate_output_tokens(
            response_text,
            provider=runtime_config.provider,
            model=runtime_config.model,
        )
        if output_estimate is None:
            return None, None
        return output_estimate, "mixed_reported_estimated"

    def _turn_usage_metadata(
        self,
        turn: _ModelTurn,
        token_breakdown: Any,
        *,
        output_estimate: int | None,
        usage_source: str | None,
    ) -> dict[str, Any]:
        token_breakdown_dict = token_breakdown.to_dict()
        metadata: dict[str, Any] = {
            "conversation_id": turn.binding.conversation_id,
            "token_breakdown": token_breakdown_dict,
        }
        if turn.context_overflow_retried:
            metadata["context_overflow_retry"] = True
        request_budget_metadata = self._request_budget_metadata(turn.budget_result)
        if request_budget_metadata is not None:
            metadata["request_budget"] = request_budget_metadata
        self._apply_runtime_metadata(metadata, turn.runtime_config)
        self._merge_context_window_usage(
            metadata,
            token_breakdown_dict,
            output_estimate=output_estimate,
            usage_source=usage_source,
        )
        return metadata

    def _add_response_content_metadata(
        self,
        metadata: dict[str, Any],
        response: Any,
        *,
        thinking: Any,
        reasoning_summary: Any,
        reasoning_tokens: Any,
    ) -> None:
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

    def _add_finish_reason(
        self,
        metadata: dict[str, Any],
        response: Any,
        response_text: str,
        tool_calls: Any,
    ) -> None:
        # The provider's own account of why it stopped. Nothing in this
        # codebase read it, so an empty candidate -- MAX_TOKENS spent on
        # thinking, SAFETY, RECITATION, MALFORMED_FUNCTION_CALL -- arrived
        # downstream as an ordinary response with no text and no reason,
        # and every consumer had to guess. A turn that produced neither
        # text nor a tool call is worth a line on its own: it is the shape
        # that fails `empty_public_content` two nodes later.
        finish_reason = _finish_reason(response)
        if finish_reason:
            metadata["finish_reason"] = finish_reason
        if not response_text.strip() and not tool_calls:
            logger.warning(
                "%s produced no text and no tool calls (finish_reason=%s, "
                "content=%.200r); the turn has nothing to publish",
                self.agent_id,
                finish_reason or "unreported",
                getattr(response, "content", None),
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
        system_prompt = f"{self._get_base_system_prompt()}{MARKDOWN_CURRENCY_GUIDANCE}"

        user_id = _.get("user_id")
        device_id = _.get("device_id")

        # Append active client-side skills
        skills_suffix = self._build_skills_suffix(user_id=user_id, device_id=device_id)
        if skills_suffix:
            system_prompt = f"{system_prompt}{skills_suffix}"

        system_prompt = f"{system_prompt}{build_runtime_time_context_block()}"

        # Append shared tool-usage guidance.
        system_prompt = f"{system_prompt}{TOOL_EXPLORATION_SUFFIX}"

        # Only describe memory when the tools are actually bound; otherwise the
        # model is told about capabilities it cannot reach.
        if getattr(settings, "enable_user_memory_tools", False) and user_id:
            system_prompt = f"{system_prompt}{USER_MEMORY_SUFFIX}"

        if getattr(settings, "enable_conversation_search_tools", False) and user_id:
            system_prompt = f"{system_prompt}{CONVERSATION_SEARCH_SUFFIX}"

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
        base = f"{self._get_base_system_prompt()}{MARKDOWN_CURRENCY_GUIDANCE}"
        suffix = self._build_skills_suffix(user_id=user_id, device_id=device_id)
        if suffix:
            base = f"{base}{suffix}"
        return f"{base}{build_runtime_time_context_block()}"

    def _build_error_response(
        self,
        message: str,
        conversation_id: str | None,
        error: str | BaseException | None = None,
    ) -> AgentResponse:
        """Build the apology response for a failed turn.

        ``error`` is either a stable code the caller chose, or the exception
        itself. An exception is stored as :func:`agent_error_code` -- the error
        field and its metadata copy are persisted and streamed, and provider
        text can carry URLs, request ids and key fragments -- while its detail
        goes to the log.
        """
        if isinstance(error, BaseException):
            logger.warning(
                "%s returned an error response: %s: %s",
                self.agent_id,
                type(error).__name__,
                error,
            )
            error = agent_error_code(error)
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
