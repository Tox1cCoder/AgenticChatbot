import asyncio
import logging
from typing import Optional, List, Dict, Any, Set
from abc import ABC, abstractmethod

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    ToolMessage,
    HumanMessage,
    AIMessage,
)
from langchain_core.tools import BaseTool

from ..schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from ..prompts import TOOL_CONTEXT_SUFFIX
from ..utils import (
    coerce_response_text,
    extract_openai_reasoning_summary,
    extract_openai_reasoning_tokens,
)
from ..agent_config import create_langchain_model, create_gemini_client, AGENT_CONFIG
from ...core.config import settings
from ..mcp_integration import get_global_mcp_manager
from ..mcp_registry import get_mcp_tools_generation
from ..token_instrumentation import (
    compute_token_breakdown,
    extract_actual_usage,
    log_token_breakdown,
    TokenBudgetExceededWarning,
)
from ..deferred_tool_binding import (
    should_use_deferred_loading,
    build_deferred_tool_list
)

logger = logging.getLogger(__name__)

_MODEL_REQUEST_SUPPORTED_AGENT_KEYS = {"chat", "rag", "search", "planning"}
_OPENAI_REASONING_SUMMARY_DISABLED_USERS: Set[str] = set()


class BaseAgent(ABC):
    """Abstract base class for all agents. Child classes must implement: agent_type, agent_id, _get_base_system_prompt()."""

    def __init__(
        self, model_name: Optional[str] = None, agent_config_key: str = "chat"
    ):
        self.agent_config_key = agent_config_key
        self.model_name = model_name or AGENT_CONFIG[agent_config_key]["model"]
        self.gemini_client = None
        self.langchain_model = None
        self.mcp_manager = None
        self.tools: List[BaseTool] = []

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
            logger.error(f"Error initializing Gemini client: {e}")
            raise

    async def _init_tools(self) -> None:
        """
        Initialize or refresh tools from MCP manager.

        This method now checks the tools_generation version from the registry
        to detect when tools need to be refreshed (e.g., after a server is
        disabled). This ensures disabled server tools are never bound to the model.

        Tools are also filtered based on per-agent allowlists configured in settings.
        """
        current_generation = get_mcp_tools_generation()

        # Check if we need to refresh tools
        needs_refresh = (
            self.mcp_manager is None
            or self._tools_generation_seen != current_generation
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

            server_status = self.mcp_manager.get_servers_status()
            active_servers = [
                name for name, status in server_status.items() if status.get("enabled")
            ]

            # Log tool refresh with generation info
            if self._tools_generation_seen > 0:
                logger.debug(
                    f"Refreshed tools for {self.agent_id} (generation={current_generation}): "
                    f"{len(self.tools)} tools (from {len(unique_tools)} available) "
                    f"from {len(active_servers)} active servers"
                )
            else:
                logger.debug(
                    f"Initialized {len(self.tools)} tools (from {len(unique_tools)} available) "
                    f"for {self.agent_id} from {len(active_servers)} MCP servers"
                )

        except Exception as e:
            logger.error(f"Error initializing MCP tools: {e}")
            self.tools = []

    def _deduplicate_tools(self, tools: List[BaseTool]) -> List[BaseTool]:
        unique_tools: Dict[str, BaseTool] = {}
        for tool in tools:
            unique_tools.setdefault(tool.name, tool)
        return list(unique_tools.values())

    def _filter_tools_by_allowlist(self, tools: List[BaseTool]) -> List[BaseTool]:
        """
        Filter tools based on per-agent allowlist configuration.

        Allowlist can contain:
        - Tool names (e.g., "tavily_search")
        - Server names (e.g., "tavily") - matches all tools from that server

        If allowlist is empty, all tools are allowed.
        """
        # Get agent-specific allowlist from settings
        allowlist_key = f"{self.agent_config_key}_agent_allowed_tools"
        allowlist = getattr(settings, allowlist_key, []) or []

        # Empty allowlist means all tools allowed
        if not allowlist:
            return tools

        # Build set of allowed names for fast lookup
        allowed_set = set(allowlist)

        filtered_tools = []
        for tool in tools:
            tool_name = getattr(tool, "name", "")

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

    def _get_allowlist(self) -> Optional[List[str]]:
        """Get the per-agent tool allowlist from settings."""
        allowlist_key = f"{self.agent_config_key}_agent_allowed_tools"
        return getattr(settings, allowlist_key, []) or []

    def _get_tools_for_binding(
        self,
        conversation_id: Optional[str] = None,
        internal_tools: Optional[List[BaseTool]] = None,
    ) -> List[BaseTool]:
        """
        Get the tools to bind to the model for this invocation.

        When mcp_tool_search_enabled is True, returns a reduced set:
        - Internal tools (if provided)
        - tool_search tool
        - Pinned MCP tools
        - Loaded deferred tools for this conversation

        When mcp_tool_search_enabled is False, returns all tools (current behavior).

        Args:
            conversation_id: Current conversation ID for deferred tool lookup
            internal_tools: Non-MCP internal tools to always include

        Returns:
            List of tools to bind to the model
        """
        use_deferred = should_use_deferred_loading(self.agent_config_key)

        if use_deferred:
            # Build deferred tool list
            tools = build_deferred_tool_list(
                conversation_id=conversation_id,
                agent_key=self.agent_config_key,
                mcp_manager=self.mcp_manager,
                all_mcp_tools=self.tools,  # self.tools contains filtered MCP tools
                internal_tools=internal_tools,
                allowlist=self._get_allowlist(),
            )

            return tools
        else:
            # Traditional mode: return all tools (with internal tools prepended)
            if internal_tools:
                # Combine internal tools with MCP tools, avoiding duplicates
                seen = {t.name for t in internal_tools}
                combined = list(internal_tools)
                for tool in self.tools:
                    if tool.name not in seen:
                        combined.append(tool)
                        seen.add(tool.name)
                return combined
            return self.tools

    def _resolve_model_request(
        self, model_request: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
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
        conversation_id: Optional[str] = None,
        internal_tools: Optional[List[BaseTool]] = None,
    ) -> Any:
        """
        Bind tools to the model for invocation.

        Args:
            model: Optional model override
            conversation_id: Conversation ID for deferred tool lookup
            internal_tools: Non-MCP internal tools to include

        Returns:
            Model with tools bound
        """
        llm = model or self.langchain_model

        # Get tools for binding (respects deferred loading setting)
        tools = self._get_tools_for_binding(
            conversation_id=conversation_id,
            internal_tools=internal_tools,
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

    def _get_openai_api_key(self, user_id: Optional[str]) -> Optional[str]:
        if not user_id:
            return None

        try:
            from uuid import UUID

            user_uuid = UUID(str(user_id))
        except Exception:
            return None

        try:
            from app.core.container import container

            provider_service = container.provider_service()
            return provider_service.get_decrypted_api_key(user_uuid, "openai")
        except Exception:
            return None

    async def _ainvoke_with_retries(
        self, llm_with_tools: Any, messages: List[BaseMessage]
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

        last_exc: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                return await llm_with_tools.ainvoke(messages)
            except Exception as exc:
                last_exc = exc
   
                if attempt >= attempts:
                    break
                sleep_for = delay * (2 ** (attempt - 1))
                logger.warning(
                    "%s: provider call failed (attempt %s/%s): %s",
                    self.agent_id,
                    attempt,
                    attempts,
                    exc,
                )
                if sleep_for:
                    await asyncio.sleep(sleep_for)

        raise last_exc or RuntimeError("Provider call failed")

    def _convert_history_to_langchain_messages(
        self, conversation_history: List[Any]
    ) -> List[BaseMessage]:
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

                if role == "user":
                    langchain_history.append(HumanMessage(content=content))
                elif role == "assistant":
                    langchain_history.append(AIMessage(content=content))
                # Skip system messages as we add our own system prompt
        return langchain_history

    async def invoke_model_with_history(
        self,
        messages: List[BaseMessage],
        conversation_history: List[Any],
        persona: Optional[str],
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        model_request: Optional[Dict[str, Any]] = None,
        **system_prompt_kwargs: Any,
    ) -> AgentResponse:
        try:
            # Always check for tool refresh (handles disabled servers, etc.)
            # _init_tools now checks generation version and only refreshes if needed
            await self._init_tools()

            resolved_request = self._resolve_model_request(model_request)

            provider = "gemini"
            effective_model_name = self.model_name
            effective_temperature = AGENT_CONFIG.get(self.agent_config_key, {}).get(
                "temperature", 1.0
            )
            used_fallback = False
            openai_api_key: Optional[str] = None
            openai_reasoning_summary_requested = False

            llm = self.langchain_model
            if resolved_request and isinstance(resolved_request, dict):
                requested_provider = (
                    str(resolved_request.get("provider") or "").strip().lower()
                )
                requested_model = resolved_request.get("model")
                requested_temp = resolved_request.get("temperature")

                if requested_provider in {"openai", "gemini"}:
                    provider = requested_provider

                if isinstance(requested_model, str) and requested_model.strip():
                    effective_model_name = requested_model.strip()
                if isinstance(requested_temp, (int, float)):
                    effective_temperature = float(requested_temp)

                if provider == "openai":
                    api_key = self._get_openai_api_key(user_id)
                    if api_key:
                        from ..model_factory import ModelFactory

                        openai_api_key = api_key
                        include_reasoning_summary = True
                        user_key = str(user_id).strip() if user_id else ""
                        if (
                            user_key
                            and user_key in _OPENAI_REASONING_SUMMARY_DISABLED_USERS
                        ):
                            include_reasoning_summary = False

                        openai_kwargs: Dict[str, Any] = {
                            "provider": "openai",
                            "model": effective_model_name,
                            "api_key": api_key,
                            "temperature": effective_temperature,
                            "timeout": settings.openai_request_timeout_seconds,
                            "streaming": True,
                        }

                        # Configure reasoning based on model type
                        model_lower = effective_model_name.lower()
                        if include_reasoning_summary:
                            # For o1/o3 models, use effort-based reasoning
                            if "o1" in model_lower or "o3" in model_lower:
                                # o1/o3 models support extended thinking with effort levels
                                openai_kwargs["reasoning"] = {"effort": "medium"}
                            else:
                                # For other models, request reasoning summary
                                openai_kwargs["reasoning"] = {"summary": "auto"}

                        llm = ModelFactory.create_model(**openai_kwargs)
                        openai_reasoning_summary_requested = include_reasoning_summary
                    else:
                        used_fallback = True
                        provider = "gemini"
                        effective_model_name = self.model_name
                        effective_temperature = AGENT_CONFIG.get(
                            self.agent_config_key, {}
                        ).get("temperature", 1.0)
                        llm = create_langchain_model(
                            agent_type=self.agent_config_key,
                            model_override=effective_model_name,
                            temperature_override=effective_temperature,
                        )

                elif provider == "gemini":
                    llm = create_langchain_model(
                        agent_type=self.agent_config_key,
                        model_override=effective_model_name,
                        temperature_override=effective_temperature,
                    )

            llm_with_tools = self._get_llm_with_tools(
                llm, conversation_id=conversation_id
            )

            # Get the tools that are actually bound (for accurate token counting)
            bound_tools = self._get_tools_for_binding(conversation_id=conversation_id)

            has_tool_context = any(
                isinstance(msg, ToolMessage)
                or (hasattr(msg, "tool_calls") and msg.tool_calls)
                or (
                    hasattr(msg, "additional_kwargs")
                    and msg.additional_kwargs.get("tool_calls")
                )
                for msg in messages
            )

            system_prompt = self._build_system_prompt(
                persona, has_tool_context, **system_prompt_kwargs
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

            # === Token Instrumentation ===
            # Compute and log token breakdown for observability
            # Use bound_tools (not self.tools) to reflect actual schema tokens sent
            token_breakdown = compute_token_breakdown(
                system_prompt=system_prompt,
                history_messages=history_messages_lc,
                current_turn_messages=messages,
                tools=bound_tools if bound_tools else None,
            )
            log_token_breakdown(
                breakdown=token_breakdown,
                agent_id=self.agent_id,
                conversation_id=conversation_id,
                level=logging.DEBUG,
            )
            # Check for budget warnings
            TokenBudgetExceededWarning.check_and_warn(
                breakdown=token_breakdown,
                agent_id=self.agent_id,
            )

            # This ensures it runs ONCE per request, not on every agent iteration
            # (prevents context bloat during ReAct loops)
            if provider == "openai" and not used_fallback:
                try:
                    response = await self._ainvoke_with_retries(
                        llm_with_tools, langchain_messages
                    )
                except Exception as exc:
                    if (
                        openai_api_key
                        and openai_reasoning_summary_requested
                    ):
                        user_key = str(user_id).strip() if user_id else ""
                        if user_key:
                            _OPENAI_REASONING_SUMMARY_DISABLED_USERS.add(user_key)


                        try:
                            from ..model_factory import ModelFactory

                            llm = ModelFactory.create_model(
                                provider="openai",
                                model=effective_model_name,
                                api_key=openai_api_key,
                                temperature=effective_temperature,
                                timeout=settings.openai_request_timeout_seconds,
                                streaming=True,
                            )
                            llm_with_tools = self._get_llm_with_tools(
                                llm, conversation_id=conversation_id
                            )
                            openai_reasoning_summary_requested = False
                            response = await self._ainvoke_with_retries(
                                llm_with_tools, langchain_messages
                            )
                        except Exception as exc2:
                            # Retry exhausted or provider error: fall back to Gemini defaults.
                            used_fallback = True
                            provider = "gemini"
                            effective_model_name = self.model_name
                            effective_temperature = AGENT_CONFIG.get(
                                self.agent_config_key, {}
                            ).get("temperature", 1.0)
                            llm = create_langchain_model(
                                agent_type=self.agent_config_key,
                                model_override=effective_model_name,
                                temperature_override=effective_temperature,
                            )
                            llm_with_tools = self._get_llm_with_tools(
                                llm, conversation_id=conversation_id
                            )
                            response = await llm_with_tools.ainvoke(langchain_messages)
                    else:
                        # Retry exhausted or provider error: fall back to Gemini defaults.
                        used_fallback = True
                        provider = "gemini"
                        effective_model_name = self.model_name
                        effective_temperature = AGENT_CONFIG.get(
                            self.agent_config_key, {}
                        ).get("temperature", 1.0)
                        llm = create_langchain_model(
                            agent_type=self.agent_config_key,
                            model_override=effective_model_name,
                            temperature_override=effective_temperature,
                        )
                        llm_with_tools = self._get_llm_with_tools(
                            llm, conversation_id=conversation_id
                        )
                        response = await llm_with_tools.ainvoke(langchain_messages)
            else:
                response = await llm_with_tools.ainvoke(langchain_messages)

            # === Extract and log actual token usage ===
            actual_usage = extract_actual_usage(response)
            if actual_usage.get("input_tokens") is not None:
                token_breakdown.actual_input_tokens = actual_usage["input_tokens"]
                token_breakdown.actual_output_tokens = actual_usage.get("output_tokens")
                logger.debug(
                    "%s: Actual token usage - input=%s, output=%s (estimated=%d)",
                    self.agent_id,
                    actual_usage["input_tokens"],
                    actual_usage.get("output_tokens"),
                    token_breakdown.total_tokens,
                )

            tool_calls = None
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_calls = response.tool_calls

            thinking = None
            if hasattr(response, "thinking") and response.thinking:
                thinking = response.thinking

            response_text = coerce_response_text(response.content)

            reasoning_summary = None
            reasoning_tokens = None
            if provider == "openai" and not used_fallback:
                reasoning_summary = extract_openai_reasoning_summary(response.content)
                reasoning_tokens = extract_openai_reasoning_tokens(response)

            metadata = {
                "model": effective_model_name,
                "provider": provider,
                "conversation_id": conversation_id,
                "has_tool_calls": tool_calls is not None,
                "token_breakdown": token_breakdown.to_dict(),
            }

            if thinking:
                metadata["thinking"] = thinking

            if isinstance(reasoning_summary, str) and reasoning_summary.strip():
                metadata["reasoning_summary"] = reasoning_summary.strip()
                if isinstance(reasoning_tokens, int) and reasoning_tokens >= 0:
                    metadata["reasoning_tokens"] = reasoning_tokens

            agent_message = AgentMessage(
                role=MessageRole.ASSISTANT, content=response_text, tool_calls=tool_calls
            )

            return AgentResponse(
                agent_type=self.agent_type,
                agent_id=self.agent_id,
                message=agent_message,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Error invoking model with history: {e}")
            return self._build_error_response(
                message="I encountered an error processing your request.",
                conversation_id=conversation_id,
                error=str(e),
            )

    def _build_system_prompt(
        self, persona: Optional[str], has_tool_context: bool, **_: Any
    ) -> str:
        system_prompt = self._get_base_system_prompt()

        if has_tool_context:
            system_prompt = f"{system_prompt}\n\n{TOOL_CONTEXT_SUFFIX}"

        if persona:
            system_prompt = (
                f"Custom Persona:\n{persona.strip()}\n\n---\n{system_prompt}"
            )

        return system_prompt

    @abstractmethod
    def _get_base_system_prompt(self) -> str:
        pass

    def _build_error_response(
        self, message: str, conversation_id: Optional[str], error: Optional[str] = None
    ) -> AgentResponse:
        return AgentResponse(
            agent_type=self.agent_type,
            agent_id=self.agent_id,
            message=AgentMessage(
                role=MessageRole.ASSISTANT, content=f"I'm sorry, but {message}"
            ),
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
        logger.debug(
            f"{self.__class__.__name__} cleanup completed (MCP manager is shared)"
        )

    @property
    @abstractmethod
    def agent_type(self) -> AgentType:
        pass

    @property
    @abstractmethod
    def agent_id(self) -> str:
        pass
