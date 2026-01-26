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

        self._init_gemini()

    def _init_gemini(self) -> None:
        try:
            self.gemini_client = create_gemini_client()
            self.langchain_model = create_langchain_model(
                agent_type=self.agent_config_key,
                model_override=self.model_name,
            )
            logger.info(
                f"Initialized Gemini client and LangChain model with model: {self.model_name}"
            )
        except Exception as e:
            logger.error(f"Error initializing Gemini client: {e}")
            raise

    async def _init_tools(self) -> None:
        if self.mcp_manager is not None:
            return

        try:
            self.mcp_manager = await get_global_mcp_manager()

            all_tools = await self.mcp_manager.get_tools()

            self.tools = self._deduplicate_tools(all_tools)

            server_status = self.mcp_manager.get_servers_status()
            active_servers = [
                name for name, status in server_status.items() if status.get("enabled")
            ]
            logger.info(
                f"Initialized {len(self.tools)} unique tools from "
                f"{len(active_servers)} active MCP servers: {active_servers}"
            )

        except Exception as e:
            logger.error(f"Error initializing MCP tools: {e}")
            self.tools = []

    def _deduplicate_tools(self, tools: List[BaseTool]) -> List[BaseTool]:
        unique_tools: Dict[str, BaseTool] = {}
        for tool in tools:
            unique_tools.setdefault(tool.name, tool)
        return list(unique_tools.values())

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

    def _get_llm_with_tools(self, model: Any = None) -> Any:
        llm = model or self.langchain_model
        if not self.tools or llm is None:
            return llm

        tool_choice = getattr(settings, "tool_choice_mode", "auto")

        from ..model_factory import ModelFactory

        return ModelFactory.bind_tools_to_model(
            llm,
            self.tools,
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

    def _is_openai_reasoning_summary_unsupported(self, exc: Exception) -> bool:
        try:
            text = str(exc)
        except Exception:
            text = repr(exc)

        text_lower = text.lower()
        if "reasoning.summary" not in text_lower:
            return False

        if "unsupported_value" in text_lower:
            return True

        if (
            "organization must be verified" in text_lower
            or "verify organization" in text_lower
        ):
            return True

        return False

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
                if self._is_openai_reasoning_summary_unsupported(exc):
                    raise
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
            if self.mcp_manager is None:
                await self._init_tools()

            resolved_request = self._resolve_model_request(model_request)

            provider = "gemini"
            effective_model_name = self.model_name
            effective_temperature = AGENT_CONFIG.get(self.agent_config_key, {}).get(
                "temperature", 1.0
            )
            used_fallback = False
            fallback_reason: Optional[str] = None
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
                        fallback_reason = "OpenAI provider selected but no API key is configured for this user"
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

            llm_with_tools = self._get_llm_with_tools(llm)

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
            if conversation_history:
                history_messages = self._convert_history_to_langchain_messages(
                    conversation_history
                )
                langchain_messages.extend(history_messages)

            # Add current turn messages
            langchain_messages.extend(messages)

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
                        and self._is_openai_reasoning_summary_unsupported(exc)
                    ):
                        user_key = str(user_id).strip() if user_id else ""
                        if user_key:
                            _OPENAI_REASONING_SUMMARY_DISABLED_USERS.add(user_key)

                        logger.warning(
                            "%s: OpenAI reasoning summaries unavailable; retrying without them: %s",
                            self.agent_id,
                            exc,
                        )

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
                            llm_with_tools = self._get_llm_with_tools(llm)
                            openai_reasoning_summary_requested = False
                            response = await self._ainvoke_with_retries(
                                llm_with_tools, langchain_messages
                            )
                        except Exception as exc2:
                            # Retry exhausted or provider error: fall back to Gemini defaults.
                            used_fallback = True
                            fallback_reason = f"{type(exc2).__name__}"
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
                            llm_with_tools = self._get_llm_with_tools(llm)
                            response = await llm_with_tools.ainvoke(langchain_messages)
                    else:
                        # Retry exhausted or provider error: fall back to Gemini defaults.
                        used_fallback = True
                        fallback_reason = f"{type(exc).__name__}"
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
                        llm_with_tools = self._get_llm_with_tools(llm)
                        response = await llm_with_tools.ainvoke(langchain_messages)
            else:
                response = await llm_with_tools.ainvoke(langchain_messages)

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
                "tool_count": len(self.tools),
            }

            if used_fallback:
                metadata["provider_fallback"] = {
                    "from": "openai",
                    "to": "gemini",
                    "reason": fallback_reason or "fallback",
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
