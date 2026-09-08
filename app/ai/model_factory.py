"""
Model Factory for creating provider-specific AI model instances.

This factory supports dynamic model creation for:
- Gemini (via langchain-google-genai)
- OpenAI (via langchain-openai)

Provides a unified interface for creating chat models with provider-agnostic
tool binding and configuration.
"""

import logging
from functools import lru_cache
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from ..core.config import settings
from ..core.runtime_modeling import ResolvedRuntimeModelConfig, StrictRuntimeResolutionError

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _bounded_tool_openai_class(base: type) -> type:
    """A ``ChatOpenAI`` subclass that cannot send a description OpenAI refuses.

    OpenAI validates the whole `tools` array before generating anything and
    rejects the entire request with 400 `string_above_max_length` when any
    `function.description` exceeds 1024 characters. Gemini has no such limit,
    so descriptions tuned for Gemini broke every OpenAI call that offered
    tools — and the configured fallback answered on Gemini instead, so the only
    symptom was a warning about "a provider error".

    Clamped here rather than at the tool definitions because the limit belongs
    to this provider: the same tool objects are bound for Gemini, which accepts
    them whole. See ``app/ai/openai_tool_limits.py``.

    Built as a cached factory over ``base`` rather than a module-level ``class``
    statement so the base is resolved when a model is created, not when this
    module is imported — otherwise substituting the ``ChatOpenAI`` name (which
    the retry-configuration tests do) would have no effect on what gets built.
    """

    class _BoundedToolChatOpenAI(base):  # type: ignore[misc,valid-type]
        def bind_tools(self, tools, **kwargs):  # type: ignore[override]
            from .openai_tool_limits import clamp_openai_tool_descriptions

            return super().bind_tools(clamp_openai_tool_descriptions(list(tools)), **kwargs)

    _BoundedToolChatOpenAI.__name__ = f"BoundedTool{base.__name__}"
    _BoundedToolChatOpenAI.__qualname__ = _BoundedToolChatOpenAI.__name__
    return _BoundedToolChatOpenAI


def _build_openai_chat_model(**model_kwargs: Any) -> Any:
    """Construct the OpenAI chat model, with the tool clamp applied.

    ``ChatOpenAI`` is read from the module namespace at call time so it stays a
    substitutable seam. A substitute that is not a class is called directly —
    there is nothing to subclass, and a caller replacing the constructor has
    already opted out of the clamp.
    """
    base = ChatOpenAI
    if not isinstance(base, type):
        return base(**model_kwargs)
    return _bounded_tool_openai_class(base)(**model_kwargs)


class ModelFactory:
    """
    Factory for creating provider-specific model instances with unified interface.

    Supports Gemini and OpenAI providers with provider-agnostic tool binding.
    """

    @staticmethod
    def create_model(
        provider: str,
        model: str,
        api_key: str,
        temperature: float = 1.0,
        thinking_config: dict[str, Any] | None = None,
        timeout: int | None = None,
        **kwargs,
    ) -> BaseChatModel:
        """
        Create a chat model instance based on provider.

        Args:
            provider: Provider name ('gemini' or 'openai')
            model: Model name/identifier
            api_key: API key for the provider
            temperature: Sampling temperature (0.0 to 2.0)
            thinking_config: Optional thinking/reasoning configuration
            timeout: Request timeout in seconds (uses config default if not provided)
            **kwargs: Additional provider-specific kwargs

        Returns:
            BaseChatModel: Configured chat model instance

        Raises:
            ValueError: If provider is not supported
        """
        provider = provider.lower()

        if provider == "gemini":
            return ModelFactory._create_gemini_model(
                model, api_key, temperature, thinking_config, **kwargs
            )
        elif provider == "openai":
            return ModelFactory._create_openai_model(model, api_key, temperature, timeout, **kwargs)
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. Supported providers: gemini, openai"
            )

    @staticmethod
    def _create_gemini_model(
        model: str,
        api_key: str,
        temperature: float,
        thinking_config: dict[str, Any] | None = None,
        **kwargs,
    ) -> ChatGoogleGenerativeAI:
        """
        Create a Gemini model instance with thinking configuration.

        Args:
            model: Gemini model name (e.g., 'gemini-3-flash-preview')
            api_key: Gemini API key
            temperature: Sampling temperature
            thinking_config: Optional dict with:
                - enabled: bool (enable thinking mode)
                - budget: int (for Gemini 2.5, token budget for thinking)
                - level: str (for Gemini 3, 'low'|'medium'|'high')
                - include_thoughts: bool (include thinking in response)
            **kwargs: Additional Gemini-specific parameters

        Returns:
            ChatGoogleGenerativeAI: Configured Gemini model
        """
        model_kwargs = {
            "model": model,
            "google_api_key": api_key,
            "temperature": temperature,
            # The application owns retry boundaries (the usage recorder makes
            # each provider attempt exactly once); disable SDK-internal retries.
            "max_retries": 0,
        }

        # Apply Gemini-specific thinking configuration
        if thinking_config and thinking_config.get("enabled"):
            if "2.5" in model.lower():
                # Gemini 2.5 uses thinking_budget
                model_kwargs["thinking_budget"] = thinking_config.get("budget", 8192)
            elif "3" in model.lower():
                # Gemini 3 uses thinking_level
                model_kwargs["thinking_level"] = thinking_config.get("level", "high")

            if thinking_config.get("include_thoughts"):
                model_kwargs["include_thoughts"] = True

        # Merge any additional kwargs
        model_kwargs.update(kwargs)

        logger.info(f"Creating Gemini model: {model} with temperature={temperature}")
        return ChatGoogleGenerativeAI(**model_kwargs)

    @staticmethod
    def _create_openai_model(
        model: str,
        api_key: str,
        temperature: float,
        timeout: int | None = None,
        **kwargs,
    ) -> ChatOpenAI:
        """
        Create an OpenAI model instance with reasoning support.

        Args:
            model: OpenAI model name (e.g., 'gpt-4o', 'o1-preview', 'o3-mini')
            api_key: OpenAI API key
            temperature: Sampling temperature
            timeout: Request timeout in seconds (uses config default if not provided)
            **kwargs: Additional OpenAI-specific parameters including:
                - reasoning: Dict with {"effort": "low"|"medium"|"high"} for o1/o3
                          or {"summary": "auto"} for extended thinking summaries
                - streaming: bool - Enable streaming responses

        Returns:
            ChatOpenAI: Configured OpenAI model
        """
        model_kwargs = {
            "model": model,
            "openai_api_key": api_key,
            "temperature": temperature,
            "timeout": timeout or settings.openai_request_timeout_seconds,
            # Application owns retries; disable SDK-internal retries.
            "max_retries": 0,
        }

        # Merge any additional kwargs (including reasoning, streaming, etc.)
        model_kwargs.update(kwargs)

        logger.info(
            f"Creating OpenAI model: {model} with temperature={temperature}, "
            f"streaming={model_kwargs.get('streaming', False)}, "
            f"reasoning={model_kwargs.get('reasoning')}"
        )

        try:
            return _build_openai_chat_model(**model_kwargs)
        except TypeError as exc:
            # Older langchain-openai versions may not support reasoning parameter
            if "reasoning" in str(exc) and "unexpected keyword argument" in str(exc):
                logger.warning(
                    "langchain-openai version does not support 'reasoning' parameter. "
                    "Consider upgrading to get extended thinking support for o1/o3 models."
                )
                model_kwargs.pop("reasoning", None)
                return _build_openai_chat_model(**model_kwargs)
            raise

    @staticmethod
    def create_model_from_runtime(
        config: ResolvedRuntimeModelConfig,
        **kwargs: Any,
    ) -> BaseChatModel:
        """Build the configured chat model from a resolved runtime config.

        This performs no fallback of its own: a missing credential or an
        unsupported provider raises ``StrictRuntimeResolutionError`` so the
        caller can return a typed failure instead of silently substituting a
        different model.
        """
        api_key = (config.api_key or "").strip()
        if not api_key:
            raise StrictRuntimeResolutionError(
                "missing_credentials",
                f"no credential available for provider {config.provider}",
            )
        model_id = (config.model or "").strip()
        if not model_id:
            raise StrictRuntimeResolutionError(
                "missing_model", f"no model configured for agent {config.agent_key}"
            )

        try:
            return ModelFactory.create_model(
                provider=config.provider,
                model=model_id,
                api_key=api_key,
                temperature=config.temperature,
                **kwargs,
            )
        except ValueError as exc:
            raise StrictRuntimeResolutionError(
                "unsupported_provider", f"provider {config.provider} is not supported"
            ) from exc

    @staticmethod
    def bind_tools_to_model(
        model: BaseChatModel, tools: list[BaseTool], tool_choice: str = "auto"
    ) -> BaseChatModel:
        """
        Bind tools to a model instance in a provider-agnostic way.

        This method uses LangChain's standard bind_tools() interface which works
        across all providers (Gemini, OpenAI, Anthropic, etc.).

        Args:
            model: The chat model instance
            tools: List of LangChain tools to bind
            tool_choice: Tool calling mode:
                - "auto": Model decides whether to call tools
                - "any": Model must call at least one tool
                - "none": Model must not call tools
                - "<tool_name>": Model must call specific tool

        Returns:
            BaseChatModel: Model with tools bound

        Note:
            For Gemini, this internally translates to function_calling_config.
            For OpenAI, this uses the tool_choice parameter directly.
        """
        if not tools:
            logger.debug("No tools provided, returning model without tool binding")
            return model

        # Normalize tool_choice to uppercase for consistency
        tool_choice_normalized = (
            tool_choice.upper() if tool_choice.lower() in ["auto", "any", "none"] else tool_choice
        )

        logger.debug(
            f"Binding {len(tools)} tools to model with tool_choice={tool_choice_normalized}"
        )

        # LangChain's bind_tools handles provider-specific differences
        # For Gemini: converts to tool_config={"function_calling_config": {"mode": ...}}
        # For OpenAI: uses tool_choice parameter directly
        if isinstance(model, ChatGoogleGenerativeAI):
            if isinstance(tool_choice, str) and tool_choice.lower() in {
                "auto",
                "any",
                "none",
            }:
                # Gemini function_calling_config mode for standard choices.
                return model.bind_tools(
                    tools,
                    tool_config={"function_calling_config": {"mode": tool_choice_normalized}},
                )

            # For specific tool names/lists, use native tool_choice forwarding.
            return model.bind_tools(tools, tool_choice=tool_choice)

        # OpenAI and others use standard tool_choice parameter
        return model.bind_tools(tools, tool_choice=tool_choice)


# Convenience function for backward compatibility
def create_model(
    provider: str, model: str, api_key: str, temperature: float = 1.0, **kwargs
) -> BaseChatModel:
    """
    Convenience wrapper for ModelFactory.create_model().

    See ModelFactory.create_model() for full documentation.
    """
    return ModelFactory.create_model(
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=temperature,
        **kwargs,
    )
