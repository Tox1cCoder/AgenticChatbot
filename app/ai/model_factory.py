"""Provider-specific LangChain chat model construction."""

import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from ..core.config import settings
from ..core.runtime_modeling import ResolvedRuntimeModelConfig, StrictRuntimeResolutionError
from .reasoning_controls import gemini_reasoning_kwargs, resolve_reasoning_control

logger = logging.getLogger(__name__)


class ModelFactory:
    """Create provider models and apply runtime controls consistently."""

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
        """Build a Gemini or OpenAI chat model."""
        provider = provider.lower()

        if provider == "gemini":
            return ModelFactory._create_gemini_model(
                model, api_key, temperature, thinking_config, **kwargs
            )
        if provider == "openai":
            return ModelFactory._create_openai_model(model, api_key, temperature, timeout, **kwargs)
        raise ValueError(f"Unsupported provider: {provider}. Supported providers: gemini, openai")

    @staticmethod
    def _create_gemini_model(
        model: str,
        api_key: str,
        temperature: float,
        thinking_config: dict[str, Any] | None = None,
        **kwargs,
    ) -> ChatGoogleGenerativeAI:
        model_kwargs = {
            "model": model,
            "google_api_key": api_key,
            "temperature": temperature,
            "max_retries": 0,
        }

        if thinking_config and thinking_config.get("enabled"):
            if "2.5" in model.lower():
                model_kwargs["thinking_budget"] = thinking_config.get("budget", 8192)
            elif "3" in model.lower():
                model_kwargs["thinking_level"] = thinking_config.get("level", "high")

            if thinking_config.get("include_thoughts"):
                model_kwargs["include_thoughts"] = True

        model_kwargs.update(kwargs)

        logger.info("Creating Gemini model %s with temperature=%s", model, temperature)
        return ChatGoogleGenerativeAI(**model_kwargs)

    @staticmethod
    def _create_openai_model(
        model: str,
        api_key: str,
        temperature: float,
        timeout: int | None = None,
        **kwargs,
    ) -> ChatOpenAI:
        model_kwargs = {
            "model": model,
            "openai_api_key": api_key,
            "temperature": temperature,
            "timeout": timeout or settings.openai_request_timeout_seconds,
            "max_retries": 0,
        }
        model_kwargs.update(kwargs)

        logger.info(
            "Creating OpenAI model %s with temperature=%s, streaming=%s, reasoning=%s",
            model,
            temperature,
            model_kwargs.get("streaming", False),
            model_kwargs.get("reasoning"),
        )
        return ChatOpenAI(**model_kwargs)

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

        runtime_kwargs = dict(kwargs)
        effort = config.reasoning_effort
        provider = config.provider.lower()
        if provider == "gemini" and effort:
            runtime_kwargs.update(gemini_reasoning_kwargs(model_id, effort))
            if effort != "none":
                # Enabling thinking is not the same as being shown it: without
                # `include_thoughts` the provider reasons and returns no thought
                # parts at all, so the reasoning channel has nothing to stream.
                runtime_kwargs.setdefault("include_thoughts", True)
                # `thinking_level` is not a field on ChatGoogleGenerativeAI and
                # the model config ignores extras, so on its own it configures
                # nothing. `reasoning_effort` is the supported control and takes
                # the same level names.
                if "thinking_level" in runtime_kwargs and "thinking_budget" not in runtime_kwargs:
                    runtime_kwargs.setdefault(
                        "reasoning_effort", runtime_kwargs["thinking_level"]
                    )
        elif provider == "openai":
            control = resolve_reasoning_control(
                provider,
                model_id,
                supports_reasoning=config.capabilities.get("supports_reasoning"),
            )
            effective_effort = effort if effort is not None else control.default_level
            if effort == "none":
                runtime_kwargs.pop("reasoning", None)
                runtime_kwargs["reasoning_effort"] = "none"
                runtime_kwargs["use_responses_api"] = False
            elif effective_effort not in {None, "none"} or runtime_kwargs.get("reasoning"):
                runtime_kwargs["use_responses_api"] = True
                reasoning = dict(runtime_kwargs.get("reasoning") or {})
                if effort:
                    reasoning["effort"] = effort
                # The Responses API returns reasoning *summaries* only when they
                # are asked for. With an effort but no summary the model reasons
                # and emits nothing visible, which reads downstream as "the
                # model never thought".
                reasoning.setdefault("summary", "auto")
                runtime_kwargs["reasoning"] = reasoning

        try:
            return ModelFactory.create_model(
                provider=provider,
                model=model_id,
                api_key=api_key,
                temperature=config.temperature,
                **runtime_kwargs,
            )
        except ValueError as exc:
            raise StrictRuntimeResolutionError(
                "unsupported_provider", f"provider {config.provider} is not supported"
            ) from exc

    @staticmethod
    def bind_tools_to_model(
        model: BaseChatModel, tools: list[BaseTool], tool_choice: str = "auto"
    ) -> BaseChatModel:
        """Bind tools using the provider's expected tool-choice shape."""
        if not tools:
            logger.debug("No tools provided, returning model without tool binding")
            return model

        tool_choice_normalized = (
            tool_choice.upper() if tool_choice.lower() in {"auto", "any", "none"} else tool_choice
        )

        logger.debug("Binding %d tools with tool_choice=%s", len(tools), tool_choice_normalized)

        if isinstance(model, ChatGoogleGenerativeAI):
            if tool_choice.lower() in {"auto", "any", "none"}:
                return model.bind_tools(
                    tools,
                    tool_config={"function_calling_config": {"mode": tool_choice_normalized}},
                )
            return model.bind_tools(tools, tool_choice=tool_choice)

        return model.bind_tools(tools, tool_choice=tool_choice)
