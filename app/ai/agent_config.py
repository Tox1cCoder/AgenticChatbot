"""
Centralized agent configuration module.

This module provides:
- AGENT_CONFIG: Unified configuration dict for all agents
- create_langchain_model(): Factory for ChatGoogleGenerativeAI instances
- create_gemini_client(): Factory for raw GenAI Client instances
"""

import logging
from typing import Any, Dict, Optional

from google import genai
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI

from ..core.config import settings

logger = logging.getLogger(__name__)


# Unified agent configuration
# Models are sourced from settings but can be overridden here
AGENT_CONFIG = {
    "chat": {
        "model": settings.chat_agent_model,
        "temperature": 1.0,
    },
    "rag": {
        "model": settings.rag_agent_model,
        "temperature": 1.0,
    },
    "search": {
        "model": settings.search_agent_model,
        "temperature": 1.0,
    },
    "planning": {
        "model": settings.chat_agent_model,  # Uses chat model
        "temperature": 1.0,
    },
    "image_generator": {
        "model": settings.image_generator_model,
        "temperature": 1.0,
        "langchain_model": "gemini-3-flash-preview",  # For tool calling
    },
    "suggestion": {
        "model": "gemini-3-flash-preview",
        "temperature": 1.0,
        "max_output_tokens": 512,
    },
    "title_generator": {
        "model": "gemini-3-flash-preview",
        "temperature": 1,
    },
    "summarization": {
        "model": (
            getattr(settings, "summarization_model", None) or "gemini-3-flash-preview"
        ),
        "temperature": 1,
    },
}


def get_api_key() -> str:
    """Get and normalize the Gemini API key."""
    api_key = settings.gemini_api_key
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")

    # Handle legacy format
    if api_key.startswith("GEMINI_API_KEY="):
        api_key = api_key.split("=", 1)[1].strip()

    return api_key


def create_gemini_client() -> genai.Client:
    """
    Create a raw GenAI Client instance.

    Returns:
        genai.Client: Initialized Gemini client
    """
    api_key = get_api_key()
    return genai.Client(api_key=api_key)


def _build_thinking_config(model_name: str) -> Optional[types.ThinkingConfig]:
    if not settings.enable_thinking:
        return None

    thinking_kwargs: Dict[str, Any] = {
        "include_thoughts": settings.include_thoughts_in_response
    }

    if "2.5" in model_name or "flash-latest" in model_name.lower():
        thinking_budget = settings.thinking_budget
        if thinking_budget == -1:
            thinking_budget = 8192
        thinking_kwargs["thinking_budget"] = thinking_budget
    else:
        thinking_kwargs["thinking_level"] = settings.thinking_level

    return types.ThinkingConfig(**thinking_kwargs)


def build_gemini_generate_config(
    model_name: str,
    include_thinking: bool = True,
    enable_code_execution: Optional[bool] = None,
    **extra_config: Any,
) -> Optional[types.GenerateContentConfig]:
    """
    Build a shared Gemini GenerateContentConfig used by direct SDK calls.

    Includes:
    - Thinking configuration (if enabled)
    - Gemini code execution tool for Agentic Vision (if enabled)
    - Optional extra config fields
    """
    config_kwargs: Dict[str, Any] = {}

    if include_thinking:
        thinking_config = _build_thinking_config(model_name)
        if thinking_config is not None:
            config_kwargs["thinking_config"] = thinking_config

    use_code_execution = (
        settings.enable_gemini_code_execution
        if enable_code_execution is None
        else bool(enable_code_execution)
    )
    if use_code_execution:
        existing_tools = extra_config.pop("tools", None) or []
        code_execution_tool = types.Tool(code_execution=types.ToolCodeExecution())
        config_kwargs["tools"] = [*existing_tools, code_execution_tool]

    for key, value in extra_config.items():
        if value is not None:
            config_kwargs[key] = value

    if not config_kwargs:
        return None

    return types.GenerateContentConfig(**config_kwargs)


def create_langchain_model(
    agent_type: str,
    model_override: Optional[str] = None,
    temperature_override: Optional[float] = None,
    include_thinking: bool = True,
) -> ChatGoogleGenerativeAI:
    """
    Create a ChatGoogleGenerativeAI instance with proper configuration.

    Args:
        agent_type: One of the keys in AGENT_CONFIG (e.g., "chat", "rag", "search")
        model_override: Optional model name to override the config
        temperature_override: Optional temperature to override the config
        include_thinking: Whether to include thinking configuration (default True)

    Returns:
        ChatGoogleGenerativeAI: Configured LangChain model
    """
    if agent_type not in AGENT_CONFIG:
        raise ValueError(
            f"Unknown agent type: {agent_type}. Valid types: {list(AGENT_CONFIG.keys())}"
        )

    config = AGENT_CONFIG[agent_type]
    api_key = get_api_key()

    model_name = model_override or config["model"]
    temperature = (
        temperature_override
        if temperature_override is not None
        else config["temperature"]
    )

    model_kwargs = {
        "model": model_name,
        "google_api_key": api_key,
        "temperature": temperature,
    }

    # Configure thinking based on settings and model version
    if include_thinking and settings.enable_thinking:
        # Enable thought output in responses
        if settings.include_thoughts_in_response:
            model_kwargs["include_thoughts"] = True

        # Use thinking_budget for Gemini 2.5, thinking_level for Gemini 3
        if "2.5" in model_name or "flash-latest" in model_name.lower():
            thinking_budget = settings.thinking_budget
            if thinking_budget == -1:
                thinking_budget = 8192
            model_kwargs["thinking_budget"] = thinking_budget
        else:
            model_kwargs["thinking_level"] = settings.thinking_level
    return ChatGoogleGenerativeAI(**model_kwargs)
