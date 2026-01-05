"""
Centralized agent configuration module.

This module provides:
- AGENT_CONFIG: Unified configuration dict for all agents
- create_langchain_model(): Factory for ChatGoogleGenerativeAI instances
- create_gemini_client(): Factory for raw GenAI Client instances
"""

import logging
from typing import Optional

from google import genai
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
            settings.summarization_model
            if hasattr(settings, "summarization_model")
            else "gemini-3-flash-preview"
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
