"""Durable conversation memory summarizer.

This module is the off-hot-path counterpart to ``app.ai.history``. The
history provider reads the durable summary; this module writes it.

The model call is delegated to the existing
``summarization_middleware.generate_summary`` so we keep one single Gemini
prompt formatter / token estimator. ``ConversationSummarizer`` is the
narrow adapter the durable-memory pipeline calls — it enforces the
timeout, fail-closed behaviour, AgentMessage-to-LangChain conversion, and
per-user model credential lookup.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import UUID

from cachetools import TTLCache
from langchain_core.messages import AIMessage, HumanMessage

from app.ai.schemas import AgentMessage, MessageRole
from app.ai.summarization_middleware import (
    SummarizationConfig,
    _get_config,
    generate_summary,
)
from app.interfaces.runtime_model_resolver_interface import IRuntimeModelResolver

logger = logging.getLogger(__name__)


def _agent_message_to_langchain(message: AgentMessage):
    if message.role == MessageRole.ASSISTANT:
        return AIMessage(content=message.content or "")
    return HumanMessage(content=message.content or "")


class ConversationSummarizer:
    """Generate or refresh a conversation summary, fail-closed on errors."""

    def __init__(
        self,
        settings: Any | None = None,
        runtime_model_resolver: IRuntimeModelResolver | None = None,
        provider_service: Any | None = None,
    ):
        self.settings = settings
        self.runtime_model_resolver = runtime_model_resolver
        self.provider_service = provider_service
        ttl_seconds = int(getattr(settings, "memory_cache_ttl_seconds", 60) or 60)
        self._api_key_cache: TTLCache = TTLCache(maxsize=1024, ttl=ttl_seconds)

    async def summarize(
        self,
        *,
        existing_summary: str | None,
        messages: list[AgentMessage],
        user_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> str | None:
        if not messages:
            return existing_summary

        config: SummarizationConfig = _get_config()
        if self.settings is not None:
            max_summary_tokens = getattr(
                self.settings, "memory_summary_max_tokens", config.max_summary_tokens
            )
            if max_summary_tokens:
                config.max_summary_tokens = int(max_summary_tokens)

        timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else getattr(self.settings, "memory_summary_timeout_seconds", 30)
        )
        if timeout is None or timeout <= 0:
            timeout = 30

        api_key_override = self._resolve_user_api_key(user_id)

        try:
            return await asyncio.wait_for(
                generate_summary(
                    [_agent_message_to_langchain(m) for m in messages],
                    config=config,
                    existing_summary=existing_summary,
                    api_key_override=api_key_override,
                ),
                timeout=float(timeout),
            )
        except (TimeoutError, asyncio.TimeoutError):
            logger.warning(
                "Conversation summarization timed out after %.1fs (user=%s) — keeping previous summary",
                float(timeout),
                user_id,
            )
            return None
        except Exception as exc:
            logger.warning(
                "Conversation summarization failed (user=%s): %s — keeping previous summary",
                user_id,
                exc,
            )
            return None

    def _resolve_user_api_key(self, user_id: str | None) -> str | None:
        """Look up the Gemini key configured for ``user_id``.

        Falls back to ``None`` (caller uses the global key) when no provider
        service is wired, the user id is missing/invalid, or lookup fails.
        Results are cached briefly per user because summary refresh can drain
        multiple pending turns for the same conversation.
        """
        if not user_id:
            return None
        try:
            user_uuid = UUID(str(user_id))
        except (TypeError, ValueError):
            return None

        cache_key = str(user_uuid)
        if cache_key in self._api_key_cache:
            return self._api_key_cache[cache_key]

        api_key: str | None = None
        if self.provider_service is not None:
            try:
                credentials = self.provider_service.resolve_provider_credentials(
                    user_uuid,
                    "gemini",
                )
                raw_key = credentials.get("api_key") if isinstance(credentials, dict) else None
                if isinstance(raw_key, str) and raw_key.strip():
                    api_key = raw_key.strip()
            except Exception as exc:
                logger.warning(
                    "Gemini credential lookup for summarization failed (user=%s): %s — falling back to global key",
                    user_id,
                    exc,
                )
        elif self.runtime_model_resolver is not None:
            # Compatibility fallback for tests or custom wiring. Only use the
            # key if the resolved provider is Gemini; the summary model is
            # Gemini-specific and must not receive an OpenAI key.
            api_key = self._resolve_user_api_key_from_runtime(user_uuid, user_id)

        self._api_key_cache[cache_key] = api_key
        return api_key

    def _resolve_user_api_key_from_runtime(
        self,
        user_uuid: UUID,
        raw_user_id: str,
    ) -> str | None:
        try:
            resolved = self.runtime_model_resolver.resolve_runtime_config(
                user_uuid,
                "chat",
            )
        except Exception as exc:
            logger.warning(
                "Runtime model resolution for summarization failed (user=%s): %s — falling back to global key",
                raw_user_id,
                exc,
            )
            return None
        if getattr(resolved, "provider", None) != "gemini":
            return None
        api_key = getattr(resolved, "api_key", None)
        return api_key.strip() if isinstance(api_key, str) and api_key.strip() else None


__all__ = ["ConversationSummarizer"]
