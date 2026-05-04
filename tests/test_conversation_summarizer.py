"""Durable conversation summarizer guards."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from app.ai.conversation_summarizer import ConversationSummarizer
from app.ai.schemas import AgentMessage, MessageRole


class _FakeProviderService:
    def __init__(self):
        self.calls: list[tuple[UUID, str]] = []

    def resolve_provider_credentials(self, user_id: UUID, provider_type: str):
        self.calls.append((user_id, provider_type))
        return {
            "configured": True,
            "api_key": "user-gemini-key",
            "key_source": "db",
        }


@pytest.mark.asyncio
async def test_summarizer_caches_user_gemini_key(monkeypatch):
    api_keys: list[str | None] = []

    async def fake_generate_summary(
        _messages,
        *,
        config,
        existing_summary,
        api_key_override=None,
    ):
        _ = config, existing_summary
        api_keys.append(api_key_override)
        return "- summarized"

    monkeypatch.setattr(
        "app.ai.conversation_summarizer.generate_summary",
        fake_generate_summary,
    )

    provider_service = _FakeProviderService()
    summarizer = ConversationSummarizer(
        settings=SimpleNamespace(
            memory_summary_max_tokens=1500,
            memory_summary_timeout_seconds=30,
        ),
        provider_service=provider_service,
    )
    user_id = uuid4()
    message = AgentMessage(role=MessageRole.USER, content="hello")

    await summarizer.summarize(
        existing_summary=None,
        messages=[message],
        user_id=str(user_id),
    )
    await summarizer.summarize(
        existing_summary="- old",
        messages=[message],
        user_id=str(user_id),
    )

    assert provider_service.calls == [(user_id, "gemini")]
    assert api_keys == ["user-gemini-key", "user-gemini-key"]
