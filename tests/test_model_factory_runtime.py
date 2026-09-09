from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.ai.model_factory import ModelFactory
from app.core.runtime_modeling import ResolvedRuntimeModelConfig


def _runtime(
    *,
    provider: str = "openai",
    model: str = "gpt-5.6-luna",
    reasoning_effort: str | None = None,
) -> ResolvedRuntimeModelConfig:
    return ResolvedRuntimeModelConfig(
        agent_key="search",
        provider=provider,
        model=model,
        temperature=0.7,
        api_key="test-key",
        key_source="user",
        source="default",
        reasoning_effort=reasoning_effort,
    )


@pytest.fixture
def captured_openai(monkeypatch):
    captured = {}

    def constructor(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr("app.ai.model_factory.ChatOpenAI", constructor)
    return captured


def test_default_gpt_56_reasoning_uses_responses_api(captured_openai):
    ModelFactory.create_model_from_runtime(_runtime())

    assert captured_openai["use_responses_api"] is True
    # A summary is requested even with no explicit effort: the Responses API
    # returns reasoning text only when asked, and without it the turn shows no
    # thinking at all.
    assert captured_openai["reasoning"] == {"summary": "auto"}


def test_explicit_openai_reasoning_uses_responses_api_and_preserves_effort(captured_openai):
    ModelFactory.create_model_from_runtime(_runtime(reasoning_effort="high"))

    assert captured_openai["use_responses_api"] is True
    assert captured_openai["reasoning"] == {"effort": "high", "summary": "auto"}


def test_explicit_none_keeps_tool_compatible_chat_completions(captured_openai):
    ModelFactory.create_model_from_runtime(_runtime(reasoning_effort="none"))

    assert captured_openai.get("use_responses_api") is not True
    assert captured_openai["reasoning_effort"] == "none"


def test_non_reasoning_openai_model_does_not_force_responses_api(captured_openai):
    ModelFactory.create_model_from_runtime(_runtime(model="gpt-4o"))

    assert captured_openai.get("use_responses_api") is not True


def test_provider_default_none_does_not_force_responses_api(captured_openai):
    ModelFactory.create_model_from_runtime(_runtime(model="gpt-5.4"))

    assert captured_openai.get("use_responses_api") is not True


def test_gemini_runtime_reasoning_reaches_provider_constructor(monkeypatch):
    captured = {}

    def constructor(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr("app.ai.model_factory.ChatGoogleGenerativeAI", constructor)

    ModelFactory.create_model_from_runtime(
        _runtime(
            provider="gemini",
            model="gemini-3-flash-preview",
            reasoning_effort="high",
        )
    )

    assert captured["thinking_level"] == "high"
