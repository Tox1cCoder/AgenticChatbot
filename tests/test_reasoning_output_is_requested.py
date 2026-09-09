"""Enabling reasoning is not the same as being shown it.

Both providers separate *doing* the reasoning from *returning* it. Gemini
thinks but emits no thought parts unless ``include_thoughts`` is set; the
OpenAI Responses API returns a reasoning summary only when one is asked for.
Setting an effort alone therefore produces a turn that reasons and streams
nothing, which reads as "the model never thought".

These pin the request side only. Whether the deltas then reach the UI is the
stream layer's contract, not this one's.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.ai.model_factory import ModelFactory


@pytest.fixture
def captured(monkeypatch):
    seen: dict = {}

    def constructor(**kwargs):
        seen.clear()
        seen.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr("app.ai.model_factory.ChatGoogleGenerativeAI", constructor)
    monkeypatch.setattr("app.ai.model_factory.ChatOpenAI", constructor)
    return seen


def _runtime(provider: str, model: str, effort: str | None):
    return SimpleNamespace(
        provider=provider,
        model=model,
        api_key="key",
        temperature=1.0,
        agent_key="chat",
        reasoning_effort=effort,
        capabilities={"supports_reasoning": True},
    )


@pytest.mark.parametrize("model", ["gemini-3-flash-preview", "gemini-2.5-pro"])
def test_gemini_asks_for_the_thoughts_it_enables(captured, model):
    ModelFactory.create_model_from_runtime(_runtime("gemini", model, "high"))

    assert captured.get("include_thoughts") is True, (
        "thinking was configured but the thoughts were never requested, so the "
        "provider returns no thought parts to stream"
    )


def test_gemini_3_uses_a_control_the_client_actually_reads(captured):
    """``thinking_level`` is not a field on ChatGoogleGenerativeAI.

    Its model config ignores unknown keys, so passing only ``thinking_level``
    configures nothing at all — the kwarg is accepted and dropped.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    assert "thinking_level" not in ChatGoogleGenerativeAI.model_fields
    assert ChatGoogleGenerativeAI.model_config.get("extra") == "ignore"

    ModelFactory.create_model_from_runtime(
        _runtime("gemini", "gemini-3-flash-preview", "high")
    )

    supported = {"reasoning_effort", "thinking_budget", "thinking_config"}
    assert supported & captured.keys(), (
        "no thinking control the installed client reads was passed; "
        f"got {sorted(captured)}"
    )


def test_openai_requests_a_reasoning_summary(captured):
    ModelFactory.create_model_from_runtime(_runtime("openai", "gpt-5.6-luna", "high"))

    assert captured["use_responses_api"] is True
    assert captured["reasoning"]["effort"] == "high"
    assert captured["reasoning"].get("summary"), (
        "the Responses API returns reasoning text only when a summary is asked "
        "for; an effort on its own reasons invisibly"
    )


def test_reasoning_off_asks_for_nothing(captured):
    """``none`` must not smuggle reasoning back in through the summary."""
    ModelFactory.create_model_from_runtime(_runtime("openai", "gpt-5.6-luna", "none"))

    assert captured.get("use_responses_api") is not True
    assert captured["reasoning_effort"] == "none"
    assert "reasoning" not in captured
