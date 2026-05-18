"""Tests for Phase 10 runtime model override plumbing.

Covers:
- ``ResolvedRuntimeModelConfig.reasoning_effort`` is threaded through to the
  underlying provider model factory for OpenAI (``reasoning.effort``) and
  Gemini (``thinking_level_override``).
- ``ModelConfigService.resolve_runtime_config`` accepts a request-only
  ``reasoning_effort`` field on the override without persisting it.
- Per-subagent runtime extensions: ``canvas`` and ``image_generator`` work for
  runtime overrides, even though the persisted catalog still only covers
  ``chat``, ``rag``, ``search``, ``planning``.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.core.runtime_modeling import ResolvedRuntimeModelConfig


class _FakeAgent:
    """Concrete BaseAgent that bypasses tool init and gemini setup."""

    def __init__(self, agent_config_key: str = "search", model_name: str = "gpt-5.4"):
        self.agent_config_key = agent_config_key
        self.model_name = model_name
        self.gemini_client = None
        self.langchain_model = object()


def _runtime_config(
    *,
    provider: str = "openai",
    model: str = "gpt-5.4",
    reasoning_effort: str | None = None,
    api_key: str | None = "fake-key",
) -> ResolvedRuntimeModelConfig:
    return ResolvedRuntimeModelConfig(
        agent_key="search",
        provider=provider,
        model=model,
        temperature=0.5,
        api_key=api_key,
        key_source="env",
        source="request",
        warnings=[],
        is_custom_model=True,
        reasoning_effort=reasoning_effort,
    )


# ---------------------------------------------------------------------------
# OpenAI explicit reasoning_effort plumbed to ModelFactory
# ---------------------------------------------------------------------------


def test_openai_explicit_reasoning_effort_reaches_model_factory(monkeypatch):
    from app.ai.agents import base_agent as base_module

    captured: dict[str, Any] = {}

    def fake_create_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "app.ai.model_factory.ModelFactory.create_model",
        staticmethod(fake_create_model),
    )

    agent = _FakeAgent()
    runtime = _runtime_config(reasoning_effort="high")
    base_module.BaseAgent._create_langchain_model_from_runtime(agent, runtime)

    reasoning = captured.get("reasoning")
    assert isinstance(reasoning, dict)
    assert reasoning.get("effort") == "high"


def test_openai_default_reasoning_summary_unchanged_when_no_explicit_effort(monkeypatch):
    """When the override has no ``reasoning_effort``, the previous
    default behavior must remain: summary='auto' for non-o-series models.
    """
    from app.ai.agents import base_agent as base_module

    captured: dict[str, Any] = {}

    def fake_create_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "app.ai.model_factory.ModelFactory.create_model",
        staticmethod(fake_create_model),
    )

    agent = _FakeAgent(model_name="gpt-4o")
    runtime = _runtime_config(model="gpt-4o")
    base_module.BaseAgent._create_langchain_model_from_runtime(agent, runtime)

    reasoning = captured.get("reasoning")
    assert reasoning == {"summary": "auto"}


# ---------------------------------------------------------------------------
# Gemini reasoning_effort -> thinking_level_override
# ---------------------------------------------------------------------------


def test_gemini_reasoning_effort_low_maps_to_thinking_level_low(monkeypatch):
    from app.ai.agents import base_agent as base_module

    captured: dict[str, Any] = {}

    def fake_create(*args, **kwargs):
        captured.update(kwargs)
        if args:
            captured["_positional"] = args
        return object()

    monkeypatch.setattr(
        "app.ai.agents.base_agent.create_langchain_model",
        fake_create,
    )

    agent = _FakeAgent(agent_config_key="search", model_name="gemini-3.1-pro-preview")
    runtime = _runtime_config(
        provider="gemini",
        model="gemini-3.1-pro-preview",
        reasoning_effort="low",
        api_key="gemini-key",
    )
    base_module.BaseAgent._create_langchain_model_from_runtime(agent, runtime)

    assert captured.get("thinking_level_override") == "low"


def test_gemini_reasoning_effort_high_maps_to_high(monkeypatch):
    from app.ai.agents import base_agent as base_module

    captured: dict[str, Any] = {}

    def fake_create(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "app.ai.agents.base_agent.create_langchain_model",
        fake_create,
    )

    agent = _FakeAgent(agent_config_key="search", model_name="gemini-3-flash-preview")
    runtime = _runtime_config(
        provider="gemini",
        model="gemini-3-flash-preview",
        reasoning_effort="xhigh",
        api_key="gemini-key",
    )
    base_module.BaseAgent._create_langchain_model_from_runtime(agent, runtime)

    assert captured.get("thinking_level_override") == "high"


def test_gemini_flash_reasoning_effort_none_normalizes_to_minimal(monkeypatch):
    """Flash supports minimal/low/medium/high. ``none`` -> ``minimal``."""
    from app.ai.agents import base_agent as base_module

    captured: dict[str, Any] = {}

    def fake_create(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "app.ai.agents.base_agent.create_langchain_model",
        fake_create,
    )

    agent = _FakeAgent(agent_config_key="search", model_name="gemini-3-flash-preview")
    runtime = _runtime_config(
        provider="gemini",
        model="gemini-3-flash-preview",
        reasoning_effort="none",
        api_key="gemini-key",
    )
    base_module.BaseAgent._create_langchain_model_from_runtime(agent, runtime)

    assert captured.get("thinking_level_override") == "minimal"


def test_gemini_pro_reasoning_effort_medium_normalizes_to_high(monkeypatch):
    """Gemini 3 Pro only supports ``low``/``high``; ``medium`` -> ``high``."""
    from app.ai.agents import base_agent as base_module

    captured: dict[str, Any] = {}

    def fake_create(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "app.ai.agents.base_agent.create_langchain_model",
        fake_create,
    )

    agent = _FakeAgent(agent_config_key="search", model_name="gemini-3.1-pro-preview")
    runtime = _runtime_config(
        provider="gemini",
        model="gemini-3.1-pro-preview",
        reasoning_effort="medium",
        api_key="gemini-key",
    )
    base_module.BaseAgent._create_langchain_model_from_runtime(agent, runtime)

    assert captured.get("thinking_level_override") == "high"


# ---------------------------------------------------------------------------
# ResolvedRuntimeModelConfig.reasoning_effort exists
# ---------------------------------------------------------------------------


def test_resolved_runtime_model_config_has_reasoning_effort_field():
    config = ResolvedRuntimeModelConfig(
        agent_key="search",
        provider="openai",
        model="gpt-5.4",
        temperature=1.0,
        api_key="key",
        key_source="env",
        source="request",
        reasoning_effort="high",
    )
    assert config.reasoning_effort == "high"


# ---------------------------------------------------------------------------
# Apply metadata: reasoning_effort shows up in response metadata
# ---------------------------------------------------------------------------


def test_apply_runtime_metadata_includes_reasoning_effort():
    from app.ai.agents import base_agent as base_module

    agent = _FakeAgent()
    metadata: dict[str, Any] = {}
    runtime = _runtime_config(reasoning_effort="medium")
    base_module.BaseAgent._apply_runtime_metadata(agent, metadata, runtime)

    assert metadata.get("reasoning_effort") == "medium"


def test_apply_runtime_metadata_omits_reasoning_effort_when_unset():
    from app.ai.agents import base_agent as base_module

    agent = _FakeAgent()
    metadata: dict[str, Any] = {}
    runtime = _runtime_config()
    base_module.BaseAgent._apply_runtime_metadata(agent, metadata, runtime)

    assert "reasoning_effort" not in metadata


# ---------------------------------------------------------------------------
# Runtime-supported agent keys: canvas + image_generator
# ---------------------------------------------------------------------------


def test_canvas_agent_key_supported_for_runtime_overrides():
    from app.ai.agents.base_agent import _MODEL_REQUEST_SUPPORTED_AGENT_KEYS

    assert "canvas" in _MODEL_REQUEST_SUPPORTED_AGENT_KEYS


def test_image_generator_agent_key_supported_for_runtime_overrides():
    from app.ai.agents.base_agent import _MODEL_REQUEST_SUPPORTED_AGENT_KEYS

    assert "image_generator" in _MODEL_REQUEST_SUPPORTED_AGENT_KEYS


def test_model_config_service_has_split_supported_agent_keys():
    """The runtime resolver should accept canvas/image_generator runtime-only
    overrides; persisted agent_model_configs must stay on the original four.
    """
    from app.services.model_config_service import (
        SUPPORTED_AGENT_KEYS,
        SUPPORTED_RUNTIME_AGENT_KEYS,
    )

    assert set(SUPPORTED_AGENT_KEYS) == {"chat", "rag", "search", "planning"}
    assert "canvas" in SUPPORTED_RUNTIME_AGENT_KEYS
    assert "image_generator" in SUPPORTED_RUNTIME_AGENT_KEYS
    # Runtime keys must be a strict superset of persisted keys.
    assert set(SUPPORTED_AGENT_KEYS) <= set(SUPPORTED_RUNTIME_AGENT_KEYS)
