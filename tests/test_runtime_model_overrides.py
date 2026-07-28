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


def test_openai_max_reasoning_effort_is_not_truncated(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_create_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "app.ai.model_factory.ModelFactory.create_model",
        staticmethod(fake_create_model),
    )
    agent = _FakeAgent(model_name="gpt-5.6-sol")
    base_module = __import__("app.ai.agents.base_agent", fromlist=["BaseAgent"])
    base_module.BaseAgent._create_langchain_model_from_runtime(
        agent, _runtime_config(model="gpt-5.6-sol", reasoning_effort="max")
    )
    assert captured["reasoning"] == {"effort": "max"}


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


def test_gemini_reasoning_effort_medium_is_passed_unchanged(monkeypatch):
    from app.ai.agents import base_agent as base_module

    captured: dict[str, Any] = {}

    def fake_create(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "app.ai.agents.base_agent.create_langchain_model",
        fake_create,
    )

    agent = _FakeAgent(agent_config_key="search", model_name="gemini-3.6-flash")
    runtime = _runtime_config(
        provider="gemini",
        model="gemini-3.6-flash",
        reasoning_effort="medium",
        api_key="gemini-key",
    )
    base_module.BaseAgent._create_langchain_model_from_runtime(agent, runtime)

    assert captured.get("thinking_level_override") == "medium"


def test_gemini_flash_minimal_is_passed_unchanged(monkeypatch):
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
        reasoning_effort="minimal",
        api_key="gemini-key",
    )
    base_module.BaseAgent._create_langchain_model_from_runtime(agent, runtime)

    assert captured.get("thinking_level_override") == "minimal"


def test_gemini_pro_reasoning_effort_medium_is_passed_unchanged(monkeypatch):
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

    assert captured.get("thinking_level_override") == "medium"


def test_gemini_25_reasoning_effort_is_passed_to_model_builder(monkeypatch):
    from app.ai.agents import base_agent as base_module

    captured: dict[str, Any] = {}

    def fake_create(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("app.ai.agents.base_agent.create_langchain_model", fake_create)
    agent = _FakeAgent(agent_config_key="search", model_name="gemini-2.5-flash")
    runtime = _runtime_config(
        provider="gemini",
        model="gemini-2.5-flash",
        reasoning_effort="high",
        api_key="gemini-key",
    )

    base_module.BaseAgent._create_langchain_model_from_runtime(agent, runtime)

    assert captured.get("thinking_level_override") == "high"


def test_gemini_25_model_builder_converts_level_to_budget(monkeypatch):
    from app.ai import agent_config

    captured: dict[str, Any] = {}

    def fake_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(agent_config, "ReasoningNormalizedChatGoogleGenerativeAI", fake_model)
    monkeypatch.setattr(agent_config.settings, "enable_thinking", True)

    agent_config.create_langchain_model(
        "search",
        model_override="gemini-2.5-flash",
        api_key_override="gemini-key",
        thinking_level_override="high",
    )

    assert captured["thinking_budget"] == 24576
    assert "thinking_level" not in captured


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


# ---------------------------------------------------------------------------
# Task 4: context_window metadata threaded through ResolvedRuntimeModelConfig
# ---------------------------------------------------------------------------


def _gemini_snapshot(
    *,
    configured: bool = True,
    key_source: str = "env",
    models: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if models is None:
        models = [
            {
                "id": "gemini-3-flash-preview",
                "display_name": "gemini-3-flash-preview",
                "provider_type": "gemini",
                "supports_vision": True,
                "supports_tool_calling": True,
                "supports_streaming": True,
                "supports_reasoning": True,
                "recommended": True,
                "context_window_tokens": 1_048_576,
                "max_input_tokens": 1_048_576,
                "max_output_tokens": 65536,
                "context_window_source": "registry",
                "context_window_known": True,
            }
        ]
    return {
        "configured": configured,
        "key_source": key_source,
        "models": models,
        "sync_status": "ready",
        "provider_type": "gemini",
    }


def _openai_snapshot(
    *,
    configured: bool = True,
    key_source: str = "env",
    models: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if models is None:
        models = [
            {
                "id": "gpt-4o",
                "display_name": "gpt-4o",
                "provider_type": "openai",
                "supports_vision": True,
                "supports_tool_calling": True,
                "supports_streaming": True,
                "supports_reasoning": False,
                "recommended": True,
                "context_window_tokens": 128000,
                "max_input_tokens": 128000,
                "max_output_tokens": 16384,
                "context_window_source": "provider_api",
                "context_window_known": True,
            }
        ]
    return {
        "configured": configured,
        "key_source": key_source,
        "models": models,
        "sync_status": "ready",
        "provider_type": "openai",
    }


def _make_service(
    *,
    provider_snapshots: dict[str, dict[str, Any]] | None = None,
    persisted_rows: list[Any] | None = None,
    credentials_by_provider: dict[str, dict[str, Any]] | None = None,
):
    from unittest.mock import MagicMock

    from app.services.model_config_service import ModelConfigService

    repo = MagicMock()
    repo.get_all_by_user.return_value = persisted_rows or []
    repo.get_by_user_and_agent_key.return_value = None

    snapshots = provider_snapshots or {}
    credentials = credentials_by_provider or {}

    provider_svc = MagicMock()
    provider_svc.get_cached_provider_status.side_effect = lambda uid, ptype: snapshots.get(
        ptype,
        {"configured": False, "key_source": "none", "models": []},
    )
    provider_svc.resolve_provider_credentials.side_effect = lambda uid, ptype: credentials.get(
        ptype,
        {"api_key": None, "key_source": "none"},
    )

    return ModelConfigService(repository=repo, provider_service=provider_svc)


def test_context_window_default_gemini_when_user_id_none():
    """``user_id=None`` early return must expose ``context_window`` for the
    default Gemini model. The default chat model is in the registry, so
    ``known`` should be True."""
    service = _make_service()

    resolved = service.resolve_runtime_config(user_id=None, agent_key="chat")

    assert resolved.context_window is not None
    assert resolved.context_window["provider"] == "gemini"
    assert resolved.context_window["model"] == resolved.model
    assert isinstance(resolved.context_window.get("known"), bool)
    # The configured default chat model ``gemini-3-flash-preview`` is in the
    # built-in registry.
    assert resolved.context_window["known"] is True
    assert resolved.context_window["source"] == "registry"
    assert resolved.context_window["context_window_tokens"] == 1_048_576


def test_context_window_runtime_override_known_openai_model():
    """A runtime override that selects OpenAI ``gpt-4o`` with a populated
    provider snapshot must surface ``context_window`` sourced from the
    catalog (``provider_api``)."""
    from uuid import uuid4

    snapshots = {
        "gemini": _gemini_snapshot(),
        "openai": _openai_snapshot(),
    }
    credentials = {
        "gemini": {"api_key": "gemini-key", "key_source": "env"},
        "openai": {"api_key": "openai-key", "key_source": "env"},
    }
    service = _make_service(
        provider_snapshots=snapshots,
        credentials_by_provider=credentials,
    )

    resolved = service.resolve_runtime_config(
        user_id=uuid4(),
        agent_key="chat",
        request_override={"provider_type": "openai", "model": "gpt-4o"},
    )

    assert resolved.provider == "openai"
    assert resolved.model == "gpt-4o"
    assert resolved.context_window is not None
    assert resolved.context_window["known"] is True
    assert resolved.context_window["source"] == "provider_api"
    assert resolved.context_window["context_window_tokens"] == 128000
    assert resolved.context_window["max_output_tokens"] == 16384


def test_context_window_custom_unknown_model():
    """A custom-model override with an unrecognised model id must yield an
    unknown ``context_window`` payload (``known=False``, ``source='unknown'``)."""
    from uuid import uuid4

    snapshots = {
        "gemini": _gemini_snapshot(),
        "openai": _openai_snapshot(),
    }
    credentials = {
        "gemini": {"api_key": "gemini-key", "key_source": "env"},
        "openai": {"api_key": "openai-key", "key_source": "env"},
    }
    service = _make_service(
        provider_snapshots=snapshots,
        credentials_by_provider=credentials,
    )

    resolved = service.resolve_runtime_config(
        user_id=uuid4(),
        agent_key="chat",
        request_override={
            "provider_type": "openai",
            "model": "weird-custom-id",
            "allow_custom_model": True,
        },
    )

    assert resolved.provider == "openai"
    assert resolved.model == "weird-custom-id"
    assert resolved.is_custom_model is True
    assert resolved.context_window is not None
    assert resolved.context_window["known"] is False
    assert resolved.context_window["source"] == "unknown"
    assert resolved.context_window["context_window_tokens"] is None


def test_context_window_reflects_final_provider_after_fallback():
    """If the requested provider is unconfigured, fallback should swap the
    provider/model AND the ``context_window`` should reflect the fallback
    target, not the originally-requested one."""
    from uuid import uuid4

    snapshots = {
        "gemini": _gemini_snapshot(),
        "openai": _openai_snapshot(configured=False, models=[]),
    }
    credentials = {
        "gemini": {"api_key": "gemini-key", "key_source": "env"},
        "openai": {"api_key": None, "key_source": "none"},
    }
    service = _make_service(
        provider_snapshots=snapshots,
        credentials_by_provider=credentials,
    )

    resolved = service.resolve_runtime_config(
        user_id=uuid4(),
        agent_key="chat",
        request_override={"provider_type": "openai", "model": "gpt-4o"},
    )

    # Fallback should have kicked in: provider becomes gemini.
    assert resolved.provider == "gemini"
    assert resolved.context_window is not None
    assert resolved.context_window["provider"] == "gemini"
    assert resolved.context_window["model"] == resolved.model
    assert resolved.context_window["known"] is True


def test_context_window_default_gemini_resolution_with_configured_provider():
    """No request override, no persisted row -> default Gemini path; the
    final ``context_window`` should reflect the chosen Gemini model."""
    from uuid import uuid4

    snapshots = {"gemini": _gemini_snapshot()}
    credentials = {"gemini": {"api_key": "gemini-key", "key_source": "env"}}
    service = _make_service(
        provider_snapshots=snapshots,
        credentials_by_provider=credentials,
    )

    resolved = service.resolve_runtime_config(
        user_id=uuid4(),
        agent_key="chat",
    )

    assert resolved.provider == "gemini"
    assert resolved.context_window is not None
    assert resolved.context_window["provider"] == "gemini"
    assert resolved.context_window["model"] == resolved.model
    assert resolved.context_window["known"] is True
    # Catalog-driven, so the source should reflect the snapshot's flag.
    assert resolved.context_window["source"] in {"provider_api", "registry"}
    assert resolved.context_window["context_window_tokens"] == 1_048_576
