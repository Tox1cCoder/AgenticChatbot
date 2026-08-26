from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.ai.workflow.routing import (
    RoutingConfigurationError,
    RoutingDecisionValidator,
    RoutingService,
)


def _service(*, router_model="gemini-3-flash-preview", provider="gemini", resolver=None):
    resolver = resolver or MagicMock()
    return RoutingService(
        runtime_model_resolver=resolver,
        model_factory=MagicMock(),
        settings=SimpleNamespace(
            routing_timeout_seconds=8.0,
            routing_max_attempts=2,
            router_model=router_model,
            router_provider=provider,
        ),
        validator=RoutingDecisionValidator(),
    )


def test_static_validation_does_not_require_user_scoped_credentials():
    resolver = MagicMock()
    service = _service(resolver=resolver)

    service.validate_static_configuration()

    assert resolver.resolve_runtime_config.call_count == 0


def test_static_validation_accepts_installed_structured_output_adapters():
    for provider in ("gemini", "openai"):
        _service(provider=provider).validate_static_configuration()


def test_static_validation_rejects_a_blank_router_model():
    with pytest.raises(RoutingConfigurationError):
        _service(router_model="   ").validate_static_configuration()


def test_static_validation_rejects_an_adapter_without_structured_output():
    with pytest.raises(RoutingConfigurationError) as exc:
        _service(provider="anthropic").validate_static_configuration()
    assert "structured" in str(exc.value).lower() or "provider" in str(exc.value).lower()


def test_static_validation_rejects_invalid_attempt_and_timeout_settings():
    service = RoutingService(
        runtime_model_resolver=MagicMock(),
        model_factory=MagicMock(),
        settings=SimpleNamespace(
            routing_timeout_seconds=0.0,
            routing_max_attempts=2,
            router_model="gemini-3-flash-preview",
            router_provider="gemini",
        ),
        validator=RoutingDecisionValidator(),
    )
    with pytest.raises(RoutingConfigurationError):
        service.validate_static_configuration()

    service = RoutingService(
        runtime_model_resolver=MagicMock(),
        model_factory=MagicMock(),
        settings=SimpleNamespace(
            routing_timeout_seconds=8.0,
            routing_max_attempts=5,
            router_model="gemini-3-flash-preview",
            router_provider="gemini",
        ),
        validator=RoutingDecisionValidator(),
    )
    with pytest.raises(RoutingConfigurationError):
        service.validate_static_configuration()


def test_model_config_service_exposes_structured_output_capability():
    from app.services.model_config_service import ModelConfigService

    service = ModelConfigService(repository=MagicMock(), provider_service=MagicMock())

    assert service._build_capabilities("gemini", "gemini-3-flash-preview", {})[
        "supports_structured_output"
    ]
    assert service._build_capabilities("openai", "gpt-5", {})["supports_structured_output"]
    assert not service._build_capabilities("anthropic", "claude", {})["supports_structured_output"]


def test_resolve_runtime_config_defaults_preserve_existing_caller_behavior():
    """Existing agents must keep provider fallback; only the router opts out."""
    import inspect

    from app.interfaces.runtime_model_resolver_interface import IRuntimeModelResolver

    signature = inspect.signature(IRuntimeModelResolver.resolve_runtime_config)
    assert signature.parameters["allow_provider_fallback"].default is True
    assert signature.parameters["require_capabilities"].default == frozenset()


def test_settings_expose_routing_bounds():
    from app.core.config import settings

    assert settings.routing_timeout_seconds == 8.0
    assert settings.routing_max_attempts == 2
    assert settings.workflow_graph_version == "routing-v2"
