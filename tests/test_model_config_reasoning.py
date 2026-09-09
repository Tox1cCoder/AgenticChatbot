from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.api.model_config import AgentModelConfigPatch
from app.models.agent_model_config import AgentModelConfig
from app.repositories.agent_model_config import AgentModelConfigRepository
from app.services.model_config_service import ModelConfigService

USER_ID = uuid4()


def _snapshot(provider: str = "gemini") -> dict:
    model = "gemini-3.6-flash" if provider == "gemini" else "gpt-5.6-sol"
    return {
        "provider_type": provider,
        "configured": True,
        "key_source": "db",
        "sync_status": "ready",
        "models": [
            {
                "id": model,
                "provider_type": provider,
                "supports_reasoning": True,
            }
        ],
    }


def _service(existing=None) -> tuple[ModelConfigService, MagicMock]:
    repository = MagicMock()
    repository.get_by_user_and_agent_key.return_value = existing
    repository.get_all_by_user.return_value = [existing] if existing else []
    provider = MagicMock()
    provider.get_cached_provider_status.side_effect = lambda _uid, name: (
        _snapshot(name) if name in {"gemini", "openai"} else {"configured": False, "models": []}
    )
    provider.sync_provider_models.side_effect = lambda **kwargs: _snapshot(kwargs["provider_type"])
    provider.resolve_provider_credentials.return_value = {"api_key": "key", "key_source": "db"}
    return ModelConfigService(repository, provider), repository


def test_persistence_surface_has_reasoning_effort() -> None:
    assert "reasoning_effort" in AgentModelConfig.__table__.columns
    assert "reasoning_effort" in inspect.signature(AgentModelConfigRepository.upsert).parameters


def test_explicit_null_survives_patch_serialization() -> None:
    patch = AgentModelConfigPatch(reasoning_effort=None)
    assert patch.model_dump(exclude_unset=True) == {"reasoning_effort": None}


@pytest.mark.asyncio
async def test_validate_patch_accepts_native_level() -> None:
    service, _ = _service()
    validated = await service._validate_patch(
        user_id=USER_ID,
        agent_key="chat",
        raw_patch={
            "provider": "gemini",
            "model": "gemini-3.6-flash",
            "reasoning_effort": "medium",
        },
        effective_config=service.get_effective_model_config(USER_ID),
    )
    assert validated["reasoning_effort"] == "medium"


@pytest.mark.asyncio
async def test_validate_patch_rejects_unsupported_native_level() -> None:
    service, _ = _service()
    with pytest.raises(ValueError, match="Accepted: minimal, low, medium, high"):
        await service._validate_patch(
            user_id=USER_ID,
            agent_key="chat",
            raw_patch={
                "provider": "gemini",
                "model": "gemini-3.6-flash",
                "reasoning_effort": "max",
            },
            effective_config=service.get_effective_model_config(USER_ID),
        )


def test_incompatible_saved_level_falls_back_to_provider_default() -> None:
    row = SimpleNamespace(
        agent_key="chat",
        provider_type="openai",
        model="gpt-5.6-sol",
        temperature=1.0,
        allow_custom_model=False,
        reasoning_effort="minimal",
    )
    service, _ = _service(row)
    config = service.get_effective_model_config(USER_ID)["chat"]
    assert config["reasoning_effort"] is None
    assert any("unsupported" in warning.lower() for warning in config["warnings"])


def test_runtime_fallback_candidate_carries_its_own_model_metadata() -> None:
    service, _ = _service()
    snapshots = {provider: _snapshot(provider) for provider in ("gemini", "openai")}

    fallback = service._build_runtime_fallback_candidate(
        user_id=USER_ID,
        agent_key="search",
        provider_snapshots=snapshots,
        effective_config={"search": {"provider": "gemini", "temperature": 0.4}},
        current_provider="gemini",
    )

    assert fallback is not None
    assert fallback.provider == "openai"
    assert fallback.capabilities["supports_reasoning"] is True
    assert fallback.context_window["provider"] == "openai"
    assert fallback.context_window["model"] == "gpt-5.6-sol"
    assert fallback.reasoning_effort is None
