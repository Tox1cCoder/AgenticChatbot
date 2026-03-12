"""
Service for managing persistent per-agent model configuration.

Provides:
- Effective (defaults + overrides) config mapping for UI
- Override-only model_request mapping for runtime
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional
from uuid import UUID

from app.ai.agent_config import AGENT_CONFIG
from app.repositories.agent_model_config import AgentModelConfigRepository

SUPPORTED_AGENT_KEYS = ("chat", "rag", "search", "planning")
SUPPORTED_PROVIDERS = ("gemini", "openai")


def _normalize_agent_key(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    key = value.strip().lower()
    return key if key in SUPPORTED_AGENT_KEYS else None


def _normalize_provider(value: Any) -> str:
    if not isinstance(value, str):
        return "gemini"
    provider = value.strip().lower()
    return provider if provider in SUPPORTED_PROVIDERS else "gemini"


def _default_agent_config(agent_key: str) -> Dict[str, Any]:
    agent_cfg = AGENT_CONFIG.get(agent_key, {})
    return {
        "provider": "gemini",
        "model": agent_cfg.get("model"),
        "temperature": float(agent_cfg.get("temperature", 1.0)),
    }


class ModelConfigService:
    def __init__(self, repository: AgentModelConfigRepository):
        self.repository = repository

    def get_effective_model_config(self, user_id: UUID) -> Dict[str, Dict[str, Any]]:
        """
        Return effective per-agent configs (defaults + persisted overrides).

        This is intended for UI use.
        """
        effective: Dict[str, Dict[str, Any]] = {
            agent_key: _default_agent_config(agent_key)
            for agent_key in SUPPORTED_AGENT_KEYS
        }

        for row in self.repository.get_all_by_user(user_id):
            agent_key = _normalize_agent_key(getattr(row, "agent_key", None))
            if not agent_key:
                continue

            provider_type = _normalize_provider(getattr(row, "provider_type", None))
            model = getattr(row, "model", None)
            temperature = getattr(row, "temperature", None)

            if isinstance(model, str) and model.strip():
                effective[agent_key]["model"] = model.strip()
            effective[agent_key]["provider"] = provider_type
            if isinstance(temperature, (int, float)):
                effective[agent_key]["temperature"] = float(temperature)

        return effective

    def get_effective_model_request(self, user_id: UUID) -> Dict[str, Dict[str, Any]]:
        """
        Return model_request overrides only (no defaults).

        This avoids forcing per-request model re-creation when no overrides exist.
        Intended for runtime use when no per-message modelConfig is provided.
        """
        model_request: Dict[str, Dict[str, Any]] = {}
        for row in self.repository.get_all_by_user(user_id):
            agent_key = _normalize_agent_key(getattr(row, "agent_key", None))
            if not agent_key:
                continue

            provider_type = _normalize_provider(getattr(row, "provider_type", None))
            model = getattr(row, "model", None)
            temperature = getattr(row, "temperature", None)

            cfg: Dict[str, Any] = {"provider": provider_type}
            if isinstance(model, str) and model.strip():
                cfg["model"] = model.strip()
            if isinstance(temperature, (int, float)):
                cfg["temperature"] = float(temperature)

            model_request[agent_key] = cfg

        return model_request

    def patch_configs(
        self, user_id: UUID, updates: Mapping[str, Mapping[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Upsert one or more per-agent configs.

        Updates may be partial per agent (provider/model/temperature).
        """
        for raw_agent_key, raw_patch in updates.items():
            agent_key = _normalize_agent_key(raw_agent_key)
            if not agent_key:
                raise ValueError(
                    f"Invalid agent_key: {raw_agent_key}. Must be one of {list(SUPPORTED_AGENT_KEYS)}"
                )

            if not isinstance(raw_patch, Mapping):
                raise ValueError(f"Invalid config payload for {agent_key}")

            provider = _normalize_provider(
                raw_patch.get("provider_type") or raw_patch.get("provider")
            )
            model_value = raw_patch.get("model")
            temp_value = raw_patch.get("temperature")

            model = str(model_value).strip() if isinstance(model_value, str) else ""
            if not model:
                default_model = _default_agent_config(agent_key).get("model")
                model = (
                    str(default_model).strip() if isinstance(default_model, str) else ""
                )

            if provider == "openai" and not model:
                raise ValueError(
                    f"OpenAI provider requires a model for agent {agent_key}"
                )

            temperature: Optional[float] = None
            if isinstance(temp_value, (int, float)):
                temperature = float(temp_value)

            self.repository.upsert(
                user_id=user_id,
                agent_key=agent_key,
                provider_type=provider,
                model=model,
                temperature=temperature,
            )

        return self.get_effective_model_config(user_id)

    def reset_configs(self, user_id: UUID) -> int:
        return self.repository.delete_all_by_user(user_id)
