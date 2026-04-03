"""
Service for persistent per-agent model configuration and settings snapshot assembly.

Provides:
- Effective (defaults + overrides + warnings) config mapping for UI
- Override-only model_request mapping for runtime compatibility
- Composite backend-owned options snapshot for the Models page
- Validation and cleanup logic for persisted provider/model selections
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from copy import deepcopy
from typing import Any
from uuid import UUID

from app.ai.agent_config import AGENT_CONFIG
from app.core.runtime_modeling import (
    ResolvedRuntimeModelConfig,
    RuntimeFallbackConfig,
)
from app.interfaces.runtime_model_resolver_interface import IRuntimeModelResolver
from app.repositories.agent_model_config import AgentModelConfigRepository
from app.services.provider_service import ProviderService

SUPPORTED_AGENT_KEYS = ("chat", "rag", "search", "planning")
SUPPORTED_PROVIDERS = ("gemini", "openai")

logger = logging.getLogger(__name__)


def _normalize_agent_key(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    key = value.strip().lower()
    return key if key in SUPPORTED_AGENT_KEYS else None


def _normalize_provider(value: Any) -> str:
    if not isinstance(value, str):
        return "gemini"
    provider = value.strip().lower()
    return provider if provider in SUPPORTED_PROVIDERS else "gemini"


def _normalize_allow_custom_model(value: Any) -> bool:
    return bool(value)


def _coerce_temperature(value: Any, *, default: float = 1.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(default)


def _validate_temperature(value: Any, *, agent_key: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"Temperature for agent {agent_key} must be a number")

    temperature = float(value)
    if not 0.0 <= temperature <= 2.0:
        raise ValueError(f"Temperature for agent {agent_key} must be between 0.0 and 2.0")
    return temperature


def _default_agent_config(agent_key: str) -> dict[str, Any]:
    agent_cfg = AGENT_CONFIG.get(agent_key, {})
    return {
        "provider": "gemini",
        "model": agent_cfg.get("model"),
        "temperature": float(agent_cfg.get("temperature", 1.0)),
        "source": "default",
        "warnings": [],
        "key_source": None,
        "is_custom_model": False,
    }


class ModelConfigService(IRuntimeModelResolver):
    def __init__(
        self,
        repository: AgentModelConfigRepository,
        provider_service: ProviderService,
    ):
        self.repository = repository
        self.provider_service = provider_service

    def _get_provider_snapshots(
        self, user_id: UUID, provider_snapshots: Mapping[str, dict[str, Any]] | None = None
    ) -> dict[str, dict[str, Any]]:
        if provider_snapshots:
            return {
                provider: deepcopy(snapshot)
                for provider, snapshot in provider_snapshots.items()
                if provider in SUPPORTED_PROVIDERS
            }

        return {
            provider: self.provider_service.get_cached_provider_status(user_id, provider)
            for provider in SUPPORTED_PROVIDERS
        }

    def _get_catalog_models(self, provider_snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
        models = provider_snapshot.get("models", [])
        return models if isinstance(models, list) else []

    def _get_catalog_model_lookup(self, provider_snapshot: Mapping[str, Any]) -> set[str]:
        return {
            str(model.get("id") or "").strip()
            for model in self._get_catalog_models(provider_snapshot)
            if isinstance(model, dict) and str(model.get("id") or "").strip()
        }

    def _pick_catalog_model(self, provider_snapshot: Mapping[str, Any], fallback_model: str) -> str:
        for model in self._get_catalog_models(provider_snapshot):
            if isinstance(model, dict) and model.get("recommended"):
                model_id = str(model.get("id") or "").strip()
                if model_id:
                    return model_id

        for model in self._get_catalog_models(provider_snapshot):
            if isinstance(model, dict):
                model_id = str(model.get("id") or "").strip()
                if model_id:
                    return model_id

        return fallback_model

    def _select_default_provider(self, provider_snapshots: Mapping[str, dict[str, Any]]) -> str:
        gemini_snapshot = provider_snapshots.get("gemini", {})
        if gemini_snapshot.get("configured"):
            return "gemini"

        openai_snapshot = provider_snapshots.get("openai", {})
        if openai_snapshot.get("configured") and openai_snapshot.get("models"):
            return "openai"

        return "gemini"

    def _build_default_effective_config(
        self,
        agent_key: str,
        provider_snapshots: Mapping[str, dict[str, Any]],
    ) -> dict[str, Any]:
        default_config = _default_agent_config(agent_key)
        provider = self._select_default_provider(provider_snapshots)
        provider_snapshot = provider_snapshots.get(provider, {})
        default_model = str(default_config.get("model") or "").strip()

        if provider == "openai":
            default_model = self._pick_catalog_model(provider_snapshot, default_model)

        default_config.update(
            {
                "provider": provider,
                "model": default_model,
                "key_source": provider_snapshot.get("key_source", "none"),
            }
        )

        if not provider_snapshot.get("configured"):
            default_config["warnings"].append(
                f"{provider.capitalize()} is not configured. Runtime calls may fall back or fail."
            )

        if provider == "openai" and not default_model:
            default_config["warnings"].append(
                "OpenAI is configured but no model catalog is available yet. Sync models before selecting it."
            )

        return default_config

    def _build_effective_model_config(
        self,
        user_id: UUID,
        provider_snapshots: Mapping[str, dict[str, Any]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        snapshots = self._get_provider_snapshots(user_id, provider_snapshots)
        effective: dict[str, dict[str, Any]] = {
            agent_key: self._build_default_effective_config(agent_key, snapshots)
            for agent_key in SUPPORTED_AGENT_KEYS
        }

        for row in self.repository.get_all_by_user(user_id):
            agent_key = _normalize_agent_key(getattr(row, "agent_key", None))
            if not agent_key:
                continue

            provider = _normalize_provider(getattr(row, "provider_type", None))
            provider_snapshot = snapshots.get(provider, {})
            model = str(getattr(row, "model", "") or "").strip()
            allow_custom_model = bool(getattr(row, "allow_custom_model", False))
            temperature = _coerce_temperature(
                getattr(row, "temperature", None),
                default=effective[agent_key]["temperature"],
            )

            if not provider_snapshot.get("configured"):
                effective[agent_key]["warnings"].append(
                    f"Saved {provider} selection is unavailable. Showing a valid default instead."
                )
                continue

            catalog_lookup = self._get_catalog_model_lookup(provider_snapshot)
            if model and catalog_lookup and model not in catalog_lookup and not allow_custom_model:
                effective[agent_key]["warnings"].append(
                    "Saved model is no longer present in the provider catalog. Showing a valid default instead."
                )
                continue

            if provider == "openai" and not model and not catalog_lookup:
                effective[agent_key]["warnings"].append(
                    "Saved OpenAI config has no validated model. Showing a valid default instead."
                )
                continue

            effective[agent_key].update(
                {
                    "provider": provider,
                    "model": model
                    or self._pick_catalog_model(provider_snapshot, effective[agent_key]["model"]),
                    "temperature": temperature,
                    "source": "persisted",
                    "key_source": provider_snapshot.get("key_source", "none"),
                    "is_custom_model": allow_custom_model,
                }
            )

            if allow_custom_model:
                effective[agent_key]["warnings"].append(
                    "Using an explicit custom model override outside the synced catalog."
                )

        return effective

    async def _get_provider_snapshot_for_validation(
        self, user_id: UUID, provider: str
    ) -> dict[str, Any]:
        """Get provider snapshot for validation, refreshing only if stale.

        Uses cached catalog when sync_status is 'ready' to avoid hitting
        provider API rate limits on every config save.
        """
        cached_status = self.provider_service.get_cached_provider_status(user_id, provider)

        # If catalog is ready with models, use the cached version
        if (
            cached_status.get("sync_status") == "ready"
            and cached_status.get("models")
            and cached_status.get("configured")
        ):
            return cached_status

        # Otherwise, force a refresh to get fresh data
        return await self.provider_service.sync_provider_models(
            user_id=user_id,
            provider_type=provider,
            force_refresh=True,
        )

    def _resolve_patch_defaults(
        self,
        *,
        user_id: UUID,
        agent_key: str,
        existing_row: Any | None,
        effective_config: Mapping[str, dict[str, Any]],
    ) -> dict[str, Any]:
        fallback_config = effective_config.get(agent_key) or _default_agent_config(agent_key)
        return {
            "provider": (
                _normalize_provider(getattr(existing_row, "provider_type", None))
                if existing_row
                else _normalize_provider(fallback_config.get("provider"))
            ),
            "model": (
                str(getattr(existing_row, "model", "") or "").strip()
                if existing_row
                else str(fallback_config.get("model") or "").strip()
            ),
            "temperature": (
                _coerce_temperature(getattr(existing_row, "temperature", None), default=1.0)
                if existing_row
                else _coerce_temperature(fallback_config.get("temperature"), default=1.0)
            ),
            "allow_custom_model": (
                bool(getattr(existing_row, "allow_custom_model", False)) if existing_row else False
            ),
        }

    async def _validate_patch(
        self,
        *,
        user_id: UUID,
        agent_key: str,
        raw_patch: Mapping[str, Any],
        effective_config: Mapping[str, dict[str, Any]],
    ) -> dict[str, Any]:
        existing_row = self.repository.get_by_user_and_agent_key(user_id, agent_key)
        defaults = self._resolve_patch_defaults(
            user_id=user_id,
            agent_key=agent_key,
            existing_row=existing_row,
            effective_config=effective_config,
        )

        provider = _normalize_provider(raw_patch.get("provider_type") or raw_patch.get("provider"))
        if "provider" not in raw_patch and "provider_type" not in raw_patch:
            provider = defaults["provider"]

        allow_custom_model = (
            _normalize_allow_custom_model(raw_patch.get("allow_custom_model"))
            if "allow_custom_model" in raw_patch
            else defaults["allow_custom_model"]
        )

        temperature = (
            _validate_temperature(raw_patch.get("temperature"), agent_key=agent_key)
            if "temperature" in raw_patch
            else defaults["temperature"]
        )

        provider_snapshot = await self._get_provider_snapshot_for_validation(user_id, provider)
        if not provider_snapshot.get("configured"):
            raise ValueError(
                f"Provider '{provider}' is not configured for agent {agent_key}. "
                "Configure the provider before saving this selection."
            )

        catalog_lookup = self._get_catalog_model_lookup(provider_snapshot)
        model = (
            str(raw_patch.get("model") or "").strip() if "model" in raw_patch else defaults["model"]
        )
        if not model:
            model = self._pick_catalog_model(provider_snapshot, "")

        if not model:
            raise ValueError(
                f"No validated model is available for provider '{provider}' on agent {agent_key}. "
                "Sync models or use an explicit custom override."
            )

        if catalog_lookup and model in catalog_lookup:
            allow_custom_model = False
        elif not allow_custom_model:
            raise ValueError(
                f"Model '{model}' is not present in the current {provider} catalog for agent {agent_key}. "
                "Enable the explicit custom model override to save it."
            )

        return {
            "provider_type": provider,
            "model": model,
            "temperature": temperature,
            "allow_custom_model": allow_custom_model,
        }

    def _get_model_metadata(
        self, provider_snapshot: Mapping[str, Any], model_id: str
    ) -> dict[str, Any] | None:
        if not model_id:
            return None

        for model in self._get_catalog_models(provider_snapshot):
            if not isinstance(model, dict):
                continue
            if str(model.get("id") or "").strip() == model_id:
                return model
        return None

    def _build_capabilities(
        self,
        provider: str,
        model_id: str,
        provider_snapshot: Mapping[str, Any],
    ) -> dict[str, bool]:
        metadata = self._get_model_metadata(provider_snapshot, model_id)
        if metadata:
            return {
                "supports_vision": bool(metadata.get("supports_vision")),
                "supports_tool_calling": bool(metadata.get("supports_tool_calling")),
                "supports_streaming": bool(metadata.get("supports_streaming", True)),
                "supports_reasoning": bool(metadata.get("supports_reasoning")),
            }

        model_lower = model_id.lower()
        if provider == "openai":
            return {
                "supports_vision": any(
                    token in model_lower
                    for token in ("gpt-4o", "gpt-4.1", "gpt-5", "o1", "o3", "o4")
                ),
                "supports_tool_calling": True,
                "supports_streaming": True,
                "supports_reasoning": model_lower.startswith(("o1", "o3", "o4", "gpt-5")),
            }

        return {
            "supports_vision": True,
            "supports_tool_calling": True,
            "supports_streaming": True,
            "supports_reasoning": any(token in model_lower for token in ("2.5", "3", "pro")),
        }

    def _build_runtime_fallback_candidate(
        self,
        *,
        user_id: UUID,
        agent_key: str,
        provider_snapshots: Mapping[str, dict[str, Any]],
        effective_config: Mapping[str, dict[str, Any]],
        current_provider: str,
        credentials_cache: dict[str, dict[str, Any]] | None = None,
    ) -> RuntimeFallbackConfig | None:
        default_model = str(AGENT_CONFIG.get(agent_key, {}).get("model") or "").strip()
        preferred_providers: list[str] = []

        base_provider = str((effective_config.get(agent_key) or {}).get("provider") or "").strip()
        if base_provider and base_provider != current_provider:
            preferred_providers.append(base_provider)

        for provider in SUPPORTED_PROVIDERS:
            if provider != current_provider and provider not in preferred_providers:
                preferred_providers.append(provider)

        for provider in preferred_providers:
            snapshot = provider_snapshots.get(provider, {})

            # Use cached credentials if available, otherwise fetch and cache
            if credentials_cache is not None and provider in credentials_cache:
                credentials = credentials_cache[provider]
            else:
                credentials = self.provider_service.resolve_provider_credentials(user_id, provider)
                if credentials_cache is not None:
                    credentials_cache[provider] = credentials

            api_key = credentials.get("api_key")
            if (
                not snapshot.get("configured")
                or not isinstance(api_key, str)
                or not api_key.strip()
            ):
                continue

            model = self._pick_catalog_model(
                snapshot,
                default_model if provider == "gemini" else "",
            )
            if not model:
                continue

            return RuntimeFallbackConfig(
                provider=provider,
                model=model,
                temperature=float((effective_config.get(agent_key) or {}).get("temperature", 1.0)),
                api_key=api_key.strip(),
                key_source=str(credentials.get("key_source") or "none"),
            )

        return None

    def resolve_runtime_config(
        self,
        user_id: UUID | None,
        agent_key: str,
        request_override: Mapping[str, Any] | None = None,
    ) -> ResolvedRuntimeModelConfig:
        normalized_agent_key = _normalize_agent_key(agent_key)
        if not normalized_agent_key:
            raise ValueError(f"Unsupported agent_key for runtime resolution: {agent_key}")

        default_temperature = float(
            AGENT_CONFIG.get(normalized_agent_key, {}).get("temperature", 1.0)
        )
        default_model = str(AGENT_CONFIG.get(normalized_agent_key, {}).get("model") or "").strip()

        if user_id is None:
            logger.warning(
                "Runtime config resolved without user context for agent=%s; using default Gemini configuration",
                normalized_agent_key,
            )
            return ResolvedRuntimeModelConfig(
                agent_key=normalized_agent_key,
                provider="gemini",
                model=default_model,
                temperature=default_temperature,
                api_key=None,
                key_source="none",
                source="default",
                warnings=["User context is missing; using default Gemini runtime configuration."],
                capabilities=self._build_capabilities("gemini", default_model, {}),
            )

        provider_snapshots = self._get_provider_snapshots(user_id)
        effective_config = self._build_effective_model_config(user_id, provider_snapshots)
        base_config = deepcopy(
            effective_config.get(normalized_agent_key)
            or self._build_default_effective_config(normalized_agent_key, provider_snapshots)
        )

        # Credentials cache to avoid repeated database lookups within this resolution
        credentials_cache: dict[str, dict[str, Any]] = {}

        provider = _normalize_provider(base_config.get("provider"))
        model = str(base_config.get("model") or "").strip()
        temperature = _coerce_temperature(
            base_config.get("temperature"), default=default_temperature
        )
        source = str(base_config.get("source") or "default")
        warnings = list(base_config.get("warnings") or [])
        is_custom_model = bool(base_config.get("is_custom_model"))
        provider_fallback: dict[str, Any] | None = None

        if request_override and isinstance(request_override, Mapping):
            requested_provider = _normalize_provider(
                request_override.get("provider_type") or request_override.get("provider")
            )
            requested_snapshot = provider_snapshots.get(requested_provider, {})

            # Cache the credentials lookup
            requested_credentials = self.provider_service.resolve_provider_credentials(
                user_id, requested_provider
            )
            credentials_cache[requested_provider] = requested_credentials
            requested_model = str(request_override.get("model") or "").strip()
            requested_temperature = _coerce_temperature(
                request_override.get("temperature"),
                default=temperature,
            )
            requested_allow_custom = _normalize_allow_custom_model(
                request_override.get("allow_custom_model")
            )

            if not requested_snapshot.get("configured") or not requested_credentials.get("api_key"):
                warnings.append(
                    f"Requested provider '{requested_provider}' is unavailable. Falling back to {provider}."
                )
                if requested_provider != provider:
                    provider_fallback = {
                        "from": requested_provider,
                        "to": provider,
                        "reason": "provider_not_configured",
                    }
            else:
                catalog_lookup = self._get_catalog_model_lookup(requested_snapshot)
                if not requested_model:
                    requested_model = self._pick_catalog_model(
                        requested_snapshot,
                        default_model if requested_provider == "gemini" else "",
                    )

                if (
                    requested_model
                    and catalog_lookup
                    and requested_model not in catalog_lookup
                    and not requested_allow_custom
                ):
                    warnings.append(
                        f"Requested model '{requested_model}' is not in the current {requested_provider} catalog. "
                        f"Falling back to {provider}:{model}."
                    )
                else:
                    provider = requested_provider
                    model = requested_model or model
                    temperature = requested_temperature
                    source = "request"
                    is_custom_model = bool(
                        requested_allow_custom
                        and (not catalog_lookup or requested_model not in catalog_lookup)
                    )
                    if is_custom_model:
                        warnings.append(
                            "Runtime is using an explicit custom model override outside the synced catalog."
                        )

        # Use cached credentials if available, otherwise fetch and cache
        if provider in credentials_cache:
            credentials = credentials_cache[provider]
        else:
            credentials = self.provider_service.resolve_provider_credentials(user_id, provider)
            credentials_cache[provider] = credentials

        api_key = credentials.get("api_key")
        key_source = str(credentials.get("key_source") or "none")

        # Compute fallback candidate once (will be reused if needed)
        fallback_config = self._build_runtime_fallback_candidate(
            user_id=user_id,
            agent_key=normalized_agent_key,
            provider_snapshots=provider_snapshots,
            effective_config=effective_config,
            current_provider=provider,
            credentials_cache=credentials_cache,
        )

        if not isinstance(api_key, str) or not api_key.strip():
            if fallback_config:
                warnings.append(
                    f"Selected provider '{provider}' is unavailable at runtime. Falling back to {fallback_config.provider}."
                )
                provider_fallback = {
                    "from": provider,
                    "to": fallback_config.provider,
                    "reason": "provider_not_configured",
                }
                provider = fallback_config.provider
                model = fallback_config.model
                temperature = fallback_config.temperature
                api_key = fallback_config.api_key
                key_source = fallback_config.key_source
                source = "fallback"

                # Recompute fallback candidate for the new provider (for downstream use)
                fallback_config = self._build_runtime_fallback_candidate(
                    user_id=user_id,
                    agent_key=normalized_agent_key,
                    provider_snapshots=provider_snapshots,
                    effective_config=effective_config,
                    current_provider=provider,
                    credentials_cache=credentials_cache,
                )

        capabilities = self._build_capabilities(
            provider,
            model,
            provider_snapshots.get(provider, {}),
        )

        if provider_fallback:
            logger.warning(
                "Runtime config fallback for user=%s agent=%s: %s -> %s (%s)",
                user_id,
                normalized_agent_key,
                provider_fallback.get("from"),
                provider_fallback.get("to"),
                provider_fallback.get("reason"),
            )

        logger.info(
            "Resolved runtime config for user=%s agent=%s provider=%s model=%s source=%s key_source=%s custom=%s warnings=%d",
            user_id,
            normalized_agent_key,
            provider,
            model,
            source,
            key_source,
            is_custom_model,
            len(warnings),
        )

        return ResolvedRuntimeModelConfig(
            agent_key=normalized_agent_key,
            provider=provider,
            model=model,
            temperature=temperature,
            api_key=api_key.strip() if isinstance(api_key, str) and api_key.strip() else None,
            key_source=key_source,
            source=source,
            warnings=warnings,
            is_custom_model=is_custom_model,
            capabilities=capabilities,
            provider_fallback=provider_fallback,
            fallback_config=fallback_config,
        )

    def get_effective_model_config(self, user_id: UUID) -> dict[str, dict[str, Any]]:
        return self._build_effective_model_config(user_id)

    async def get_model_config_options(self, user_id: UUID) -> dict[str, Any]:
        provider_results = await asyncio.gather(
            *(
                self.provider_service.get_provider_status(
                    user_id=user_id,
                    provider_type=provider_type,
                    refresh_if_missing=True,
                )
                for provider_type in SUPPORTED_PROVIDERS
            )
        )
        provider_snapshots = {
            str(snapshot.get("provider_type") or ""): snapshot for snapshot in provider_results
        }

        return {
            "providers": provider_results,
            "agent_config": self._build_effective_model_config(user_id, provider_snapshots),
        }

    async def patch_configs(
        self, user_id: UUID, updates: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        effective_config = self.get_effective_model_config(user_id)

        for raw_agent_key, raw_patch in updates.items():
            agent_key = _normalize_agent_key(raw_agent_key)
            if not agent_key:
                raise ValueError(
                    f"Invalid agent_key: {raw_agent_key}. Must be one of {list(SUPPORTED_AGENT_KEYS)}"
                )

            if not isinstance(raw_patch, Mapping):
                raise ValueError(f"Invalid config payload for {agent_key}")

            validated = await self._validate_patch(
                user_id=user_id,
                agent_key=agent_key,
                raw_patch=raw_patch,
                effective_config=effective_config,
            )

            self.repository.upsert(
                user_id=user_id,
                agent_key=agent_key,
                provider_type=validated["provider_type"],
                model=validated["model"],
                allow_custom_model=validated["allow_custom_model"],
                temperature=validated["temperature"],
            )

            effective_config[agent_key] = {
                "provider": validated["provider_type"],
                "model": validated["model"],
                "temperature": validated["temperature"],
                "source": "persisted",
                "warnings": (
                    ["Using an explicit custom model override outside the synced catalog."]
                    if validated["allow_custom_model"]
                    else []
                ),
                "key_source": self.provider_service.get_cached_provider_status(
                    user_id, validated["provider_type"]
                ).get("key_source", "none"),
                "is_custom_model": validated["allow_custom_model"],
            }

        return self.get_effective_model_config(user_id)

    def cleanup_configs_for_provider(self, user_id: UUID, provider_type: str) -> list[str]:
        provider = _normalize_provider(provider_type)
        return self.repository.delete_by_user_and_provider_type(user_id, provider)

    def reset_configs(self, user_id: UUID) -> int:
        return self.repository.delete_all_by_user(user_id)
