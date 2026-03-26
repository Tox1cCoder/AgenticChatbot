"""
Provider Service for managing AI provider API keys, catalog sync, and status snapshots.

Handles:
- Encryption/decryption of stored API keys
- Provider credential resolution, including Gemini env fallback
- Model catalog sync and caching in provider_metadata
- Normalized provider status used by the settings UI and runtime validation
"""

from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import settings
from app.models.model_provider import ModelProvider
from app.repositories.model_provider import ModelProviderRepository

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ("gemini", "openai", "anthropic")
CATALOG_METADATA_KEY = "catalog"
IN_SCOPE_PROVIDER_TYPES = ("gemini", "openai")
OPENAI_PREFERRED_MODEL_ORDER = (
    "gpt-5-mini",
    "gpt-5",
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4o-mini",
    "gpt-4o",
    "o4-mini",
    "o3-mini",
)
GEMINI_PREFERRED_MODEL_ORDER = (
    "gemini-3-flash-preview",
    "gemini-2.5-flash-latest",
    "gemini-2.5-pro",
    "gemini-3.1-pro-preview",
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProviderService:
    """
    Service for managing AI provider configurations with encrypted API keys.

    Provides methods for:
    - Storing/retrieving encrypted API keys
    - Resolving provider availability
    - Fetching and caching normalized provider model catalogs
    """

    def __init__(self, provider_repository: ModelProviderRepository):
        self.repository = provider_repository

        if not settings.model_encryption_key:
            raise ValueError(
                "MODEL_ENCRYPTION_KEY must be configured in environment. "
                "Generate one with: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
            )

        try:
            self.cipher = Fernet(settings.model_encryption_key.encode())
        except Exception as e:
            raise ValueError(
                "Invalid MODEL_ENCRYPTION_KEY format. Must be 32 url-safe "
                f"base64-encoded bytes. Error: {e}"
            ) from e

    def _normalize_provider_type(self, provider_type: str) -> str:
        normalized = str(provider_type or "").strip().lower()
        if normalized not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Invalid provider_type: {provider_type}. Must be one of {list(SUPPORTED_PROVIDERS)}"
            )
        return normalized

    def _encrypt_key(self, api_key: str) -> str:
        try:
            encrypted = self.cipher.encrypt(api_key.encode())
            return encrypted.decode()
        except Exception as e:
            logger.error("Failed to encrypt API key: %s", e)
            raise ValueError("Failed to encrypt API key") from e

    def _decrypt_key(self, encrypted_key: str) -> str:
        try:
            decrypted = self.cipher.decrypt(encrypted_key.encode())
            return decrypted.decode()
        except InvalidToken as e:
            logger.error("Failed to decrypt API key: invalid encryption key or corrupted data")
            raise ValueError("Failed to decrypt API key: Invalid encryption") from e
        except Exception as e:
            logger.error("Failed to decrypt API key: %s", e)
            raise ValueError("Failed to decrypt API key") from e

    def _get_public_provider_metadata(
        self, provider_metadata: dict[str, Any] | None
    ) -> dict[str, Any]:
        metadata = deepcopy(provider_metadata or {})
        metadata.pop(CATALOG_METADATA_KEY, None)
        return metadata

    def _normalize_catalog_metadata(
        self, provider_metadata: dict[str, Any] | None
    ) -> dict[str, Any]:
        raw_catalog = (provider_metadata or {}).get(CATALOG_METADATA_KEY, {})
        if not isinstance(raw_catalog, dict):
            raw_catalog = {}

        raw_models = raw_catalog.get("models", [])
        models = raw_models if isinstance(raw_models, list) else []

        sync_status = str(raw_catalog.get("sync_status") or "").strip().lower() or "never_synced"
        if sync_status not in {"never_synced", "ready", "error", "stale", "not_configured"}:
            sync_status = "never_synced"

        last_synced_at = raw_catalog.get("last_synced_at")
        if not isinstance(last_synced_at, str) or not last_synced_at.strip():
            last_synced_at = None

        sync_error = raw_catalog.get("sync_error")
        if not isinstance(sync_error, str) or not sync_error.strip():
            sync_error = None

        catalog_version = raw_catalog.get("catalog_version")
        try:
            catalog_version = int(catalog_version)
        except Exception:
            catalog_version = 1

        return {
            "models": models,
            "last_synced_at": last_synced_at,
            "sync_status": sync_status,
            "sync_error": sync_error,
            "catalog_version": catalog_version,
        }

    def _build_provider_metadata_with_catalog(
        self,
        provider_metadata: dict[str, Any] | None,
        *,
        models: list[dict[str, Any]] | None = None,
        sync_status: str | None = None,
        sync_error: str | None = None,
        last_synced_at: str | None = None,
    ) -> dict[str, Any]:
        merged_metadata = deepcopy(provider_metadata or {})
        catalog = self._normalize_catalog_metadata(merged_metadata)

        if models is not None:
            catalog["models"] = models
            catalog["catalog_version"] = int(catalog.get("catalog_version") or 1) + 1
        if sync_status is not None:
            catalog["sync_status"] = sync_status
        if sync_error is not None or sync_status == "ready":
            catalog["sync_error"] = sync_error
        if last_synced_at is not None:
            catalog["last_synced_at"] = last_synced_at

        merged_metadata[CATALOG_METADATA_KEY] = catalog
        return merged_metadata

    def _merge_provider_metadata_for_upsert(
        self,
        existing_metadata: dict[str, Any] | None,
        incoming_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        merged_metadata = deepcopy(existing_metadata or {})
        if incoming_metadata:
            for key, value in incoming_metadata.items():
                merged_metadata[key] = value

        catalog = self._normalize_catalog_metadata(merged_metadata)
        if catalog.get("models") or catalog.get("last_synced_at"):
            catalog["sync_status"] = "stale"
            catalog["sync_error"] = None
        merged_metadata[CATALOG_METADATA_KEY] = catalog
        return merged_metadata

    def _normalize_env_gemini_api_key(self) -> str | None:
        api_key = settings.gemini_api_key
        if not api_key:
            return None
        if api_key.startswith("GEMINI_API_KEY="):
            api_key = api_key.split("=", 1)[1].strip()
        return api_key or None

    def add_provider(
        self,
        user_id: UUID,
        provider_type: str,
        api_key: str,
        is_default: bool = False,
        provider_metadata: dict[str, Any] | None = None,
    ) -> ModelProvider:
        provider_type = self._normalize_provider_type(provider_type)

        existing = self.repository.get_by_user_and_type(user_id, provider_type)
        merged_metadata = self._merge_provider_metadata_for_upsert(
            getattr(existing, "provider_metadata", None),
            provider_metadata,
        )

        encrypted_key = self._encrypt_key(api_key)
        provider = self.repository.upsert(
            user_id=user_id,
            provider_type=provider_type,
            api_key_encrypted=encrypted_key,
            is_default=is_default,
            provider_metadata=merged_metadata,
        )

        logger.info(
            "Added/updated %s provider for user %s (id=%s)", provider_type, user_id, provider.id
        )
        return provider

    def get_provider_config(
        self, user_id: UUID, provider_type: str, decrypt: bool = False
    ) -> dict[str, Any] | None:
        provider_type = self._normalize_provider_type(provider_type)
        provider = self.repository.get_by_user_and_type(user_id, provider_type)
        if not provider:
            return None

        config = {
            "id": str(provider.id),
            "provider_type": provider.provider_type,
            "is_default": provider.is_default,
            "created_at": provider.created_at.isoformat(),
            "provider_metadata": self._get_public_provider_metadata(provider.provider_metadata),
        }

        if decrypt:
            try:
                config["api_key"] = self._decrypt_key(provider.api_key_encrypted)
            except ValueError as e:
                logger.error("Failed to decrypt API key for provider %s: %s", provider.id, e)
                config["api_key"] = None

        return config

    def get_decrypted_api_key(self, user_id: UUID, provider_type: str) -> str | None:
        provider_type = self._normalize_provider_type(provider_type)
        provider = self.repository.get_by_user_and_type(user_id, provider_type)
        if not provider:
            return None

        try:
            return self._decrypt_key(provider.api_key_encrypted)
        except ValueError:
            return None

    def get_all_providers(
        self, user_id: UUID, include_encrypted: bool = False
    ) -> list[dict[str, Any]]:
        providers = self.repository.get_all_by_user(user_id)

        result = []
        for provider in providers:
            config = {
                "id": str(provider.id),
                "provider_type": provider.provider_type,
                "is_default": provider.is_default,
                "created_at": provider.created_at.isoformat(),
                "provider_metadata": self._get_public_provider_metadata(provider.provider_metadata),
            }

            if include_encrypted:
                encrypted = provider.api_key_encrypted
                config["key_preview"] = f"...{encrypted[-4:]}" if len(encrypted) >= 4 else "***"

            result.append(config)

        return result

    def delete_provider(self, user_id: UUID, provider_type: str) -> bool:
        provider_type = self._normalize_provider_type(provider_type)
        deleted = self.repository.delete_by_user_and_type(user_id, provider_type)
        if deleted:
            logger.info("Deleted %s provider for user %s", provider_type, user_id)
        return deleted

    def resolve_provider_credentials(self, user_id: UUID, provider_type: str) -> dict[str, Any]:
        provider_type = self._normalize_provider_type(provider_type)
        provider = self.repository.get_by_user_and_type(user_id, provider_type)

        if provider:
            try:
                api_key = self._decrypt_key(provider.api_key_encrypted)
            except ValueError:
                return {
                    "configured": False,
                    "api_key": None,
                    "key_source": "db",
                    "provider": provider,
                    "provider_metadata": provider.provider_metadata or {},
                    "config_error": "Stored API key could not be decrypted",
                }

            return {
                "configured": bool(api_key),
                "api_key": api_key,
                "key_source": "db",
                "provider": provider,
                "provider_metadata": provider.provider_metadata or {},
                "config_error": None,
            }

        if provider_type == "gemini":
            env_api_key = self._normalize_env_gemini_api_key()
            if env_api_key:
                return {
                    "configured": True,
                    "api_key": env_api_key,
                    "key_source": "env",
                    "provider": None,
                    "provider_metadata": {},
                    "config_error": None,
                }

        return {
            "configured": False,
            "api_key": None,
            "key_source": "none",
            "provider": None,
            "provider_metadata": {},
            "config_error": None,
        }

    def get_cached_provider_models(self, user_id: UUID, provider_type: str) -> list[dict[str, Any]]:
        provider_type = self._normalize_provider_type(provider_type)
        credentials = self.resolve_provider_credentials(user_id, provider_type)
        catalog = self._normalize_catalog_metadata(credentials.get("provider_metadata"))
        models = catalog.get("models", [])
        return deepcopy(models) if isinstance(models, list) else []

    def get_cached_provider_status(self, user_id: UUID, provider_type: str) -> dict[str, Any]:
        provider_type = self._normalize_provider_type(provider_type)
        credentials = self.resolve_provider_credentials(user_id, provider_type)
        catalog = self._normalize_catalog_metadata(credentials.get("provider_metadata"))

        configured = bool(credentials.get("configured"))
        key_source = str(credentials.get("key_source") or "none")
        config_error = credentials.get("config_error")

        sync_status = str(catalog.get("sync_status") or "never_synced")
        if not configured:
            sync_status = "error" if config_error else "not_configured"

        return {
            "provider_type": provider_type,
            "configured": configured,
            "key_source": key_source,
            "sync_status": sync_status,
            "last_synced_at": catalog.get("last_synced_at"),
            "sync_error": config_error or catalog.get("sync_error"),
            "models": deepcopy(catalog.get("models", [])),
            "warnings": [],
        }

    async def get_provider_status(
        self,
        user_id: UUID,
        provider_type: str,
        *,
        refresh_if_missing: bool = False,
    ) -> dict[str, Any]:
        provider_type = self._normalize_provider_type(provider_type)
        cached_status = self.get_cached_provider_status(user_id, provider_type)

        should_refresh = (
            refresh_if_missing
            and provider_type in IN_SCOPE_PROVIDER_TYPES
            and cached_status.get("configured")
            and not cached_status.get("models")
        )
        if should_refresh:
            try:
                return await self.sync_provider_models(
                    user_id=user_id,
                    provider_type=provider_type,
                    force_refresh=True,
                )
            except Exception as exc:
                logger.warning(
                    "Provider catalog refresh failed for %s user=%s: %s",
                    provider_type,
                    user_id,
                    exc,
                )
                cached_status["sync_status"] = "error"
                cached_status["sync_error"] = str(exc)

        return cached_status

    async def sync_provider_models(
        self, user_id: UUID, provider_type: str, force_refresh: bool = False
    ) -> dict[str, Any]:
        provider_type = self._normalize_provider_type(provider_type)

        if provider_type not in IN_SCOPE_PROVIDER_TYPES:
            raise ValueError(f"Model sync is not implemented for provider: {provider_type}")

        credentials = self.resolve_provider_credentials(user_id, provider_type)
        provider = credentials.get("provider")
        configured = bool(credentials.get("configured"))
        api_key = credentials.get("api_key")
        key_source = str(credentials.get("key_source") or "none")

        if not configured or not api_key:
            sync_error = credentials.get("config_error") or "Provider not configured"
            if provider:
                provider_metadata = self._build_provider_metadata_with_catalog(
                    provider.provider_metadata,
                    sync_status="not_configured",
                    sync_error=sync_error,
                )
                self.repository.update(provider.id, provider_metadata=provider_metadata)

            return {
                "provider_type": provider_type,
                "configured": False,
                "key_source": key_source,
                "sync_status": "not_configured",
                "last_synced_at": None,
                "sync_error": sync_error,
                "models": [],
                "warnings": [],
            }

        if not force_refresh:
            cached_status = self.get_cached_provider_status(user_id, provider_type)
            if cached_status.get("models") and cached_status.get("sync_status") == "ready":
                return cached_status

        try:
            if provider_type == "openai":
                models = await self._fetch_openai_models(api_key)
            else:
                models = await self._fetch_gemini_models(api_key)

            last_synced_at = _utcnow_iso()
            if provider:
                provider_metadata = self._build_provider_metadata_with_catalog(
                    provider.provider_metadata,
                    models=models,
                    sync_status="ready",
                    sync_error=None,
                    last_synced_at=last_synced_at,
                )
                self.repository.update(provider.id, provider_metadata=provider_metadata)

            logger.info(
                "Synced %d %s models for user %s (key_source=%s)",
                len(models),
                provider_type,
                user_id,
                key_source,
            )
            return {
                "provider_type": provider_type,
                "configured": True,
                "key_source": key_source,
                "sync_status": "ready",
                "last_synced_at": last_synced_at,
                "sync_error": None,
                "models": models,
                "warnings": [],
            }
        except Exception as exc:
            sync_error = str(exc)
            if provider:
                provider_metadata = self._build_provider_metadata_with_catalog(
                    provider.provider_metadata,
                    sync_status="error",
                    sync_error=sync_error,
                )
                self.repository.update(provider.id, provider_metadata=provider_metadata)
            logger.error("Provider model sync failed for %s: %s", provider_type, exc)
            raise ValueError(sync_error) from exc

    async def validate_provider(self, user_id: UUID, provider_type: str) -> dict[str, Any]:
        provider_type = self._normalize_provider_type(provider_type)

        if provider_type in IN_SCOPE_PROVIDER_TYPES:
            try:
                status = await self.sync_provider_models(
                    user_id=user_id,
                    provider_type=provider_type,
                    force_refresh=True,
                )
                return {
                    "valid": True,
                    "models": status.get("models", []),
                    "message": f"{provider_type} provider validated",
                }
            except Exception as e:
                logger.error("Provider validation failed for %s: %s", provider_type, e)
                return {"valid": False, "error": str(e)}

        credentials = self.resolve_provider_credentials(user_id, provider_type)
        if credentials.get("configured"):
            return {"valid": True, "message": f"{provider_type} provider configured"}
        return {"valid": False, "error": "Provider not configured"}

    def _should_include_openai_model(self, model_id: str) -> bool:
        model_id = model_id.strip().lower()
        if not model_id:
            return False

        include_prefixes = ("gpt-", "chatgpt-", "o1", "o3", "o4")
        exclude_fragments = (
            "audio",
            "transcribe",
            "tts",
            "embedding",
            "image",
            "moderation",
            "realtime",
            "search",
            "whisper",
            "instruct",
        )

        return model_id.startswith(include_prefixes) and not any(
            fragment in model_id for fragment in exclude_fragments
        )

    def _normalize_openai_model(self, model_id: str) -> dict[str, Any]:
        model_lower = model_id.lower()
        supports_reasoning = model_lower.startswith(("o1", "o3", "o4", "gpt-5"))
        supports_vision = any(
            token in model_lower for token in ("gpt-4o", "gpt-4.1", "gpt-5", "o1", "o3", "o4")
        )

        return {
            "id": model_id,
            "display_name": model_id,
            "provider_type": "openai",
            "supports_vision": supports_vision,
            "supports_tool_calling": True,
            "supports_streaming": True,
            "supports_reasoning": supports_reasoning,
            "recommended": False,
        }

    def _should_include_gemini_model(self, model_id: str, supported_actions: list[str]) -> bool:
        model_lower = model_id.lower()
        if not model_lower.startswith("gemini"):
            return False

        excluded_fragments = ("embedding", "aqa", "imagen", "image", "veo", "tts", "transcribe")
        if any(fragment in model_lower for fragment in excluded_fragments):
            return False

        return not (supported_actions and "generateContent" not in supported_actions)

    def _normalize_gemini_model(
        self,
        model_id: str,
        display_name: str | None,
        supported_actions: list[str],
        thinking_metadata: Any | None,
    ) -> dict[str, Any]:
        model_lower = model_id.lower()
        supports_reasoning = bool(thinking_metadata) or any(
            token in model_lower for token in ("2.5", "3", "pro")
        )

        return {
            "id": model_id,
            "display_name": display_name or model_id,
            "provider_type": "gemini",
            "supports_vision": True,
            "supports_tool_calling": True,
            "supports_streaming": True,
            "supports_reasoning": supports_reasoning,
            "recommended": False,
        }

    def _apply_recommended_model_flags(
        self, provider_type: str, models: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        preferred_order = (
            OPENAI_PREFERRED_MODEL_ORDER
            if provider_type == "openai"
            else GEMINI_PREFERRED_MODEL_ORDER
        )
        model_map = {str(model.get("id") or ""): model for model in models}

        recommended_id = None
        for candidate in preferred_order:
            if candidate in model_map:
                recommended_id = candidate
                break
        if not recommended_id and models:
            recommended_id = str(models[0].get("id") or "")

        for model in models:
            model["recommended"] = bool(recommended_id and model.get("id") == recommended_id)

        return sorted(
            models,
            key=lambda item: (
                0 if item.get("recommended") else 1,
                str(item.get("display_name") or item.get("id") or "").lower(),
            ),
        )

    async def _fetch_openai_models(self, api_key: str) -> list[dict[str, Any]]:
        try:
            import openai

            async_client_cls = getattr(openai, "AsyncOpenAI", None)
            if async_client_cls is not None:
                client = async_client_cls(api_key=api_key)
                models_response = await client.models.list()
            else:
                client = openai.OpenAI(api_key=api_key)
                models_response = await asyncio.to_thread(client.models.list)

            normalized_models: list[dict[str, Any]] = []
            for model in getattr(models_response, "data", None) or []:
                model_id = getattr(model, "id", None)
                if not isinstance(model_id, str) or not self._should_include_openai_model(model_id):
                    continue
                normalized_models.append(self._normalize_openai_model(model_id.strip()))

            normalized_models = self._apply_recommended_model_flags("openai", normalized_models)
            logger.info("Fetched %d normalized OpenAI models", len(normalized_models))
            return normalized_models
        except Exception as e:
            logger.error("Failed to fetch OpenAI models: %s", e)
            raise ValueError(f"Failed to fetch OpenAI models: {e}") from e

    async def _fetch_gemini_models(self, api_key: str) -> list[dict[str, Any]]:
        try:
            from google import genai

            client = genai.Client(api_key=api_key)
            pager = client.models.list(config={"page_size": 100, "query_base": True})

            normalized_models: list[dict[str, Any]] = []
            for model in pager:
                model_name = getattr(model, "name", None)
                if not isinstance(model_name, str) or not model_name.strip():
                    continue

                model_id = model_name.rsplit("/", 1)[-1].strip()
                supported_actions = list(getattr(model, "supported_actions", None) or [])
                if not self._should_include_gemini_model(model_id, supported_actions):
                    continue

                normalized_models.append(
                    self._normalize_gemini_model(
                        model_id=model_id,
                        display_name=getattr(model, "display_name", None),
                        supported_actions=supported_actions,
                        thinking_metadata=getattr(model, "thinking", None),
                    )
                )

            normalized_models = self._apply_recommended_model_flags("gemini", normalized_models)
            logger.info("Fetched %d normalized Gemini models", len(normalized_models))
            return normalized_models
        except Exception as e:
            logger.error("Failed to fetch Gemini models: %s", e)
            raise ValueError(f"Failed to fetch Gemini models: {e}") from e
