"""
Provider Service for managing AI provider API keys and configurations.

Handles encryption/decryption of API keys, provider validation, and model listing
from various AI providers.
"""

import asyncio
import logging
from typing import Any
from uuid import UUID

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import settings
from app.models.model_provider import ModelProvider
from app.repositories.model_provider import ModelProviderRepository

logger = logging.getLogger(__name__)


class ProviderService:
    """
    Service for managing AI provider configurations with encrypted API keys.

    Provides methods for:
    - Storing/retrieving encrypted API keys
    - Validating provider credentials
    - Fetching available models from provider APIs
    """

    def __init__(self, provider_repository: ModelProviderRepository):
        """
        Initialize provider service.

        Args:
            provider_repository: Repository for provider database operations

        Raises:
            ValueError: If MODEL_ENCRYPTION_KEY is not configured
        """
        self.repository = provider_repository

        # Initialize encryption cipher
        if not settings.model_encryption_key:
            raise ValueError(
                "MODEL_ENCRYPTION_KEY must be configured in environment. "
                "Generate one with: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
            )

        try:
            self.cipher = Fernet(settings.model_encryption_key.encode())
        except Exception as e:
            raise ValueError(
                f"Invalid MODEL_ENCRYPTION_KEY format. Must be 32 url-safe base64-encoded bytes. Error: {e}"
            )

    def _encrypt_key(self, api_key: str) -> str:
        """
        Encrypt an API key using Fernet symmetric encryption.

        Args:
            api_key: Plain text API key

        Returns:
            str: Encrypted API key (base64 encoded)
        """
        try:
            encrypted = self.cipher.encrypt(api_key.encode())
            return encrypted.decode()
        except Exception as e:
            logger.error(f"Failed to encrypt API key: {e}")
            raise ValueError("Failed to encrypt API key")

    def _decrypt_key(self, encrypted_key: str) -> str:
        """
        Decrypt an encrypted API key.

        Args:
            encrypted_key: Encrypted API key (base64 encoded)

        Returns:
            str: Decrypted plain text API key

        Raises:
            ValueError: If decryption fails (invalid key or corruption)
        """
        try:
            decrypted = self.cipher.decrypt(encrypted_key.encode())
            return decrypted.decode()
        except InvalidToken:
            logger.error("Failed to decrypt API key: Invalid encryption key or corrupted data")
            raise ValueError("Failed to decrypt API key: Invalid encryption")
        except Exception as e:
            logger.error(f"Failed to decrypt API key: {e}")
            raise ValueError("Failed to decrypt API key")

    def add_provider(
        self,
        user_id: UUID,
        provider_type: str,
        api_key: str,
        is_default: bool = False,
        provider_metadata: dict[str, Any] | None = None,
    ) -> ModelProvider:
        """
        Add or update a provider for a user.

        Args:
            user_id: User UUID
            provider_type: Provider type ('gemini', 'openai', 'anthropic')
            api_key: Plain text API key
            is_default: Whether this should be the default provider
            provider_metadata: Optional provider-specific metadata

        Returns:
            ModelProvider: Created or updated provider

        Raises:
            ValueError: If provider_type is invalid or encryption fails
        """
        # Validate provider type
        valid_providers = ["gemini", "openai", "anthropic"]
        if provider_type.lower() not in valid_providers:
            raise ValueError(
                f"Invalid provider_type: {provider_type}. Must be one of {valid_providers}"
            )

        # Encrypt API key
        encrypted_key = self._encrypt_key(api_key)

        # Upsert provider
        provider = self.repository.upsert(
            user_id=user_id,
            provider_type=provider_type.lower(),
            api_key_encrypted=encrypted_key,
            is_default=is_default,
            provider_metadata=provider_metadata,
        )

        logger.info(f"Added/updated {provider_type} provider for user {user_id} (id={provider.id})")
        return provider

    def get_provider_config(
        self, user_id: UUID, provider_type: str, decrypt: bool = False
    ) -> dict[str, Any] | None:
        """
        Get provider configuration for a user.

        Args:
            user_id: User UUID
            provider_type: Provider type
            decrypt: If True, include decrypted API key (use with caution!)

        Returns:
            Optional[Dict]: Provider config dict or None if not found
        """
        provider = self.repository.get_by_user_and_type(user_id, provider_type)
        if not provider:
            return None

        config = {
            "id": str(provider.id),
            "provider_type": provider.provider_type,
            "is_default": provider.is_default,
            "created_at": provider.created_at.isoformat(),
            "provider_metadata": provider.provider_metadata or {},
        }

        if decrypt:
            # CAUTION: Only decrypt when absolutely necessary
            try:
                config["api_key"] = self._decrypt_key(provider.api_key_encrypted)
            except ValueError as e:
                logger.error(f"Failed to decrypt API key for provider {provider.id}: {e}")
                config["api_key"] = None

        return config

    def get_decrypted_api_key(self, user_id: UUID, provider_type: str) -> str | None:
        """
        Get decrypted API key for a provider.

        Args:
            user_id: User UUID
            provider_type: Provider type

        Returns:
            Optional[str]: Decrypted API key or None if not found
        """
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
        """
        Get all provider configurations for a user.

        Args:
            user_id: User UUID
            include_encrypted: If True, include last 4 chars of encrypted key

        Returns:
            List[Dict]: List of provider configs
        """
        providers = self.repository.get_all_by_user(user_id)

        result = []
        for provider in providers:
            config = {
                "id": str(provider.id),
                "provider_type": provider.provider_type,
                "is_default": provider.is_default,
                "created_at": provider.created_at.isoformat(),
                "provider_metadata": provider.provider_metadata or {},
            }

            if include_encrypted:
                # Show only last 4 chars for verification
                encrypted = provider.api_key_encrypted
                config["key_preview"] = f"...{encrypted[-4:]}" if len(encrypted) >= 4 else "***"

            result.append(config)

        return result

    def delete_provider(self, user_id: UUID, provider_type: str) -> bool:
        """
        Delete a provider configuration for a user.

        Args:
            user_id: User UUID
            provider_type: Provider type

        Returns:
            bool: True if deleted, False if not found
        """
        deleted = self.repository.delete_by_user_and_type(user_id, provider_type)
        if deleted:
            logger.info(f"Deleted {provider_type} provider for user {user_id}")
        return deleted

    async def validate_provider(self, user_id: UUID, provider_type: str) -> dict[str, Any]:
        """
        Validate provider credentials by attempting to fetch models.

        Args:
            user_id: User UUID
            provider_type: Provider type

        Returns:
            Dict: Validation result with 'valid' boolean and optional 'models' list
        """
        api_key = self.get_decrypted_api_key(user_id, provider_type)
        if not api_key:
            return {"valid": False, "error": "Provider not configured"}

        try:
            if provider_type == "openai":
                models = await self._fetch_openai_models(api_key)
                return {"valid": True, "models": models}
            elif provider_type == "gemini":
                # Gemini validation can be added here if needed
                return {"valid": True, "message": "Gemini provider configured"}
            elif provider_type == "anthropic":
                # Anthropic validation can be added here
                return {"valid": True, "message": "Anthropic provider configured"}
            else:
                return {"valid": False, "error": f"Unknown provider: {provider_type}"}
        except Exception as e:
            logger.error(f"Provider validation failed for {provider_type}: {e}")
            return {"valid": False, "error": str(e)}

    async def _fetch_openai_models(self, api_key: str) -> list[dict[str, Any]]:
        """
        Fetch available models from OpenAI API.

        Args:
            api_key: OpenAI API key

        Returns:
            List[Dict]: List of available models with metadata
        """
        try:
            import openai

            async_client_cls = getattr(openai, "AsyncOpenAI", None)
            if async_client_cls is not None:
                client = async_client_cls(api_key=api_key)
                models_response = await client.models.list()
            else:
                client = openai.OpenAI(api_key=api_key)
                models_response = await asyncio.to_thread(client.models.list)

            models: list[dict[str, Any]] = []
            for model in getattr(models_response, "data", None) or []:
                model_id = getattr(model, "id", None)
                if not model_id:
                    continue

                models.append(
                    {
                        "id": model_id,
                        "name": model_id,
                        "created": getattr(model, "created", None),
                        "owned_by": getattr(model, "owned_by", None),
                    }
                )

            # Sort newest first (fallback to id when timestamps are missing)
            models.sort(
                key=lambda item: (
                    int(item.get("created") or 0),
                    str(item.get("name") or ""),
                ),
                reverse=True,
            )

            logger.info(f"Fetched {len(models)} OpenAI models")
            return models

        except Exception as e:
            logger.error(f"Failed to fetch OpenAI models: {e}")
            raise ValueError(f"Failed to fetch OpenAI models: {e}")
