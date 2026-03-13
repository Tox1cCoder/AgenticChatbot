"""
API endpoints for AI provider management.

Handles CRUD operations for user-specific AI provider configurations with encrypted API key storage.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.auth import get_current_user
from app.core.dependency_injection import AppAutoInjector
from app.models.user import User
from app.schemas.responses import ApiResponse
from app.services.provider_service import ProviderService

router = APIRouter(prefix="/providers", tags=["providers"])


# Request/Response Schemas
class ProviderAddRequest(BaseModel):
    """Request to add or update a provider."""

    provider_type: str = Field(
        ...,
        description="Provider type: 'gemini', 'openai', or 'anthropic'",
        pattern="^(gemini|openai|anthropic)$",
    )
    api_key: str = Field(
        ..., min_length=1, description="API key for the provider (will be encrypted)"
    )
    is_default: bool = Field(default=False, description="Set as default provider for this user")
    provider_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional provider-specific metadata (e.g., organization ID)",
    )


class ProviderResponse(BaseModel):
    """Provider configuration response (without API key)."""

    id: str
    provider_type: str
    is_default: bool
    created_at: str
    provider_metadata: dict[str, Any]
    key_preview: str = Field(
        default="***", description="Last 4 characters of encrypted key for verification"
    )


class ProviderValidationResponse(BaseModel):
    """Provider validation result."""

    valid: bool
    models: list[dict[str, Any]] = Field(default_factory=list)
    message: str = Field(default="")
    error: str = Field(default="")


@router.post(
    "",
    response_model=ApiResponse[ProviderResponse],
    status_code=status.HTTP_201_CREATED,
)
@AppAutoInjector.auto_inject()
async def add_provider(
    request: ProviderAddRequest,
    provider_service: ProviderService,
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> ApiResponse[ProviderResponse]:
    """
    Add or update an AI provider configuration.

    Stores the API key encrypted in the database. If a provider of the same type
    already exists for the user, it will be updated (upsert operation).
    """
    try:
        provider = provider_service.add_provider(
            user_id=current_user.id,
            provider_type=request.provider_type,
            api_key=request.api_key,
            is_default=request.is_default,
            provider_metadata=request.provider_metadata,
        )

        response_data = ProviderResponse(
            id=str(provider.id),
            provider_type=provider.provider_type,
            is_default=provider.is_default,
            created_at=provider.created_at.isoformat(),
            provider_metadata=provider.provider_metadata or {},
            key_preview=(
                f"...{provider.api_key_encrypted[-4:]}"
                if len(provider.api_key_encrypted) >= 4
                else "***"
            ),
        )

        return ApiResponse(
            success=True,
            message=f"{request.provider_type} provider added/updated successfully",
            data=response_data,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add provider: {str(e)}",
        ) from e


@router.get("", response_model=ApiResponse[list[ProviderResponse]])
@AppAutoInjector.auto_inject()
async def list_providers(
    provider_service: ProviderService,
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> ApiResponse[list[ProviderResponse]]:
    """
    List all provider configurations for the current user.

    Returns provider information without decrypted API keys.
    Shows last 4 characters of encrypted key for verification.
    """
    try:
        providers = provider_service.get_all_providers(
            user_id=current_user.id,
            include_encrypted=True,
        )

        response_data = [
            ProviderResponse(
                id=p["id"],
                provider_type=p["provider_type"],
                is_default=p["is_default"],
                created_at=p["created_at"],
                provider_metadata=p.get("provider_metadata", {}),
                key_preview=p.get("key_preview", "***"),
            )
            for p in providers
        ]

        return ApiResponse(
            success=True,
            message="Providers retrieved successfully",
            data=response_data,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list providers: {str(e)}",
        ) from e


@router.get("/{provider_type}", response_model=ApiResponse[ProviderResponse])
@AppAutoInjector.auto_inject()
async def get_provider(
    provider_type: str,
    provider_service: ProviderService,
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> ApiResponse[ProviderResponse]:
    """
    Get a specific provider configuration.

    Returns provider information without decrypted API key.
    """
    try:
        config = provider_service.get_provider_config(
            user_id=current_user.id,
            provider_type=provider_type,
            decrypt=False,
        )

        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"{provider_type} provider not found",
            )

        response_data = ProviderResponse(
            id=config["id"],
            provider_type=config["provider_type"],
            is_default=config["is_default"],
            created_at=config["created_at"],
            provider_metadata=config.get("provider_metadata", {}),
        )

        return ApiResponse(
            success=True,
            message="Provider retrieved successfully",
            data=response_data,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get provider: {str(e)}",
        ) from e


@router.delete("/{provider_type}", response_model=ApiResponse[dict[str, str]])
@AppAutoInjector.auto_inject()
async def delete_provider(
    provider_type: str,
    provider_service: ProviderService,
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> ApiResponse[dict[str, str]]:
    """
    Delete a provider configuration.

    Soft deletes the provider, removing access to the encrypted API key.
    """
    try:
        deleted = provider_service.delete_provider(
            user_id=current_user.id,
            provider_type=provider_type,
        )

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"{provider_type} provider not found",
            )

        return ApiResponse(
            success=True,
            message=f"{provider_type} provider deleted successfully",
            data={"provider_type": provider_type},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete provider: {str(e)}",
        ) from e


@router.post("/{provider_type}/validate", response_model=ApiResponse[ProviderValidationResponse])
@AppAutoInjector.auto_inject()
async def validate_provider(
    provider_type: str,
    provider_service: ProviderService,
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> ApiResponse[ProviderValidationResponse]:
    """
    Validate provider credentials and fetch available models.

    Tests the stored API key by attempting to list models from the provider's API.
    For OpenAI, returns a list of available models.
    """
    try:
        result = await provider_service.validate_provider(
            user_id=current_user.id,
            provider_type=provider_type,
        )

        response_data = ProviderValidationResponse(
            valid=result.get("valid", False),
            models=result.get("models", []),
            message=result.get("message", ""),
            error=result.get("error", ""),
        )

        return ApiResponse(
            success=result.get("valid", False),
            message=("Provider validated" if result.get("valid") else "Provider validation failed"),
            data=response_data,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate provider: {str(e)}",
        ) from e


@router.get("/{provider_type}/models", response_model=ApiResponse[list[dict[str, Any]]])
@AppAutoInjector.auto_inject()
async def list_provider_models(
    provider_type: str,
    provider_service: ProviderService,
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> ApiResponse[list[dict[str, Any]]]:
    """
    List available models from a provider.

    Fetches the list of models available from the provider's API using
    the stored API key.
    """
    try:
        result = await provider_service.validate_provider(
            user_id=current_user.id,
            provider_type=provider_type,
        )

        if not result.get("valid"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "Provider validation failed"),
            )

        models = result.get("models", [])

        return ApiResponse(
            success=True,
            message=f"Retrieved {len(models)} models from {provider_type}",
            data=models,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}",
        ) from e
