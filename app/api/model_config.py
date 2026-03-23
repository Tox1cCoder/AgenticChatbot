"""
API endpoints for persistent per-agent model configuration.

Allows users to store provider/model/temperature selection for:
- chat
- rag
- search
- planning
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, RootModel

from app.core.auth import get_current_user
from app.core.dependency_injection import AppAutoInjector
from app.models.user import User
from app.schemas.responses import ApiResponse
from app.services.model_config_service import ModelConfigService
from app.utils.case_conversion import to_camel_case as to_camel

router = APIRouter(prefix="/model-config", tags=["model-config"])


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class AgentModelConfigPatch(CamelModel):
    provider: str | None = Field(
        default=None,
        description="Provider to use for this agent",
        pattern="^(gemini|openai)$",
    )
    model: str | None = Field(
        default=None,
        description="Model identifier (provider-specific)",
    )
    temperature: float | None = Field(
        default=None,
        description="Sampling temperature",
    )
    allow_custom_model: bool | None = Field(
        default=None,
        description="Allow saving a non-catalog custom model identifier for this agent",
    )


class ModelConfigUpdateRequest(RootModel[dict[str, AgentModelConfigPatch]]):
    """
    Request body is a partial mapping of agent_key -> config patch.

    Example:
      {
        "chat": {"provider": "openai", "model": "gpt-4o", "temperature": 0.7}
      }
    """


class ProviderModelOption(CamelModel):
    id: str
    display_name: str
    provider_type: str
    supports_vision: bool = False
    supports_tool_calling: bool = False
    supports_streaming: bool = False
    supports_reasoning: bool = False
    recommended: bool = False


class ProviderOptionsSnapshot(CamelModel):
    provider_type: str
    configured: bool
    key_source: str
    sync_status: str
    last_synced_at: str | None = None
    sync_error: str | None = None
    models: list[ProviderModelOption] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AgentModelConfigSnapshot(CamelModel):
    provider: str
    model: str | None = None
    temperature: float = 1.0
    source: str = "default"
    warnings: list[str] = Field(default_factory=list)
    key_source: str | None = None
    is_custom_model: bool = False


class ModelConfigOptionsSnapshot(CamelModel):
    providers: list[ProviderOptionsSnapshot] = Field(default_factory=list)
    agent_config: dict[str, AgentModelConfigSnapshot] = Field(default_factory=dict)


@router.get("", response_model=ApiResponse[dict[str, dict[str, Any]]])
@AppAutoInjector.auto_inject()
async def get_model_config(
    model_config_service: ModelConfigService,
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> ApiResponse[dict[str, dict[str, Any]]]:
    try:
        config = model_config_service.get_effective_model_config(current_user.id)
        return ApiResponse(
            success=True,
            message="Model config retrieved successfully",
            data=config,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model config: {str(e)}",
        ) from e


@router.get("/options", response_model=ApiResponse[ModelConfigOptionsSnapshot])
@AppAutoInjector.auto_inject()
async def get_model_config_options(
    model_config_service: ModelConfigService,
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> ApiResponse[ModelConfigOptionsSnapshot]:
    try:
        snapshot = await model_config_service.get_model_config_options(current_user.id)
        response_data = ModelConfigOptionsSnapshot.model_validate(snapshot)
        return ApiResponse(
            success=True,
            message="Model config options retrieved successfully",
            data=response_data,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model config options: {str(e)}",
        ) from e


@router.patch("", response_model=ApiResponse[dict[str, dict[str, Any]]])
@AppAutoInjector.auto_inject()
async def patch_model_config(
    request: ModelConfigUpdateRequest,
    model_config_service: ModelConfigService,
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> ApiResponse[dict[str, dict[str, Any]]]:
    try:
        updates = {
            agent_key: patch.model_dump(exclude_none=True)
            for agent_key, patch in (request.root or {}).items()
        }
        config = await model_config_service.patch_configs(current_user.id, updates)
        return ApiResponse(
            success=True,
            message="Model config updated successfully",
            data=config,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update model config: {str(e)}",
        ) from e


@router.post("/reset", response_model=ApiResponse[dict[str, dict[str, Any]]])
@AppAutoInjector.auto_inject()
async def reset_model_config(
    model_config_service: ModelConfigService,
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> ApiResponse[dict[str, dict[str, Any]]]:
    try:
        model_config_service.reset_configs(current_user.id)
        config = model_config_service.get_effective_model_config(current_user.id)
        return ApiResponse(
            success=True,
            message="Model config reset to defaults",
            data=config,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset model config: {str(e)}",
        ) from e
