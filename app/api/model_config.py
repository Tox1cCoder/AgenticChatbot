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
from pydantic import BaseModel, Field, RootModel

from app.core.auth import get_current_user
from app.core.dependency_injection import AppAutoInjector
from app.models.user import User
from app.schemas.responses import ApiResponse
from app.services.model_config_service import ModelConfigService

router = APIRouter(prefix="/model-config", tags=["model-config"])


class AgentModelConfigPatch(BaseModel):
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


class ModelConfigUpdateRequest(RootModel[dict[str, AgentModelConfigPatch]]):
    """
    Request body is a partial mapping of agent_key -> config patch.

    Example:
      {
        "chat": {"provider": "openai", "model": "gpt-4o", "temperature": 0.7}
      }
    """


@router.get("", response_model=ApiResponse[dict[str, dict[str, Any]]])
@AppAutoInjector.auto_inject()
async def get_model_config(
    model_config_service: ModelConfigService,
    current_user: User = Depends(get_current_user),
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
        )


@router.patch("", response_model=ApiResponse[dict[str, dict[str, Any]]])
@AppAutoInjector.auto_inject()
async def patch_model_config(
    request: ModelConfigUpdateRequest,
    model_config_service: ModelConfigService,
    current_user: User = Depends(get_current_user),
) -> ApiResponse[dict[str, dict[str, Any]]]:
    try:
        updates = {
            agent_key: patch.model_dump(exclude_none=True)
            for agent_key, patch in (request.root or {}).items()
        }
        config = model_config_service.patch_configs(current_user.id, updates)
        return ApiResponse(
            success=True,
            message="Model config updated successfully",
            data=config,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update model config: {str(e)}",
        )


@router.post("/reset", response_model=ApiResponse[dict[str, dict[str, Any]]])
@AppAutoInjector.auto_inject()
async def reset_model_config(
    model_config_service: ModelConfigService,
    current_user: User = Depends(get_current_user),
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
        )
