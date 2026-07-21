"""Authenticated model-usage analytics endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends

from app.core.config import settings
from app.core.dependency_injection import AppAutoInjector
from app.core.exceptions import ResourceNotFoundException
from app.interfaces.model_usage_service_interface import IModelUsageService
from app.schemas.model_usage import (
    ConversationUsageQueryParams,
    ConversationUsageResponse,
    UsageDashboard,
    UsageDashboardQueryParams,
)
from app.schemas.responses import ApiResponse


def _require_usage_ui_enabled() -> None:
    if not settings.model_usage_ui_enabled:
        raise ResourceNotFoundException()


router = APIRouter(
    prefix="/usage",
    tags=["usage"],
    dependencies=[Depends(_require_usage_ui_enabled)],
)


@router.get("/dashboard", response_model=ApiResponse[UsageDashboard])
@AppAutoInjector.auto_inject()
def get_usage_dashboard(
    query: UsageDashboardQueryParams,
    model_usage_service: IModelUsageService,
    user_id: UUID,
) -> ApiResponse[UsageDashboard]:
    """Return usage analytics scoped to the authenticated user."""
    result = model_usage_service.get_dashboard(user_id=user_id, query=query)
    return ApiResponse(success=True, message="Usage dashboard retrieved", data=result)


@router.get(
    "/conversations/{conversation_id}",
    response_model=ApiResponse[ConversationUsageResponse],
)
@AppAutoInjector.auto_inject()
def get_conversation_usage(
    conversation_id: UUID,
    query: ConversationUsageQueryParams,
    model_usage_service: IModelUsageService,
    user_id: UUID,
) -> ApiResponse[ConversationUsageResponse]:
    """Return analytics for one conversation owned by the authenticated user."""
    result = model_usage_service.get_conversation_usage(
        user_id=user_id,
        conversation_id=conversation_id,
        query=query,
    )
    return ApiResponse(success=True, message="Conversation usage retrieved", data=result)
