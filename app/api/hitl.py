from uuid import UUID

from fastapi import APIRouter, Query

from app.core.dependency_injection import AppAutoInjector
from app.core.exceptions import CustomHTTPException
from app.repositories.hitl_interrupt import HITLInterruptRepository
from app.schemas.hitl import (
    HitlInterruptStateResponse,
    HitlScopeRule,
    HitlSettingsResponse,
    HitlSettingsUpdate,
)
from app.schemas.responses import ApiResponse
from app.services.hitl_settings_service import HitlSettingsService
from app.models.hitl_interrupt import HITLInterruptStatus

# No router-level auth dependency: the auto-injected ``user_id: UUID`` resolves to
# ``Depends(get_current_user_id)`` (DI magic), which both authenticates the request
# and yields the caller's id. This mirrors the custom-agents router and lets the API
# test authenticate by overriding ``get_current_user_id`` alone.
router = APIRouter(prefix="/hitl", tags=["hitl"])


def _to_response(data: dict) -> HitlSettingsResponse:
    return HitlSettingsResponse(
        master_enabled=data["master_enabled"],
        global_tools=data["global_tools"],
        servers=[HitlScopeRule(**r) for r in data["servers"]],
        tools=[HitlScopeRule(**r) for r in data["tools"]],
    )


@router.get("/settings")
@AppAutoInjector.auto_inject()
async def get_hitl_settings(
    hitl_settings_service: HitlSettingsService,
    user_id: UUID,
) -> ApiResponse[HitlSettingsResponse]:
    data = hitl_settings_service.get_settings(user_id)
    return ApiResponse(success=True, message="HITL settings retrieved", data=_to_response(data))


@router.post("/settings")
@AppAutoInjector.auto_inject()
async def update_hitl_settings(
    payload: HitlSettingsUpdate,
    hitl_settings_service: HitlSettingsService,
    user_id: UUID,
) -> ApiResponse[HitlSettingsResponse]:
    items = [r.model_dump() for r in payload.items]
    data = hitl_settings_service.apply(user_id, items)
    return ApiResponse(success=True, message="HITL settings updated", data=_to_response(data))


@router.delete("/settings")
@AppAutoInjector.auto_inject()
async def clear_hitl_setting(
    hitl_settings_service: HitlSettingsService,
    user_id: UUID,
    scope_type: str = Query(...),
    scope_value: str = Query(...),
) -> ApiResponse[HitlSettingsResponse]:
    data = hitl_settings_service.clear(user_id, scope_type, scope_value)
    return ApiResponse(success=True, message="HITL setting cleared", data=_to_response(data))


@router.get("/interrupts/{interrupt_id}")
@AppAutoInjector.auto_inject()
async def get_hitl_interrupt_state(
    interrupt_id: str,
    hitl_interrupt_repository: HITLInterruptRepository,
    user_id: UUID,
) -> ApiResponse[HitlInterruptStateResponse]:
    interrupt = hitl_interrupt_repository.get_by_id_for_user(interrupt_id, user_id)
    if interrupt is None:
        raise CustomHTTPException(404, "HITL interrupt not found.", "INTERRUPT_NOT_FOUND")

    if interrupt.status == HITLInterruptStatus.PENDING:
        hitl_interrupt_repository.expire_pending_if_due_for_user(interrupt_id, user_id)
        interrupt = hitl_interrupt_repository.get_by_id_for_user(interrupt_id, user_id)
        if interrupt is None:
            raise CustomHTTPException(404, "HITL interrupt not found.", "INTERRUPT_NOT_FOUND")

    return ApiResponse(
        success=True,
        message="HITL interrupt state retrieved",
        data=HitlInterruptStateResponse(
            interrupt_id=interrupt.id,
            conversation_id=interrupt.conversation_id,
            status=interrupt.status.value,
            expires_at=interrupt.expires_at,
            updated_at=interrupt.updated_at,
        ),
    )
