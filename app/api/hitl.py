from uuid import UUID

from fastapi import APIRouter, Query

from app.core.dependency_injection import AppAutoInjector
from app.core.exceptions import CustomHTTPException
from app.models.hitl_interrupt import HITLInterruptStatus
from app.repositories.hitl_interrupt import HITLInterruptRepository
from app.schemas.hitl import (
    HitlInterruptStateResponse,
    HitlScopeRuleState,
    HitlSettingsResponse,
    HitlSettingsUpdate,
)
from app.schemas.responses import ApiResponse
from app.services.hitl_settings_service import HitlSettingsService

# No router-level auth dependency: the auto-injected ``user_id: UUID`` resolves to
# ``Depends(get_current_user_id)`` (DI magic), which both authenticates the request
# and yields the caller's id. This mirrors the custom-agents router and lets the API
# test authenticate by overriding ``get_current_user_id`` alone.
router = APIRouter(prefix="/hitl", tags=["hitl"])


def _to_response(data: dict) -> HitlSettingsResponse:
    return HitlSettingsResponse(
        device_id=data["device_id"],
        master_enabled=data["master_enabled"],
        global_tools=data["global_tools"],
        servers=[HitlScopeRuleState(**r) for r in data["servers"]],
        tools=[HitlScopeRuleState(**r) for r in data["tools"]],
    )


@router.get("/settings")
@AppAutoInjector.auto_inject()
async def get_hitl_settings(
    hitl_settings_service: HitlSettingsService,
    user_id: UUID,
    device_id: str | None = Query(
        None,
        alias="deviceId",
        description="Registered client device whose editable HITL rules are requested.",
    ),
    device_id_snake: str | None = Query(None, alias="device_id", include_in_schema=False),
) -> ApiResponse[HitlSettingsResponse]:
    data = hitl_settings_service.get_settings(user_id, device_id=device_id or device_id_snake)
    return ApiResponse(success=True, message="HITL settings retrieved", data=_to_response(data))


@router.post("/settings")
@AppAutoInjector.auto_inject()
async def update_hitl_settings(
    payload: HitlSettingsUpdate,
    hitl_settings_service: HitlSettingsService,
    user_id: UUID,
    device_id: str | None = Query(None, alias="deviceId"),
    device_id_snake: str | None = Query(None, alias="device_id", include_in_schema=False),
) -> ApiResponse[HitlSettingsResponse]:
    items = [r.model_dump() for r in payload.items]
    data = hitl_settings_service.apply(user_id, device_id or device_id_snake, items)
    return ApiResponse(success=True, message="HITL settings updated", data=_to_response(data))


@router.delete("/settings")
@AppAutoInjector.auto_inject()
async def clear_hitl_setting(
    hitl_settings_service: HitlSettingsService,
    user_id: UUID,
    device_id: str | None = Query(None, alias="deviceId"),
    device_id_snake: str | None = Query(None, alias="device_id", include_in_schema=False),
    tool_origin: str | None = Query(None, alias="toolOrigin"),
    scope_type: str | None = Query(None, alias="scopeType"),
    scope_value: str | None = Query(None, alias="scopeValue"),
    scope_type_snake: str | None = Query(None, alias="scope_type", include_in_schema=False),
    scope_value_snake: str | None = Query(None, alias="scope_value", include_in_schema=False),
    tool_origin_snake: str | None = Query(None, alias="tool_origin", include_in_schema=False),
) -> ApiResponse[HitlSettingsResponse]:
    resolved_scope_type = scope_type or scope_type_snake
    resolved_scope_value = scope_value or scope_value_snake
    resolved_tool_origin = tool_origin or tool_origin_snake
    if not resolved_scope_type or not resolved_scope_value or not resolved_tool_origin:
        raise CustomHTTPException(
            422,
            "toolOrigin, scopeType, and scopeValue query parameters are required.",
            "HITL_SCOPE_PARAMS_REQUIRED",
        )
    data = hitl_settings_service.clear(
        user_id,
        device_id or device_id_snake,
        resolved_tool_origin,
        resolved_scope_type,
        resolved_scope_value,
    )
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
