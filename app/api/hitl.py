from uuid import UUID

from fastapi import APIRouter, Query

from app.core.dependency_injection import AppAutoInjector
from app.schemas.hitl import HitlScopeRule, HitlSettingsResponse, HitlSettingsUpdate
from app.schemas.responses import ApiResponse
from app.services.hitl_settings_service import HitlSettingsService

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
